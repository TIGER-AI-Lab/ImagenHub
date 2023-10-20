from .styledrop_src import open_clip, taming, utils
from .styledrop_src.libs.muse import MUSE
from .styledrop_src.custom.custom_dataset import test_custom_dataset,train_custom_dataset,Discriptor
import torch
from torch import multiprocessing as mp
import accelerate
import ml_collections
from loguru import logger
from tqdm import tqdm
import os
import sys
import einops
import random
import numpy as np
from torchvision.utils import make_grid, save_image
from torch.utils._pytree import tree_map
import time, datetime
logging = logger

def LSimple(x0, nnet, schedule, **kwargs):
    labels, masked_ids = schedule.sample(x0)
    logits = nnet(masked_ids, **kwargs, use_adapter=True)
    # b (h w) c, b (h w)
    loss = schedule.loss(logits, labels)
    return loss

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
def d(**kwargs):
    """Helper of creating a config dict."""
    return ml_collections.ConfigDict(initial_dictionary=kwargs)

def get_config(outdir_for_train=os.path.join('checkpoints','StyleDrop')):
    config = ml_collections.ConfigDict()
    
    config.seed = 1234
    config.z_shape = (8, 16, 16)

    config.autoencoder = d(
        config_file='vq-f16-jax.yaml',
    )

    # data for training
    config.data_path="data/one_style.json"
    config.resume_root="assets/ckpts/cc3m-285000.ckpt"
    
    config.adapter_path=None
    config.sample_interval=True
    config.train = d(
        n_steps=1000,
        batch_size=8,
        log_interval=20,
        eval_interval=100,
        save_interval=100,
        fid_interval=20000,
        num_workers=8,
        resampled=False,
    )

    config.optimizer = d(
        name='adamw',
        lr=0.0003,
        weight_decay=0.03,
        betas=(0.99, 0.99),
    )

    config.lr_scheduler = d(
        name='customized',
        warmup_steps=-1, # 5000
    )

    config.nnet = d(
        name='uvit_t2i_vq',
        img_size=16,
        codebook_size=1024,
        in_chans=4,
        embed_dim=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=False,
        clip_dim=1280,
        num_clip_token=77,
        use_checkpoint=False,
        skip=True,
        d_prj=32,# Stage I: 32; Stage II: TODO 
        is_shared=False, # Stage I: False; Stage II: False
    )

    config.muse = d(
        ignore_ind=-1,
        smoothing=0.1,
        gen_temp=4.5
    )

    config.sample = d(
        sample_steps=36,
        n_samples=50,
        mini_batch_size=8,
        cfg=True,
        linear_inc_scale=True,
        scale=10.,
        path='',
        lambdaA=2.0, # Stage I: 2.0; Stage II: TODO
        lambdaB=5.0, # Stage I: 5.0; Stage II: TODO
    )

    config.workdir = outdir_for_train
    stage = "_I" if config.nnet.is_shared else "_II"
    config.ckpt_root = os.path.join(config.workdir, 'ckpts'+stage)
    config.sample_dir = os.path.join(config.workdir, 'samples'+stage)

    return config

class StyleDropPipeline():
    def __init__(self, device="cuda") -> None:
        self.device = device
        self.config = get_config()

        # for inference
        self.empty_context = self.extract_empty_feature()
        self.muse = None
        self.prompt_model = None
        self.vq_model = None
        self.tokenizer = None
        self.nnet_ema = None
        self.style_adapter_weight = None
        self.adapter_set = False


    def extract_empty_feature(self):
        prompts = ['',]
        model, _, _ = open_clip.create_model_and_transforms('ViT-bigG-14', 'laion2b_s39b_b160k')
        model = model.to(self.device)
        model.eval()
        tokenizer = open_clip.get_tokenizer('ViT-bigG-14')

        text_tokens = tokenizer(prompts).to(self.device)
        latent = model.encode_text(text_tokens)
        del model
        del tokenizer
        c = latent[0].detach().cpu().float().numpy() # in shape torch.Size([1, 77, 1280])
        return c

    def train(self, config):
        if config.get('benchmark', False):
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False

        try:
            mp.set_start_method('spawn', force=True)
            print("spawned")
        except RuntimeError:
            pass

        accelerator = accelerate.Accelerator()
        device = accelerator.device
        accelerate.utils.set_seed(config.seed, device_specific=True)
        logging.info(f'Process {accelerator.process_index} using device: {device}')

        config.mixed_precision = accelerator.mixed_precision
        config = ml_collections.ConfigDict(config)

        assert config.train.batch_size % accelerator.num_processes == 0
        mini_batch_size = config.train.batch_size // accelerator.num_processes
        
        # Load open_clip and vq model
        prompt_model,_,_ = open_clip.create_model_and_transforms('ViT-bigG-14', 'laion2b_s39b_b160k')
        prompt_model = prompt_model.to(device)
        prompt_model.eval()
        tokenizer = open_clip.get_tokenizer('ViT-bigG-14')
        
        vq_model = taming.models.vqgan.get_model('vq-f16-jax.yaml')
        vq_model.eval()
        vq_model.requires_grad_(False)
        vq_model.to(device)
        
        if accelerator.is_main_process:
            os.makedirs(config.ckpt_root, exist_ok=True)
            os.makedirs(config.sample_dir, exist_ok=True)
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            logging.info(config)
        else:
            logging.remove()
            logger.add(sys.stderr, level='ERROR')
        logging.info(f'Run on {accelerator.num_processes} devices')

        ckpts = list(filter(lambda x: '.ckpt' in x, os.listdir(config.ckpt_root)))
        if not ckpts:
            resume_step = 0
        else:
            steps = map(lambda x: int(x.split(".")[0]), ckpts)
            resume_step = max(steps)

        logger.info(f'world size is {accelerator.num_processes}')



        dataset = train_custom_dataset(
            train_file=config.data_path,
        )
        test_dataset = test_custom_dataset(dataset.style)
        
        discriptor = Discriptor(dataset.style)
        
        train_dataset_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=config.train.batch_size,
        )
        test_dataset_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=10,
        )

        prompt_loader = torch.utils.data.DataLoader(
            dataset = discriptor,
            batch_size = config.sample.mini_batch_size,
        )

        autoencoder = taming.models.vqgan.get_model(**config.autoencoder)
        autoencoder.to(device)

        train_state = utils.initialize_train_state(config, device)
        lr_scheduler = train_state.lr_scheduler
        train_state.resume(config.resume_root,config.adapter_path)

        train_state.freeze()
        nnet, nnet_ema, optimizer = accelerator.prepare(
            train_state.nnet, train_state.nnet_ema, train_state.optimizer)
        

        
        @torch.cuda.amp.autocast()#type: ignore
        def encode(_batch):
            res = autoencoder.encode(_batch)[-1][-1].reshape(len(_batch), -1)
            return res

        @torch.cuda.amp.autocast()#type: ignore
        def decode(_batch):
            return autoencoder.decode_code(_batch)

        def get_data_generator():
            while True:
                for data in tqdm(train_dataset_loader, disable=not accelerator.is_main_process, desc='epoch'):
                    image, prompt = data
                    prompt = list(prompt)
                    text_tokens = tokenizer(prompt).to(device)
                    text_embedding = prompt_model.encode_text(text_tokens).detach().cpu()
                    
                    image = image.to(device)
                    image_embedding = vq_model(image)[-1][-1].detach().cpu()
                    image_embedding = image_embedding.unsqueeze(dim=0)
                    yield [image_embedding, text_embedding]

        data_generator = get_data_generator()

        def get_context_generator():
            while True:
                for data in test_dataset_loader:
                    _, _context = data
                    _context = list(_context)
                    text_tokens = tokenizer(_context).to(device)
                    _context = prompt_model.encode_text(text_tokens).detach().cpu()
                    yield _context

        context_generator = get_context_generator()

        def get_eval_context():
            while True:
                for data in prompt_loader:
                    prompt = list(data)
                    text_tokens = tokenizer(prompt).to(device)
                    embedding = prompt_model.encode_text(text_tokens).detach().cpu()
                    yield prompt,embedding
        prompt_generator = get_eval_context()
        
        def get_constant_context():

            for data in test_dataset_loader:
                prompt,_context = data
                _context = list(_context)
                break
            text_tokens = tokenizer(_context).to(device)
            embedding = prompt_model.encode_text(text_tokens).detach().cpu()
            return prompt,embedding
        
        muse = MUSE(codebook_size=autoencoder.n_embed, device=device, **config.muse)

        def cfg_nnet(x, context, scale=None,lambdaA=None,lambdaB=None):
            _cond = nnet_ema(x, context=context)
            _cond_w_adapter = nnet_ema(x,context=context,use_adapter=True)
            
            _empty_context = torch.tensor(dataset.empty_context, device=device)
            _empty_context = einops.repeat(_empty_context, 'L D -> B L D', B=x.size(0))
            _uncond = nnet_ema(x, context=_empty_context)
            res = _cond + scale * (_cond - _uncond)
            if lambdaA is not None:
                res = _cond_w_adapter + lambdaA*(_cond_w_adapter - _cond) + lambdaB*(_cond - _uncond)
            return res

        def train_step(_batch):
            _metrics = dict()
            optimizer.zero_grad()
            _z, context = proc_batch_feat(_batch)
            loss = LSimple(_z, nnet, muse, context=context)  # currently only support the extracted feature version
            metric_logger.update(loss=accelerator.gather(loss.detach()).mean())
            accelerator.backward(loss.mean())
            optimizer.step()
            lr_scheduler.step()
            train_state.ema_update(config.get('ema_rate', 0.9999))
            train_state.step += 1
            loss_scale, grad_norm = accelerator.scaler.get_scale(), utils.get_grad_norm_(nnet.parameters())  # type: ignore
            metric_logger.update(loss_scale=loss_scale)
            metric_logger.update(grad_norm=grad_norm)
            return dict(lr=train_state.optimizer.param_groups[0]['lr'],
                        **{k: v.value for k, v in metric_logger.meters.items()})

        def proc_batch_feat(_batch):
            _z = _batch[0].reshape(-1, 256)
            context = _batch[1].reshape(_z.shape[0], 77, -1)
            assert context.shape[-1] == config.nnet.clip_dim # type: ignore
            return _z, context

        def eval_step(n_samples, sample_steps):
            logging.info(f'eval_step: n_samples={n_samples}, sample_steps={sample_steps}'
                        f'mini_batch_size={config.sample.mini_batch_size}') # type: ignore

            def sample_fn(_n_samples):
                # _context = next(context_generator)
                prompt, _context = next(prompt_generator)
                _context = _context.to(device).reshape(-1, 77, config.nnet.clip_dim)
                kwargs = dict(context=_context)
                return muse.generate(config, _context.shape[0], cfg_nnet, decode,is_eval=True, **kwargs)

            if accelerator.is_main_process:
                path = f'{config.workdir}/eval_samples/{train_state.step}_{datetime.datetime.now().strftime("%m%d_%H%M%S")}'
                logging.info(f'Path for Eval images: {path}')
            else:
                path = None

            utils.sample2dir(accelerator, path, n_samples, config.sample.mini_batch_size, sample_fn,  # type: ignore
                            dataset.unpreprocess)

            return 0

        if eval_ckpt_path := os.getenv('EVAL_CKPT', ''):
            adapter_path = os.getenv('ADAPTER',None)
            nnet.eval()
            train_state.resume(eval_ckpt_path,adapter_path)
            logging.info(f'Eval {train_state.step}...')
            eval_step(n_samples=config.sample.n_samples, sample_steps=config.sample.sample_steps) # type: ignore
            return

        logging.info(f'Start fitting, step={train_state.step}, mixed_precision={config.mixed_precision}')
        step_fid = []
        metric_logger = utils.MetricLogger()
        cur_step = train_state.step
        while train_state.step < config.train.n_steps + cur_step:   # type: ignore
            nnet.train()
            data_time_start = time.time()
            batch = tree_map(lambda x: x.to(device), next(data_generator))
            metric_logger.update(data_time=time.time() - data_time_start)
            metrics = train_step(batch)

            nnet.eval()

            if train_state.step % config.train.save_interval == 0 or train_state.step == config.train.n_steps:   # type: ignore
                torch.cuda.empty_cache()
                logging.info(f'Save checkpoint {train_state.step}...')
                if accelerator.local_process_index == 0:
                    train_state.save(os.path.join(config.ckpt_root, f'{train_state.step}.ckpt'),adapter_only=True)
                    if config.train.eval_interval:
                        eval_step(n_samples=config.sample.n_samples, sample_steps=config.sample.sample_steps) # type: ignore
            accelerator.wait_for_everyone()

            if accelerator.is_main_process and train_state.step % config.train.log_interval == 0:   # type: ignore
                logger.info(f'step: {train_state.step} {metric_logger}')

            if train_state.step % config.train.eval_interval == 0:   # type: ignore
                torch.cuda.empty_cache()
                logging.info('Save a grid of images...')
                # contexts = torch.tensor(dataset.contexts, device=device)[: 2 * 5]
                prompt, contexts = get_constant_context()
                contexts = contexts.to(device)
                print(f"Eval prompt: {prompt[0]}")
                print(f"Shape of contexts :[{contexts.shape}]")
                samples = muse.generate(config, contexts.shape[0], cfg_nnet, decode, context=contexts)
                samples = make_grid(dataset.unpreprocess(samples), contexts.shape[0])
                save_image(samples, os.path.join(config.sample_dir, f'{train_state.step}_{accelerator.process_index}.png'))
                torch.cuda.empty_cache()
            accelerator.wait_for_everyone()

            if train_state.step % config.train.fid_interval == 0 or train_state.step == config.train.n_steps:   # type: ignore
                torch.cuda.empty_cache()
                logging.info(f'Eval {train_state.step}...')
                fid = eval_step(n_samples=config.eval.n_samples,   # type: ignore
                                sample_steps=config.eval.sample_steps)  # calculate fid of the saved checkpoint   # type: ignore
                step_fid.append((train_state.step, fid))
                torch.cuda.empty_cache()
            accelerator.wait_for_everyone()
            
            
            
        del metrics

    def set_inference(self,):
        prompt_model,_,_ = open_clip.create_model_and_transforms('ViT-bigG-14', 'laion2b_s39b_b160k')
        prompt_model = prompt_model.to(self.device)
        prompt_model.eval()
        self.prompt_model = prompt_model
        tokenizer = open_clip.get_tokenizer('ViT-bigG-14')
        self.tokenizer = tokenizer
        vq_model = taming.models.vqgan.get_model('vq-f16-jax.yaml')
        vq_model.eval()
        vq_model.requires_grad_(False)
        vq_model.to(self.device)
        self.vq_model = vq_model
        muse = MUSE(codebook_size=vq_model.n_embed, device=self.device, **self.config.muse)
        self.muse = muse
        train_state = utils.initialize_train_state(self.config, self.device)
        train_state.resume(ckpt_root=self.config.resume_root)
        nnet_ema = train_state.nnet_ema
        nnet_ema.eval()
        nnet_ema.requires_grad_(False)
        nnet_ema.to(self.device)
        self.nnet_ema = nnet_ema

    def decode(self, _batch):
        return self.vq_model.decode_code(_batch)

    def cfg_nnet(self, x, context, scale=None,lambdaA=None,lambdaB=None):
        _cond = self.nnet_ema(x, context=context)
        _cond_w_adapter = self.nnet_ema(x,context=context,use_adapter=True)
        _empty_context = torch.tensor(self.empty_context, device=self.device)
        _empty_context = einops.repeat(_empty_context, 'L D -> B L D', B=x.size(0))
        _uncond = self.nnet_ema(x, context=_empty_context)
        res = _cond + scale * (_cond - _uncond)
        if lambdaA is not None:
            res = _cond_w_adapter + lambdaA*(_cond_w_adapter - _cond) + lambdaB*(_cond - _uncond)
        return res

    def process(self, prompt, adapter_postfix, adapter_path=None, lambdaA=2.0,lambdaB=5.0,seed=42,sample_steps=36, num_samples=1):
        self.config.sample.lambdaA = lambdaA
        self.config.sample.lambdaB = lambdaB
        self.config.sample.sample_steps = sample_steps

        if adapter_path is not None:
            self.nnet_ema.adapter.load_state_dict(torch.load(adapter_path))
            print(f"loaded adapter {adapter_path}")
        else:
            self.config.sample.lambdaA=None
            self.config.sample.lambdaB=None
        
        # Encode prompt
        prompt = prompt+adapter_postfix
        text_tokens = self.tokenizer(prompt).to(self.device)
        text_embedding = self.prompt_model.encode_text(text_tokens)
        text_embedding = text_embedding.repeat(num_samples, 1, 1) # B 77 1280
        print(text_embedding.shape)
    
        print(f"lambdaA: {lambdaA}, lambdaB: {lambdaB}, sample_steps: {sample_steps}")
        if seed<0:
            seed = random.randint(0,65535)
        self.config.seed = seed
        print(f"seed: {seed}")
        set_seed(self.config.seed)
        res = self.muse.generate(self.config,num_samples,self.cfg_nnet,self.decode,is_eval=True,context=text_embedding)
        print(res.shape)
        res = (res*255+0.5).clamp_(0,255).permute(0,2,3,1).to('cpu',torch.uint8).numpy()
        im = [res[i] for i in range(num_samples)]
        return im
        