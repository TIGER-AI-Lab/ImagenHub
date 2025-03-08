# modified from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/util/html.py
import dominate
from dominate.tags import meta, h1, h2, h3, h4, h5, h6, table, tr, td, p, a, img, br, div, script, button, style
import os
from typing import Union, List

# VSCode's Live Server plugin support viewing HTML files if you are in a remote environment.


class HTML:
    """This HTML class allows us to save images and write texts into a single HTML file.

     It consists of functions such as <add_header> (add a text header to the HTML file),
     <add_images> (add a row of images to the HTML file), and <save> (save the HTML to the disk).
     It is based on Python library 'dominate', a Python library for creating and manipulating HTML documents using a DOM API.
    """

    def __init__(self, web_dir: str, title: str = "Hello World", refresh: int = 0):
        """
        Initialize an instance of the HTML class.

        Args:
            web_dir (str): Directory to store the generated webpage and images.
            title (str, optional): Title of the webpage. Defaults to "Hello World".
            refresh (int, optional): Interval (in seconds) for the webpage to auto-refresh. If 0, no refreshing. Defaults to 0.
        """
        self.title = title
        self.web_dir = web_dir
        if not os.path.exists(self.web_dir):
            os.makedirs(self.web_dir)

        self.doc = dominate.document(title=title)
        if refresh > 0:
            with self.doc.head:
                meta(http_equiv="refresh", content=str(refresh))
        
        # Add CSS for toggle buttons
        with self.doc.head:
            style("""
                .toggle-button {
                    padding: 8px 16px;
                    margin: 5px;
                    background-color: #4CAF50;
                    color: white;
                    border: none;
                    border-radius: 4px;
                    cursor: pointer;
                }
                .toggle-button:hover {
                    background-color: #45a049;
                }
                .toggle-button.active {
                    background-color: #3e8e41;
                }
                .toggle-container {
                    margin: 10px 0;
                    padding: 10px;
                    background-color: #f5f5f5;
                    border-radius: 4px;
                }
            """)

    def add_header(self, text, header_type=1):
        """
        Add a header to the HTML document.

        Args:
            text (str): Header text.
            header_type (int, optional): Header level (1-6). Defaults to 1.

        Raises:
            ValueError: If header_type is not in the range 1-6.
        """
        if header_type not in range(1, 7):
            raise ValueError("header_type can only ranging from 1 to 6")
        with self.doc:
            if (header_type == 1):
                h1(text)
            elif (header_type == 2):
                h2(text)
            elif (header_type == 3):
                h3(text)
            elif (header_type == 4):
                h4(text)
            elif (header_type == 5):
                h5(text)
            elif (header_type == 6):
                h6(text)

    def add_paragraph(self, text: str = ''):
        """
        Add a paragraph of text to the HTML document.

        Args:
            text (str, optional): The text content for the paragraph. Defaults to an empty string.
        """
        with self.doc:
            texts = text.split('\n')
            with p():
                from dominate.util import text
                for t in texts:
                    text(t)
                    br()

    def add_images(self, ims: Union[str, List], txts: Union[str, List], links: Union[str, List], width: int = 512):
        """
        Add a row of images to the HTML document.

        Args:
            ims (Union[str, List[str]]): List of image paths or a single image path.
            txts (Union[str, List[str]]): List of image captions or a single caption.
            links (Union[str, List[str]]): List of hyperlinks or a single hyperlink associated with the images.
            width (int, optional): Display width of the images in pixels. Defaults to 512.
        """
        self.t = table(
            border=1, style="table-layout: fixed;")  # Insert a table
        self.doc.add(self.t)
        with self.t:
            with tr():
                for im, txt, link in zip(ims, txts, links):
                    with td(style="word-wrap: break-word;", halign="center", valign="top", cls=f"column-{txt}"):
                        with p():
                            with a(href=os.path.join(link)):
                                img(style="width:%dpx" %
                                    width, src=os.path.join(im))
                            br()
                            p(txt)

    def add_toggle_buttons(self, folder_names: List[str]):
        """
        Add toggle buttons to show/hide specific columns in the HTML document.
        
        Args:
            folder_names (List[str]): List of folder names to create toggle buttons for.
        """
        with self.doc:
            with div(cls="toggle-container"):
                h3("Toggle Visibility")
                button("Show All", cls="toggle-button", onclick="showAll()")
                button("Hide All", cls="toggle-button", onclick="hideAll()")
                for folder in folder_names:
                    button(f"Toggle {folder}", cls="toggle-button active", 
                           id=f"btn-{folder}", onclick=f"toggleColumn('{folder}')")
            
            # Add JavaScript for toggling columns
            with script(type="text/javascript"):
                from dominate.util import raw
                js_code = """
                function toggleColumn(folderName) {
                    const columns = document.querySelectorAll(`.column-${folderName}`);
                    const button = document.getElementById(`btn-${folderName}`);
                    
                    let isVisible = !button.classList.contains('active');
                    
                    columns.forEach(col => {
                        col.style.display = isVisible ? 'table-cell' : 'none';
                    });
                    
                    if (isVisible) {
                        button.classList.add('active');
                    } else {
                        button.classList.remove('active');
                    }
                }
                
                function showAll() {
                    const buttons = document.querySelectorAll('.toggle-button');
                    buttons.forEach(btn => {
                        if (btn.id && btn.id.startsWith('btn-')) {
                            btn.classList.add('active');
                            const folderName = btn.id.replace('btn-', '');
                            const columns = document.querySelectorAll(`.column-${folderName}`);
                            columns.forEach(col => {
                                col.style.display = 'table-cell';
                            });
                        }
                    });
                }
                
                function hideAll() {
                    const buttons = document.querySelectorAll('.toggle-button');
                    buttons.forEach(btn => {
                        if (btn.id && btn.id.startsWith('btn-')) {
                            btn.classList.remove('active');
                            const folderName = btn.id.replace('btn-', '');
                            const columns = document.querySelectorAll(`.column-${folderName}`);
                            columns.forEach(col => {
                                col.style.display = 'none';
                            });
                        }
                    });
                }
                """
                raw(js_code)

    def save(self, filename='index.html'):
        """
        Save the constructed HTML content to a file.

        Args:
            filename (str, optional): Name of the HTML file. Defaults to 'index.html'.
        """
        html_file = os.path.join(self.web_dir, filename)
        f = open(html_file, 'wt')
        f.write(self.doc.render())
        f.close()


if __name__ == '__main__':  # we show an example usage here.
    html = HTML(web_dir=os.path.join(
        'results', 'test_html'), title='test_html')
    html.add_header('hello world')

    # Entries for one row
    ims, txts, links = [], [], []
    for n in range(4):
        ims.append('image_%d.png' % n)
        txts.append('text_%d' % n)
        links.append('image_%d.png' % n)

    # Adding toggle buttons for folder names
    html.add_toggle_buttons(['text_0', 'text_1', 'text_2', 'text_3'])
    
    # Adding mulitple rows
    html.add_paragraph('row1')
    html.add_images(ims, txts, links)
    html.add_paragraph('row2')
    html.add_images(ims, txts, links)
    html.add_paragraph('row3')
    html.add_images(ims, txts, links)
    html.add_paragraph('row4')
    html.add_images(ims, txts, links)
    html.save()
