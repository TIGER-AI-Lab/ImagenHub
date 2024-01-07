# Human Evaluation


## Human Evaluation Standardization

To standardize the conduction of a rigorous human evaluation, we stipulate the criteria for each measurement as follows:

### Measurement Criteria

- **Semantic Consistency (SC)**
  - **Score in Range:** [0, 0.5, 1]
  - **Description:** It measures the level that the generated image is coherent in terms of the condition provided (i.e. Prompts, Subject Token, etc.).

- **Perceptual Quality (PQ)**
  - **Score in Range:** [0, 0.5, 1]
  - **Description:** It measures the level at which the generated image is visually convincing and gives off a natural sense.

### Meaning of Semantic Consistency (SC) score

- **SC=0**: Image not following one or more of the conditions at all (e.g. not following the prompt at all, different background in editing task, wrong subject in subject-driven task, etc.)
- **SC=0.5**: all the conditions are partly following the requirements.
- **SC=1**: The rater agrees that the overall idea is correct.

### Meaning of Perceptual Quality (PQ) score

- **PQ=0**: The rater spotted obvious distortion or artifacts at first glance and those distorts make the objects unrecognizable.
- **PQ=0.5**: The rater found out the image gives off an unnatural sense. Or the rater spotted some minor artifacts and the objects are still recognizable.
- **PQ=1**: The rater agrees that the resulting image looks genuine.

**Artifacts and Unusual sense, respectively, are:**

- **Artifacts**:
  - Distortion
  - Watermark
  - Scratches
  - Blurred faces
  - Unusual body parts
  - Subjects not harmonized

- **Unusual Sense**:
  - Wrong sense of distance (subject too big or too small compared to others)
  - Wrong shadow
  - Wrong lighting, etc.

## Implementation of Human Evaluation

* In execute, we require raters to strictly follow this table for rating.

Each image is rated as a list value `[SC, PQ]`.

### SC Rating Table

| Condition 1           | Condition 2 (if applicable)   | Condition 3 (if applicable)   | SC rating |
|-----------------------|------------------------------|------------------------------|-----------|
| no following at all   | Any                          | Any                          | 0         |
| Any                   | no following at all          | Any                          | 0         |
| Any                   | Any                          | no following at all          | 0         |
| following some part   | following some or most part  | following some or most part  | 0.5       |
| following some or most part | following some part         | following some or most part  | 0.5       |
| following some part or more | following some or most part | following some part         | 0.5       |
| following most part   | following most part          | following most part          | 1         |

### PQ Rating Table

| Objects in image  | Artifacts | Unusual sense   | PQ rating |
|-------------------|-----------|-----------------|-----------|
| Unrecognizable    | serious   | Any             | 0         |
| Recognizable      | some      | Any             | 0.5       |
| Recognizable      | Any       | some            | 0.5       |
| Recognizable      | none      | little or None  | 1         |

### Collecting Human Evaluation Data

In the `results` folder, there should be a `dataset_lookup.csv` file to edit.

| uid  | TheModel | 
|-------------------|-----------|
| sample_1.jpg    | [0, 1]   |
| sample_2.jpg      | [1, 1]      |
| sample_3.jpg      | [1, 0.5]       |
| ...      | ...      |

## Task-specific Guidelines

Some predefined conditions:
* Condition A: Is the Image generation following the prompt?
* Condition B: Is the Image editing following the instruction?
* Condition C: Is the Image performing minimal edit without changing the background? 
* Condition D_i: Is the Object_i in the Image following the token subject?
* Condition E: Is the Image following the control guidance?

### Text-guided Image Generation (known as Text-to-Image)

* Condition A: Is the Image generation following the prompt?

SC Table:
| Condition A           | SC rating |
|-----------------------|-----------|
| no following at all   | 0         |
| following some part   | 0.5       |
| following most part   | 1         |

PQ Table:
| Objects in image  | Artifacts | Unusual sense   | PQ rating |
|-------------------|-----------|-----------------|-----------|
| Unrecognizable    | serious   | Any             | 0         |
| Recognizable      | some      | Any             | 0.5       |
| Recognizable      | Any       | some            | 0.5       |
| Recognizable      | none      | little or None  | 1         |


### Mask-guided Image Editing 

* Condition B: Is the Image editing following the instruction?
* Condition C: Is the Image performing minimal edit without changing the background? 

SC Table:
| Condition B           | Condition C   | SC rating |
|-----------------------|------------------------------|-----------|
| no following at all   | Any                          | 0         |
| Any                   | background completely changed          | 0         |
| following some part   | with a few overedit or mostly minimal  | 0.5       |
| following some or most part | with a few overedit | 0.5       |
| following most part   | mostly minimal          | 1         |

PQ Table:
| Objects in image  | Artifacts | Unusual sense   | PQ rating |
|-------------------|-----------|-----------------|-----------|
| Unrecognizable    | serious   | Any             | 0         |
| Recognizable      | some      | Any             | 0.5       |
| Recognizable      | Any       | some            | 0.5       |
| Recognizable      | none      | little or None  | 1         |


### Text-guided Image Editing


* Condition B: Is the Image editing following the instruction?
* Condition C: Is the Image performing minimal edit without changing the background? 

SC Table:
| Condition B           | Condition C   | SC rating |
|-----------------------|------------------------------|-----------|
| no following at all   | Any                          | 0         |
| Any                   | background completely changed          | 0         |
| following some part   | with a few overedit or mostly minimal  | 0.5       |
| following some or most part | with a few overedit | 0.5       |
| following most part   | mostly minimal          | 1         |

PQ Table:
| Objects in image  | Artifacts | Unusual sense   | PQ rating |
|-------------------|-----------|-----------------|-----------|
| Unrecognizable    | serious   | Any             | 0         |
| Recognizable      | some      | Any             | 0.5       |
| Recognizable      | Any       | some            | 0.5       |
| Recognizable      | none      | little or None  | 1         |

### Subject-driven Image Generation


* Condition A: Is the Image generation following the prompt?
* Condition D: Is the Object in the Image following the token subject?

SC Table:
| Condition A           | Condition D   | SC rating |
|-----------------------|------------------------------|-----------|
| no following at all   | Any                          | 0         |
| Any                   | no following at all          | 0         |
| following some part   | following some or most part  | 0.5       |
| following some or most part | following some part | 0.5       |
| following most part   | following most part          | 1         |

PQ Table:
| Objects in image  | Artifacts | Unusual sense   | PQ rating |
|-------------------|-----------|-----------------|-----------|
| Unrecognizable    | serious   | Any             | 0         |
| Recognizable      | some      | Any             | 0.5       |
| Recognizable      | Any       | some            | 0.5       |
| Recognizable      | none      | little or None  | 1         |

### Subject-driven Image Editing

* Condition C: Is the Image performing minimal edit without changing the background? 
* Condition D: Is the Object in the Image following the token subject?

SC Table:
| Condition C           | Condition D   | SC rating |
|-----------------------|------------------------------|-----------|
| no following at all   | Any                          | 0         |
| Any                   | no following at all          | 0         |
| following some part   | following some or most part  | 0.5       |
| following some or most part | following some part | 0.5       |
| following most part   | following most part          | 1         |

PQ Table:
| Objects in image  | Artifacts | Unusual sense   | PQ rating |
|-------------------|-----------|-----------------|-----------|
| Unrecognizable    | serious   | Any             | 0         |
| Recognizable      | some      | Any             | 0.5       |
| Recognizable      | Any       | some            | 0.5       |
| Recognizable      | none      | little or None  | 1         |


### Multi-concept Image Composition 

* Condition A: Is the Image generation following the prompt?
* Condition D_i: Is the Object_i in the Image following the token subject?

SC Table:
| Condition A           | Condition D_1   | Condition D_2   | SC rating |
|-----------------------|------------------------------|------------------------------|-----------|
| no following at all   | Any                          | Any                          | 0         |
| Any                   | no following at all          | Any                          | 0         |
| Any                   | Any                          | no following at all          | 0         |
| following some part   | following some or most part  | following some or most part  | 0.5       |
| following some or most part | following some part         | following some or most part  | 0.5       |
| following some part or more | following some or most part | following some part         | 0.5       |
| following most part   | following most part          | following most part          | 1         |

PQ Table:
| Objects in image  | Artifacts | Unusual sense   | PQ rating |
|-------------------|-----------|-----------------|-----------|
| Unrecognizable    | serious   | Any             | 0         |
| Recognizable      | some      | Any             | 0.5       |
| Recognizable      | Any       | some            | 0.5       |
| Recognizable      | none      | little or None  | 1         |


### Control-guided Image Generation 

* Condition A: Is the Image generation following the prompt?
* Condition E: Is the Image following the control guidance?

SC Table:
| Condition A           | Condition E   | SC rating |
|-----------------------|------------------------------|-----------|
| no following at all   | Any                          | 0         |
| Any                   | no following at all          | 0         |
| following some part   | following some or most part  | 0.5       |
| following some or most part | following some part | 0.5       |
| following most part   | following most part          | 1         |

PQ Table:
| Objects in image  | Artifacts | Unusual sense   | PQ rating |
|-------------------|-----------|-----------------|-----------|
| Unrecognizable    | serious   | Any             | 0         |
| Recognizable      | some      | Any             | 0.5       |
| Recognizable      | Any       | some            | 0.5       |
| Recognizable      | none      | little or None  | 1         |


## Statistical Tools for Human Evaluation Data

(Under Construction)