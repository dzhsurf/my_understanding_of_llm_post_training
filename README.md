# My Understanding of LLM Post Training



## I. Introduction

Large Language Models (LLMs) are typically developed through a two-stage training process: Pre-Training and Post-Training. During Pre-Training, models absorb vast amounts of text using self-regressive tasks (such as predicting the next token) to grasp linguistic patterns and world knowledge. However, at this stage the “base model” exhibits several critical shortcomings:

- Inability to follow explicit instructions
- Tendency to generate harmful or inappropriate content
- Lack of adeptness in handling interactive dialogues



The Post-Training stage was specifically designed to remedy these issues. By aligning the model’s responses with human values and practical requirements, this phase transforms the base model into a safe, reliable, and user-friendly system capable of meeting real-world demands.



## II. Basic Concepts in Pre-Training and Post-Training

**1. Pre-Training**

Pre-Training involves leveraging massive, unannotated datasets to enable the model to learn the statistical patterns and basic semantics inherent to natural language. Methods typically include autoregressive techniques (as seen in the GPT series) or autoencoding approaches (such as those employed in the BERT series), where the model predicts subsequent tokens or reconstructs masked inputs. Although this produces a powerful base model with general language generation abilities, its responses are largely governed by statistical correlations rather than tailored to specific tasks or precise dialogue needs.



**2. Post-Training**

Post-Training refines the pre-trained model by employing further fine-tuning techniques aimed at specific applications—ranging from dialogue and summarization to question answering. This stage, often referred to as “model alignment,” ensures that the model’s output is not only factually correct, but also safe, ethical, and contextually appropriate. Particularly in conversational systems, post-training greatly enhances the model’s capacity to understand context, maintain coherence across interactions, and exhibit flexible, context-aware responses.



## III. Methods and Categories in Post-Training

The post-training process can be organized into two main categories: the methods used for parameter adjustments and the training objectives pursued.



**1. Based on Parameter Adjustment Methods**

- Full Fine-Tuning

  In this approach, every parameter of the model is updated. It is effective in environments with abundant training data and computational resources but comes at the cost of significant resource consumption and extended training times—especially for very large models.

- Parameter-Efficient Fine-Tuning (PEFT)

  Rather than updating the entirety of the model, PEFT adjusts only a small subset of parameters. This strategy greatly reduces memory and computation overhead while preserving the stability of most pre-trained weights. Common techniques include:

  - LoRA (Low-Rank Adaptation): Introduces low-rank matrices to capture the necessary parameter changes with efficiency and minimal overhead.
  - QLoRA: Extends the LoRA method to accommodate quantized models, making high-performance fine-tuning achievable even under resource constraints.



**2. Based on Training Objectives**

- Supervised Fine-Tuning (SFT)
  This method fine-tunes the model using a fixed dataset—often meticulously curated and annotated by humans—to imbue the model with domain-specific knowledge and a targeted response style. The cross-entropy loss function is typically employed to directly maximize the probability of generating correct responses.

- Reinforcement Learning (RL)

  In contrast, reinforcement learning emphasizes enhancing the overall behavior and performance of the model on a given task. It is implemented via methods such as:

  - RLHF (Reinforcement Learning from Human Feedback): This involves training a reward model based on human preference indicators, followed by using optimization algorithms like Proximal Policy Optimization (PPO) under a composite loss framework that includes policy loss, KL divergence penalties (to keep the model’s outputs from deviating excessively from its pre-trained behavior), and value function updates.

It is worth noting that models such as InstructGPT usually undergo an initial phase of supervised fine-tuning (SFT) before proceeding to RLHF, ensuring that the responses are well-aligned with human expectations at both levels.

> By the way, Deepseek R1 demonstrates another RL approach that refines performance in specific applications using simple reinforcement learning. Compared to RLHF, it doesn’t require an extensive human feedback dataset and is easier and more efficient to train.



## IV. Data Generation and Computational Resource Considerations

**1. Data Generation Methods**

- In the SFT phase, training data typically originates from pre-constructed, high-quality, human-annotated datasets. This not only ensures consistency but also solidifies the training target with well-defined instructional and response parameters.

- In the RL phase, however, the model is required to generate samples within a more open-ended or dynamic environment. These samples are then evaluated by a reward model. The process often involves rounds of sampling, reward computation, advantage estimation, and KL divergence calculation, thereby making data generation substantially more intricate.

  

**2. Computational Resources and Complexity**

Since the SFT stage is based on a static, well-defined dataset, its computational demands are relatively predictable and manageable. Conversely, the RL stage, which involves dynamic sample generation coupled with multiple layers of advanced computations, generally incurs higher overall resource consumption and increased training complexity.



## V. Key Considerations in Engineering Practice

**1. Data Engineering**

For SFT data, it is crucial to ensure:
- Diversity in instructions (e.g., incorporating both system commands and open-ended queries)
- Consistency in responses (verified through cross-validation among multiple annotators)
- Standardized formatting (including role markers and structured separators)

For RLHF data, additional considerations include:
- Ensuring that preference data covers scenarios with potential value conflicts
- Constructing robust comparison loss functions to accurately guide the reward-based evaluation



**2. Infrastructure and Workflow**

Beyond data quality, establishing a solid engineering infrastructure is critical. This encompasses designing efficient data pipelines, implementing robust monitoring systems, and ensuring that the training architecture is scalable, thereby catering to both the SFT and RL phases effectively.



## VI. Training Acceleration Techniques

Modern training frameworks harness several parallelization strategies to expedite model training:

- Data Parallelism: Distributing a single batch across multiple GPUs to simultaneously process data segments.
- Pipeline Parallelism: Vertically splitting the model’s layers into sequential ‘stages’ that can be processed in parallel pipelines.
- Tensor Parallelism: Horizontally distributing single matrix operations across multiple devices to accelerate specific computations.

Common tools and frameworks used to facilitate these strategies include:

- Accelerator frameworks that abstract the complexity of distributed training.
- Distributed training libraries such as DeepSpeed and Fully Sharded Data Parallel (FSDP), which enable efficient scaling and resource management for large models.



## VII. Training Data Design: Prompt-Template





## VIII. Application and Practice Tools

- Fine-Tuning Tools: Axolotl is well-known for its ease and efficiency in fine-tuning models. It integrates many components, such as Accelerator frameworks, LoRA, DeepSpeed, Ray, and more.
- Monitoring Tools: Solutions like WandB (Weights & Biases) play a pivotal role in real-time tracking of training metrics, visualizing model performance, and diagnosing issues during the training process.