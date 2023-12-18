# Latest Large Language Model Attack Papers
**update at 2023-12-18 10:03:08**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM_CN.md)

## **1. No-Skim: Towards Efficiency Robustness Evaluation on Skimming-based Language Models**

cs.CR

**SubmitDate**: 2023-12-15    [abs](http://arxiv.org/abs/2312.09494v1) [paper-pdf](http://arxiv.org/pdf/2312.09494v1)

**Authors**: Shengyao Zhang, Mi Zhang, Xudong Pan, Min Yang

**Abstract**: To reduce the computation cost and the energy consumption in large language models (LLM), skimming-based acceleration dynamically drops unimportant tokens of the input sequence progressively along layers of the LLM while preserving the tokens of semantic importance. However, our work for the first time reveals the acceleration may be vulnerable to Denial-of-Service (DoS) attacks. In this paper, we propose No-Skim, a general framework to help the owners of skimming-based LLM to understand and measure the robustness of their acceleration scheme. Specifically, our framework searches minimal and unnoticeable perturbations at character-level and token-level to generate adversarial inputs that sufficiently increase the remaining token ratio, thus increasing the computation cost and energy consumption. We systematically evaluate the vulnerability of the skimming acceleration in various LLM architectures including BERT and RoBERTa on the GLUE benchmark. In the worst case, the perturbation found by No-Skim substantially increases the running cost of LLM by over 145% on average. Moreover, No-Skim extends the evaluation framework to various scenarios, making the evaluation conductible with different level of knowledge.



## **2. AutoDAN: Interpretable Gradient-Based Adversarial Attacks on Large Language Models**

cs.CR

Version 2 updates: Added comparison of three more evaluation methods  and their reliability check using human labeling. Added results for  jailbreaking Llama2 (individual behavior) and included complexity and  hyperparameter analysis. Revised objectives for prompt leaking. Other minor  changes made

**SubmitDate**: 2023-12-14    [abs](http://arxiv.org/abs/2310.15140v2) [paper-pdf](http://arxiv.org/pdf/2310.15140v2)

**Authors**: Sicheng Zhu, Ruiyi Zhang, Bang An, Gang Wu, Joe Barrow, Zichao Wang, Furong Huang, Ani Nenkova, Tong Sun

**Abstract**: Safety alignment of Large Language Models (LLMs) can be compromised with manual jailbreak attacks and (automatic) adversarial attacks. Recent studies suggest that defending against these attacks is possible: adversarial attacks generate unlimited but unreadable gibberish prompts, detectable by perplexity-based filters; manual jailbreak attacks craft readable prompts, but their limited number due to the necessity of human creativity allows for easy blocking. In this paper, we show that these solutions may be too optimistic. We introduce AutoDAN, an interpretable, gradient-based adversarial attack that merges the strengths of both attack types. Guided by the dual goals of jailbreak and readability, AutoDAN optimizes and generates tokens one by one from left to right, resulting in readable prompts that bypass perplexity filters while maintaining high attack success rates. Notably, these prompts, generated from scratch using gradients, are interpretable and diverse, with emerging strategies commonly seen in manual jailbreak attacks. They also generalize to unforeseen harmful behaviors and transfer to black-box LLMs better than their unreadable counterparts when using limited training data or a single proxy model. Furthermore, we show the versatility of AutoDAN by automatically leaking system prompts using a customized objective. Our work offers a new way to red-team LLMs and understand jailbreak mechanisms via interpretability.



## **3. FigStep: Jailbreaking Large Vision-language Models via Typographic Visual Prompts**

cs.CR

Technical Report

**SubmitDate**: 2023-12-13    [abs](http://arxiv.org/abs/2311.05608v2) [paper-pdf](http://arxiv.org/pdf/2311.05608v2)

**Authors**: Yichen Gong, Delong Ran, Jinyuan Liu, Conglei Wang, Tianshuo Cong, Anyu Wang, Sisi Duan, Xiaoyun Wang

**Abstract**: Ensuring the safety of artificial intelligence-generated content (AIGC) is a longstanding topic in the artificial intelligence (AI) community, and the safety concerns associated with Large Language Models (LLMs) have been widely investigated. Recently, large vision-language models (VLMs) represent an unprecedented revolution, as they are built upon LLMs but can incorporate additional modalities (e.g., images). However, the safety of VLMs lacks systematic evaluation, and there may be an overconfidence in the safety guarantees provided by their underlying LLMs. In this paper, to demonstrate that introducing additional modality modules leads to unforeseen AI safety issues, we propose FigStep, a straightforward yet effective jailbreaking algorithm against VLMs. Instead of feeding textual harmful instructions directly, FigStep converts the harmful content into images through typography to bypass the safety alignment within the textual module of the VLMs, inducing VLMs to output unsafe responses that violate common AI safety policies. In our evaluation, we manually review 46,500 model responses generated by 3 families of the promising open-source VLMs, i.e., LLaVA, MiniGPT4, and CogVLM (a total of 6 VLMs). The experimental results show that FigStep can achieve an average attack success rate of 82.50% on 500 harmful queries in 10 topics. Moreover, we demonstrate that the methodology of FigStep can even jailbreak GPT-4V, which already leverages an OCR detector to filter harmful queries. Above all, our work reveals that VLMs are vulnerable to jailbreaking attacks, which highlights the necessity of novel safety alignments between visual and textual modalities.



## **4. Efficient Representation of the Activation Space in Deep Neural Networks**

cs.LG

**SubmitDate**: 2023-12-13    [abs](http://arxiv.org/abs/2312.08143v1) [paper-pdf](http://arxiv.org/pdf/2312.08143v1)

**Authors**: Tanya Akumu, Celia Cintas, Girmaw Abebe Tadesse, Adebayo Oshingbesan, Skyler Speakman, Edward McFowland III

**Abstract**: The representations of the activation space of deep neural networks (DNNs) are widely utilized for tasks like natural language processing, anomaly detection and speech recognition. Due to the diverse nature of these tasks and the large size of DNNs, an efficient and task-independent representation of activations becomes crucial. Empirical p-values have been used to quantify the relative strength of an observed node activation compared to activations created by already-known inputs. Nonetheless, keeping raw data for these calculations increases memory resource consumption and raises privacy concerns. To this end, we propose a model-agnostic framework for creating representations of activations in DNNs using node-specific histograms to compute p-values of observed activations without retaining already-known inputs. Our proposed approach demonstrates promising potential when validated with multiple network architectures across various downstream tasks and compared with the kernel density estimates and brute-force empirical baselines. In addition, the framework reduces memory usage by 30% with up to 4 times faster p-value computing time while maintaining state of-the-art detection power in downstream tasks such as the detection of adversarial attacks and synthesized content. Moreover, as we do not persist raw data at inference time, we could potentially reduce susceptibility to attacks and privacy issues.



## **5. PromptBench: A Unified Library for Evaluation of Large Language Models**

cs.AI

An extension to PromptBench (arXiv:2306.04528) for unified evaluation  of LLMs using the same name; code: https://github.com/microsoft/promptbench

**SubmitDate**: 2023-12-13    [abs](http://arxiv.org/abs/2312.07910v1) [paper-pdf](http://arxiv.org/pdf/2312.07910v1)

**Authors**: Kaijie Zhu, Qinlin Zhao, Hao Chen, Jindong Wang, Xing Xie

**Abstract**: The evaluation of large language models (LLMs) is crucial to assess their performance and mitigate potential security risks. In this paper, we introduce PromptBench, a unified library to evaluate LLMs. It consists of several key components that are easily used and extended by researchers: prompt construction, prompt engineering, dataset and model loading, adversarial prompt attack, dynamic evaluation protocols, and analysis tools. PromptBench is designed to be an open, general, and flexible codebase for research purposes that can facilitate original study in creating new benchmarks, deploying downstream applications, and designing new evaluation protocols. The code is available at: https://github.com/microsoft/promptbench and will be continuously supported.



## **6. Causality Analysis for Evaluating the Security of Large Language Models**

cs.AI

**SubmitDate**: 2023-12-13    [abs](http://arxiv.org/abs/2312.07876v1) [paper-pdf](http://arxiv.org/pdf/2312.07876v1)

**Authors**: Wei Zhao, Zhe Li, Jun Sun

**Abstract**: Large Language Models (LLMs) such as GPT and Llama2 are increasingly adopted in many safety-critical applications. Their security is thus essential. Even with considerable efforts spent on reinforcement learning from human feedback (RLHF), recent studies have shown that LLMs are still subject to attacks such as adversarial perturbation and Trojan attacks. Further research is thus needed to evaluate their security and/or understand the lack of it. In this work, we propose a framework for conducting light-weight causality-analysis of LLMs at the token, layer, and neuron level. We applied our framework to open-source LLMs such as Llama2 and Vicuna and had multiple interesting discoveries. Based on a layer-level causality analysis, we show that RLHF has the effect of overfitting a model to harmful prompts. It implies that such security can be easily overcome by `unusual' harmful prompts. As evidence, we propose an adversarial perturbation method that achieves 100\% attack success rate on the red-teaming tasks of the Trojan Detection Competition 2023. Furthermore, we show the existence of one mysterious neuron in both Llama2 and Vicuna that has an unreasonably high causal effect on the output. While we are uncertain on why such a neuron exists, we show that it is possible to conduct a ``Trojan'' attack targeting that particular neuron to completely cripple the LLM, i.e., we can generate transferable suffixes to prompts that frequently make the LLM produce meaningless responses.



## **7. DeceptPrompt: Exploiting LLM-driven Code Generation via Adversarial Natural Language Instructions**

cs.CR

**SubmitDate**: 2023-12-12    [abs](http://arxiv.org/abs/2312.04730v2) [paper-pdf](http://arxiv.org/pdf/2312.04730v2)

**Authors**: Fangzhou Wu, Xiaogeng Liu, Chaowei Xiao

**Abstract**: With the advancement of Large Language Models (LLMs), significant progress has been made in code generation, enabling LLMs to transform natural language into programming code. These Code LLMs have been widely accepted by massive users and organizations. However, a dangerous nature is hidden in the code, which is the existence of fatal vulnerabilities. While some LLM providers have attempted to address these issues by aligning with human guidance, these efforts fall short of making Code LLMs practical and robust. Without a deep understanding of the performance of the LLMs under the practical worst cases, it would be concerning to apply them to various real-world applications. In this paper, we answer the critical issue: Are existing Code LLMs immune to generating vulnerable code? If not, what is the possible maximum severity of this issue in practical deployment scenarios? In this paper, we introduce DeceptPrompt, a novel algorithm that can generate adversarial natural language instructions that drive the Code LLMs to generate functionality correct code with vulnerabilities. DeceptPrompt is achieved through a systematic evolution-based algorithm with a fine grain loss design. The unique advantage of DeceptPrompt enables us to find natural prefix/suffix with totally benign and non-directional semantic meaning, meanwhile, having great power in inducing the Code LLMs to generate vulnerable code. This feature can enable us to conduct the almost-worstcase red-teaming on these LLMs in a real scenario, where users are using natural language. Our extensive experiments and analyses on DeceptPrompt not only validate the effectiveness of our approach but also shed light on the huge weakness of LLMs in the code generation task. When applying the optimized prefix/suffix, the attack success rate (ASR) will improve by average 50% compared with no prefix/suffix applying.



## **8. Practical Membership Inference Attacks against Fine-tuned Large Language Models via Self-prompt Calibration**

cs.CL

**SubmitDate**: 2023-12-12    [abs](http://arxiv.org/abs/2311.06062v2) [paper-pdf](http://arxiv.org/pdf/2311.06062v2)

**Authors**: Wenjie Fu, Huandong Wang, Chen Gao, Guanghua Liu, Yong Li, Tao Jiang

**Abstract**: Membership Inference Attacks (MIA) aim to infer whether a target data record has been utilized for model training or not. Prior attempts have quantified the privacy risks of language models (LMs) via MIAs, but there is still no consensus on whether existing MIA algorithms can cause remarkable privacy leakage on practical Large Language Models (LLMs). Existing MIAs designed for LMs can be classified into two categories: reference-free and reference-based attacks. They are both based on the hypothesis that training records consistently strike a higher probability of being sampled. Nevertheless, this hypothesis heavily relies on the overfitting of target models, which will be mitigated by multiple regularization methods and the generalization of LLMs. The reference-based attack seems to achieve promising effectiveness in LLMs, which measures a more reliable membership signal by comparing the probability discrepancy between the target model and the reference model. However, the performance of reference-based attack is highly dependent on a reference dataset that closely resembles the training dataset, which is usually inaccessible in the practical scenario. Overall, existing MIAs are unable to effectively unveil privacy leakage over practical fine-tuned LLMs that are overfitting-free and private. We propose a Membership Inference Attack based on Self-calibrated Probabilistic Variation (SPV-MIA). Specifically, since memorization in LLMs is inevitable during the training process and occurs before overfitting, we introduce a more reliable membership signal, probabilistic variation, which is based on memorization rather than overfitting. Furthermore, we introduce a self-prompt approach, which constructs the dataset to fine-tune the reference model by prompting the target LLM itself. In this manner, the adversary can collect a dataset with a similar distribution from public APIs.



## **9. Safety Alignment in NLP Tasks: Weakly Aligned Summarization as an In-Context Attack**

cs.CL

17 pages,10 figures

**SubmitDate**: 2023-12-12    [abs](http://arxiv.org/abs/2312.06924v1) [paper-pdf](http://arxiv.org/pdf/2312.06924v1)

**Authors**: Yu Fu, Yufei Li, Wen Xiao, Cong Liu, Yue Dong

**Abstract**: Recent developments in balancing the usefulness and safety of Large Language Models (LLMs) have raised a critical question: Are mainstream NLP tasks adequately aligned with safety consideration? Our study, focusing on safety-sensitive documents obtained through adversarial attacks, reveals significant disparities in the safety alignment of various NLP tasks. For instance, LLMs can effectively summarize malicious long documents but often refuse to translate them. This discrepancy highlights a previously unidentified vulnerability: attacks exploiting tasks with weaker safety alignment, like summarization, can potentially compromise the integraty of tasks traditionally deemed more robust, such as translation and question-answering (QA). Moreover, the concurrent use of multiple NLP tasks with lesser safety alignment increases the risk of LLMs inadvertently processing harmful content. We demonstrate these vulnerabilities in various safety-aligned LLMs, particularly Llama2 models and GPT-4, indicating an urgent need for strengthening safety alignments across a broad spectrum of NLP tasks.



## **10. GPTBIAS: A Comprehensive Framework for Evaluating Bias in Large Language Models**

cs.CL

**SubmitDate**: 2023-12-11    [abs](http://arxiv.org/abs/2312.06315v1) [paper-pdf](http://arxiv.org/pdf/2312.06315v1)

**Authors**: Jiaxu Zhao, Meng Fang, Shirui Pan, Wenpeng Yin, Mykola Pechenizkiy

**Abstract**: Warning: This paper contains content that may be offensive or upsetting. There has been a significant increase in the usage of large language models (LLMs) in various applications, both in their original form and through fine-tuned adaptations. As a result, LLMs have gained popularity and are being widely adopted by a large user community. However, one of the concerns with LLMs is the potential generation of socially biased content. The existing evaluation methods have many constraints, and their results exhibit a limited degree of interpretability. In this work, we propose a bias evaluation framework named GPTBIAS that leverages the high performance of LLMs (e.g., GPT-4 \cite{openai2023gpt4}) to assess bias in models. We also introduce prompts called Bias Attack Instructions, which are specifically designed for evaluating model bias. To enhance the credibility and interpretability of bias evaluation, our framework not only provides a bias score but also offers detailed information, including bias types, affected demographics, keywords, reasons behind the biases, and suggestions for improvement. We conduct extensive experiments to demonstrate the effectiveness and usability of our bias evaluation framework.



## **11. InferDPT: Privacy-Preserving Inference for Black-box Large Language Model**

cs.CR

**SubmitDate**: 2023-12-11    [abs](http://arxiv.org/abs/2310.12214v5) [paper-pdf](http://arxiv.org/pdf/2310.12214v5)

**Authors**: Meng Tong, Kejiang Chen, Jie Zhang, Yuang Qi, Weiming Zhang, Nenghai Yu

**Abstract**: Large language models (LLMs), like ChatGPT, have greatly simplified text generation tasks. However, they have also raised concerns about privacy risks such as data leakage and unauthorized data collection. Existing solutions for privacy-preserving inference face practical challenges related to computation time and communication costs. In this paper, we propose InferDPT, the first practical framework for the privacy-preserving Inference of black-box LLMs, implementing Differential Privacy in Text generation. InferDPT comprises two key modules: the "perturbation module" utilizes the exponential mechanism to generate a perturbed prompt, facilitating privacy-preserving inference with black-box LLMs, and the "extraction module", inspired by knowledge distillation and retrieval-augmented generation, extracts coherent and consistent text from the perturbed generation result, ensuring successful text generation completion. To address privacy concerns related to previous exponential mechanisms' susceptibility to embedding revision attacks, we introduce RANTEXT, a novel differential privacy mechanism integrated into the perturbation module of InferDPT, which introduces the concept of "RANdom adjacency" for TEXT perturbation within the prompt. Experimental results across three datasets demonstrate that the text generation quality of InferDPT is comparable to that of non-private GPT-4, and RANTEXT surpasses existing state-of-the-art mechanisms, namely, SANTEXT+ and CUSTEXT+ in the trade-off between privacy and utility. Even with an privacy parameter epsilon value of 6.0, RANTEXT achieves an average privacy protection rate exceeding 90% against embedding revision attacks, which is 0.58 times higher than that of SANTEXT+ and 3.35 times higher than that of CUSTEXT+.



## **12. METAL: Metamorphic Testing Framework for Analyzing Large-Language Model Qualities**

cs.SE

Accepted to International Conference on Software Testing,  Verification and Validation (ICST) 2024 / Key words: Large-language models,  Metamorphic testing, Quality evaluation, Text perturbations

**SubmitDate**: 2023-12-11    [abs](http://arxiv.org/abs/2312.06056v1) [paper-pdf](http://arxiv.org/pdf/2312.06056v1)

**Authors**: Sangwon Hyun, Mingyu Guo, M. Ali Babar

**Abstract**: Large-Language Models (LLMs) have shifted the paradigm of natural language data processing. However, their black-boxed and probabilistic characteristics can lead to potential risks in the quality of outputs in diverse LLM applications. Recent studies have tested Quality Attributes (QAs), such as robustness or fairness, of LLMs by generating adversarial input texts. However, existing studies have limited their coverage of QAs and tasks in LLMs and are difficult to extend. Additionally, these studies have only used one evaluation metric, Attack Success Rate (ASR), to assess the effectiveness of their approaches. We propose a MEtamorphic Testing for Analyzing LLMs (METAL) framework to address these issues by applying Metamorphic Testing (MT) techniques. This approach facilitates the systematic testing of LLM qualities by defining Metamorphic Relations (MRs), which serve as modularized evaluation metrics. The METAL framework can automatically generate hundreds of MRs from templates that cover various QAs and tasks. In addition, we introduced novel metrics that integrate the ASR method into the semantic qualities of text to assess the effectiveness of MRs accurately. Through the experiments conducted with three prominent LLMs, we have confirmed that the METAL framework effectively evaluates essential QAs on primary LLM tasks and reveals the quality risks in LLMs. Moreover, the newly proposed metrics can guide the optimal MRs for testing each task and suggest the most effective method for generating MRs.



## **13. Occlusion-based Detection of Trojan-triggering Inputs in Large Language Models of Code**

cs.SE

**SubmitDate**: 2023-12-10    [abs](http://arxiv.org/abs/2312.04004v2) [paper-pdf](http://arxiv.org/pdf/2312.04004v2)

**Authors**: Aftab Hussain, Md Rafiqul Islam Rabin, Toufique Ahmed, Mohammad Amin Alipour, Bowen Xu

**Abstract**: Large language models (LLMs) are becoming an integrated part of software development. These models are trained on large datasets for code, where it is hard to verify each data point. Therefore, a potential attack surface can be to inject poisonous data into the training data to make models vulnerable, aka trojaned. It can pose a significant threat by hiding manipulative behaviors inside models, leading to compromising the integrity of the models in downstream tasks.   In this paper, we propose an occlusion-based human-in-the-loop technique, OSeql, to distinguish trojan-triggering inputs of code. The technique is based on the observation that trojaned neural models of code rely heavily on the triggering part of input; hence, its removal would change the confidence of the models in their prediction substantially. Our results suggest that OSeql can detect the triggering inputs with almost 100% recall. We discuss the problem of false positives and how to address them. These results provide a baseline for future studies in this field.



## **14. Towards Robust Pruning: An Adaptive Knowledge-Retention Pruning Strategy for Language Models**

cs.CL

**SubmitDate**: 2023-12-10    [abs](http://arxiv.org/abs/2310.13191v2) [paper-pdf](http://arxiv.org/pdf/2310.13191v2)

**Authors**: Jianwei Li, Qi Lei, Wei Cheng, Dongkuan Xu

**Abstract**: The pruning objective has recently extended beyond accuracy and sparsity to robustness in language models. Despite this, existing methods struggle to enhance robustness against adversarial attacks when continually increasing model sparsity and require a retraining process. As humans step into the era of large language models, these issues become increasingly prominent. This paper proposes that the robustness of language models is proportional to the extent of pre-trained knowledge they encompass. Accordingly, we introduce a post-training pruning strategy designed to faithfully replicate the embedding space and feature space of dense language models, aiming to conserve more pre-trained knowledge during the pruning process. In this setup, each layer's reconstruction error not only originates from itself but also includes cumulative error from preceding layers, followed by an adaptive rectification. Compared to other state-of-art baselines, our approach demonstrates a superior balance between accuracy, sparsity, robustness, and pruning cost with BERT on datasets SST2, IMDB, and AGNews, marking a significant stride towards robust pruning in language models.



## **15. Temporal-Distributed Backdoor Attack Against Video Based Action Recognition**

cs.CV

accepted by AAAI 2024

**SubmitDate**: 2023-12-09    [abs](http://arxiv.org/abs/2308.11070v3) [paper-pdf](http://arxiv.org/pdf/2308.11070v3)

**Authors**: Xi Li, Songhe Wang, Ruiquan Huang, Mahanth Gowda, George Kesidis

**Abstract**: Deep neural networks (DNNs) have achieved tremendous success in various applications including video action recognition, yet remain vulnerable to backdoor attacks (Trojans). The backdoor-compromised model will mis-classify to the target class chosen by the attacker when a test instance (from a non-target class) is embedded with a specific trigger, while maintaining high accuracy on attack-free instances. Although there are extensive studies on backdoor attacks against image data, the susceptibility of video-based systems under backdoor attacks remains largely unexplored. Current studies are direct extensions of approaches proposed for image data, e.g., the triggers are independently embedded within the frames, which tend to be detectable by existing defenses. In this paper, we introduce a simple yet effective backdoor attack against video data. Our proposed attack, adding perturbations in a transformed domain, plants an imperceptible, temporally distributed trigger across the video frames, and is shown to be resilient to existing defensive strategies. The effectiveness of the proposed attack is demonstrated by extensive experiments with various well-known models on two video recognition benchmarks, UCF101 and HMDB51, and a sign language recognition benchmark, Greek Sign Language (GSL) dataset. We delve into the impact of several influential factors on our proposed attack and identify an intriguing effect termed "collateral damage" through extensive studies.



## **16. HuRef: HUman-REadable Fingerprint for Large Language Models**

cs.CL

**SubmitDate**: 2023-12-08    [abs](http://arxiv.org/abs/2312.04828v1) [paper-pdf](http://arxiv.org/pdf/2312.04828v1)

**Authors**: Boyi Zeng, Chenghu Zhou, Xinbing Wang, Zhouhan Lin

**Abstract**: Protecting the copyright of large language models (LLMs) has become crucial due to their resource-intensive training and accompanying carefully designed licenses. However, identifying the original base model of an LLM is challenging due to potential parameter alterations through fine-tuning or continued pretraining. In this study, we introduce HuRef, a human-readable fingerprint for LLMs that uniquely identifies the base model without exposing model parameters or interfering with training. We first observe that the vector direction of LLM parameters remains stable after the model has converged during pretraining, showing negligible perturbations through subsequent training steps, including continued pretraining, supervised fine-tuning (SFT), and RLHF, which makes it a sufficient condition to identify the base model. The necessity is validated by continuing to train an LLM with an extra term to drive away the model parameters' direction and the model becomes damaged. However, this direction is vulnerable to simple attacks like dimension permutation or matrix rotation, which significantly change it without affecting performance. To address this, leveraging the Transformer structure, we systematically analyze potential attacks and define three invariant terms that identify an LLM's base model. We make these invariant terms human-readable by mapping them to a Gaussian vector using a convolutional encoder and then converting it into a natural image with StyleGAN2. Our method generates a dog image as an identity fingerprint for an LLM, where the dog's appearance strongly indicates the LLM's base model. Experimental results across various LLMs demonstrate the effectiveness of our method, the generated dog image remains invariant to different training steps, including SFT, RLHF, or even continued pretraining with augmented vocabulary in a new language.



## **17. Goal-Oriented Prompt Attack and Safety Evaluation for LLMs**

cs.CL

**SubmitDate**: 2023-12-08    [abs](http://arxiv.org/abs/2309.11830v2) [paper-pdf](http://arxiv.org/pdf/2309.11830v2)

**Authors**: Chengyuan Liu, Fubang Zhao, Lizhi Qing, Yangyang Kang, Changlong Sun, Kun Kuang, Fei Wu

**Abstract**: Large Language Models (LLMs) presents significant priority in text understanding and generation. However, LLMs suffer from the risk of generating harmful contents especially while being employed to applications. There are several black-box attack methods, such as Prompt Attack, which can change the behaviour of LLMs and induce LLMs to generate unexpected answers with harmful contents. Researchers are interested in Prompt Attack and Defense with LLMs, while there is no publicly available dataset with high successful attacking rate to evaluate the abilities of defending prompt attack. In this paper, we introduce a pipeline to construct high-quality prompt attack samples, along with a Chinese prompt attack dataset called CPAD. Our prompts aim to induce LLMs to generate unexpected outputs with several carefully designed prompt attack templates and widely concerned attacking contents. Different from previous datasets involving safety estimation, we construct the prompts considering three dimensions: contents, attacking methods and goals. Especially, the attacking goals indicate the behaviour expected after successfully attacking the LLMs, thus the responses can be easily evaluated and analysed. We run several popular Chinese LLMs on our dataset, and the results show that our prompts are significantly harmful to LLMs, with around 70% attack success rate to GPT-3.5. CPAD is publicly available at https://github.com/liuchengyuan123/CPAD.



## **18. Make Them Spill the Beans! Coercive Knowledge Extraction from (Production) LLMs**

cs.CR

**SubmitDate**: 2023-12-08    [abs](http://arxiv.org/abs/2312.04782v1) [paper-pdf](http://arxiv.org/pdf/2312.04782v1)

**Authors**: Zhuo Zhang, Guangyu Shen, Guanhong Tao, Siyuan Cheng, Xiangyu Zhang

**Abstract**: Large Language Models (LLMs) are now widely used in various applications, making it crucial to align their ethical standards with human values. However, recent jail-breaking methods demonstrate that this alignment can be undermined using carefully constructed prompts. In our study, we reveal a new threat to LLM alignment when a bad actor has access to the model's output logits, a common feature in both open-source LLMs and many commercial LLM APIs (e.g., certain GPT models). It does not rely on crafting specific prompts. Instead, it exploits the fact that even when an LLM rejects a toxic request, a harmful response often hides deep in the output logits. By forcefully selecting lower-ranked output tokens during the auto-regressive generation process at a few critical output positions, we can compel the model to reveal these hidden responses. We term this process model interrogation. This approach differs from and outperforms jail-breaking methods, achieving 92% effectiveness compared to 62%, and is 10 to 20 times faster. The harmful content uncovered through our method is more relevant, complete, and clear. Additionally, it can complement jail-breaking strategies, with which results in further boosting attack performance. Our findings indicate that interrogation can extract toxic knowledge even from models specifically designed for coding tasks.



## **19. Forcing Generative Models to Degenerate Ones: The Power of Data Poisoning Attacks**

cs.CR

19 pages, 6 figures. Published at NeurIPS 2023 Workshop on Backdoors  in Deep Learning: The Good, the Bad, and the Ugly

**SubmitDate**: 2023-12-07    [abs](http://arxiv.org/abs/2312.04748v1) [paper-pdf](http://arxiv.org/pdf/2312.04748v1)

**Authors**: Shuli Jiang, Swanand Ravindra Kadhe, Yi Zhou, Ling Cai, Nathalie Baracaldo

**Abstract**: Growing applications of large language models (LLMs) trained by a third party raise serious concerns on the security vulnerability of LLMs.It has been demonstrated that malicious actors can covertly exploit these vulnerabilities in LLMs through poisoning attacks aimed at generating undesirable outputs. While poisoning attacks have received significant attention in the image domain (e.g., object detection), and classification tasks, their implications for generative models, particularly in the realm of natural language generation (NLG) tasks, remain poorly understood. To bridge this gap, we perform a comprehensive exploration of various poisoning techniques to assess their effectiveness across a range of generative tasks. Furthermore, we introduce a range of metrics designed to quantify the success and stealthiness of poisoning attacks specifically tailored to NLG tasks. Through extensive experiments on multiple NLG tasks, LLMs and datasets, we show that it is possible to successfully poison an LLM during the fine-tuning stage using as little as 1\% of the total tuning data samples. Our paper presents the first systematic approach to comprehend poisoning attacks targeting NLG tasks considering a wide range of triggers and attack settings. We hope our findings will assist the AI security community in devising appropriate defenses against such threats.



## **20. Defending Against Alignment-Breaking Attacks via Robustly Aligned LLM**

cs.CL

16 Pages, 5 Figures, 6 Tables

**SubmitDate**: 2023-12-07    [abs](http://arxiv.org/abs/2309.14348v2) [paper-pdf](http://arxiv.org/pdf/2309.14348v2)

**Authors**: Bochuan Cao, Yuanpu Cao, Lu Lin, Jinghui Chen

**Abstract**: Recently, Large Language Models (LLMs) have made significant advancements and are now widely used across various domains. Unfortunately, there has been a rising concern that LLMs can be misused to generate harmful or malicious content. Though a line of research has focused on aligning LLMs with human values and preventing them from producing inappropriate content, such alignments are usually vulnerable and can be bypassed by alignment-breaking attacks via adversarially optimized or handcrafted jailbreaking prompts. In this work, we introduce a Robustly Aligned LLM (RA-LLM) to defend against potential alignment-breaking attacks. RA-LLM can be directly constructed upon an existing aligned LLM with a robust alignment checking function, without requiring any expensive retraining or fine-tuning process of the original LLM. Furthermore, we also provide a theoretical analysis for RA-LLM to verify its effectiveness in defending against alignment-breaking attacks. Through real-world experiments on open-source large language models, we demonstrate that RA-LLM can successfully defend against both state-of-the-art adversarial prompts and popular handcrafted jailbreaking prompts by reducing their attack success rates from nearly 100% to around 10% or less.



## **21. Domain Private Transformers for Multi-Domain Dialog Systems**

cs.CL

Accepted to Findings of EMNLP 2023 (short paper). Code available at  https://github.com/asappresearch/domain-private-transformers

**SubmitDate**: 2023-12-07    [abs](http://arxiv.org/abs/2305.14208v2) [paper-pdf](http://arxiv.org/pdf/2305.14208v2)

**Authors**: Anmol Kabra, Ethan R. Elenberg

**Abstract**: Large, general purpose language models have demonstrated impressive performance across many different conversational domains. While multi-domain language models achieve low overall perplexity, their outputs are not guaranteed to stay within the domain of a given input prompt. This paper proposes domain privacy as a novel way to quantify how likely a conditional language model will leak across domains. We also develop policy functions based on token-level domain classification, and propose an efficient fine-tuning method to improve the trained model's domain privacy. Experiments on membership inference attacks show that our proposed method has comparable resiliency to methods adapted from recent literature on differentially private language models.



## **22. Analyzing the Inherent Response Tendency of LLMs: Real-World Instructions-Driven Jailbreak**

cs.CL

**SubmitDate**: 2023-12-07    [abs](http://arxiv.org/abs/2312.04127v1) [paper-pdf](http://arxiv.org/pdf/2312.04127v1)

**Authors**: Yanrui Du, Sendong Zhao, Ming Ma, Yuhan Chen, Bing Qin

**Abstract**: Extensive work has been devoted to improving the safety mechanism of Large Language Models (LLMs). However, in specific scenarios, LLMs still generate harmful responses when faced with malicious instructions, a phenomenon referred to as "Jailbreak Attack". In our research, we introduce a novel jailbreak attack method (\textbf{RADIAL}), which consists of two steps: 1) Inherent Response Tendency Analysis: we analyze the inherent affirmation and rejection tendency of LLMs to react to real-world instructions. 2) Real-World Instructions-Driven Jailbreak: based on our analysis, we strategically choose several real-world instructions and embed malicious instructions into them to amplify the LLM's potential to generate harmful responses. On three open-source human-aligned LLMs, our method achieves excellent jailbreak attack performance for both Chinese and English malicious instructions. Besides, we guided detailed ablation experiments and verified the effectiveness of our core idea "Inherent Response Tendency Analysis". Our exploration also exposes the vulnerability of LLMs to being induced into generating more detailed harmful responses in subsequent rounds of dialogue.



## **23. Mark My Words: Analyzing and Evaluating Language Model Watermarks**

cs.CR

18 pages, 11 figures

**SubmitDate**: 2023-12-07    [abs](http://arxiv.org/abs/2312.00273v2) [paper-pdf](http://arxiv.org/pdf/2312.00273v2)

**Authors**: Julien Piet, Chawin Sitawarin, Vivian Fang, Norman Mu, David Wagner

**Abstract**: The capabilities of large language models have grown significantly in recent years and so too have concerns about their misuse. In this context, the ability to distinguish machine-generated text from human-authored content becomes important. Prior works have proposed numerous schemes to watermark text, which would benefit from a systematic evaluation framework. This work focuses on text watermarking techniques - as opposed to image watermarks - and proposes MARKMYWORDS, a comprehensive benchmark for them under different tasks as well as practical attacks. We focus on three main metrics: quality, size (e.g. the number of tokens needed to detect a watermark), and tamper-resistance. Current watermarking techniques are good enough to be deployed: Kirchenbauer et al. [1] can watermark Llama2-7B-chat with no perceivable loss in quality, the watermark can be detected with fewer than 100 tokens, and the scheme offers good tamper-resistance to simple attacks. We argue that watermark indistinguishability, a criteria emphasized in some prior works, is too strong a requirement: schemes that slightly modify logit distributions outperform their indistinguishable counterparts with no noticeable loss in generation quality. We publicly release our benchmark (https://github.com/wagner-group/MarkMyWords)



## **24. Dr. Jekyll and Mr. Hyde: Two Faces of LLMs**

cs.CR

**SubmitDate**: 2023-12-06    [abs](http://arxiv.org/abs/2312.03853v1) [paper-pdf](http://arxiv.org/pdf/2312.03853v1)

**Authors**: Matteo Gioele Collu, Tom Janssen-Groesbeek, Stefanos Koffas, Mauro Conti, Stjepan Picek

**Abstract**: This year, we witnessed a rise in the use of Large Language Models, especially when combined with applications like chatbot assistants. Safety mechanisms and specialized training procedures are put in place to prevent improper responses from these assistants. In this work, we bypass these measures for ChatGPT and Bard (and, to some extent, Bing chat) by making them impersonate complex personas with opposite characteristics as those of the truthful assistants they are supposed to be. We start by creating elaborate biographies of these personas, which we then use in a new session with the same chatbots. Our conversation followed a role-play style to get the response the assistant was not allowed to provide. By making use of personas, we show that the response that is prohibited is actually provided, making it possible to obtain unauthorized, illegal, or harmful information. This work shows that by using adversarial personas, one can overcome safety mechanisms set out by ChatGPT and Bard. It also introduces several ways of activating such adversarial personas, altogether showing that both chatbots are vulnerable to this kind of attack.



## **25. Clinical Notes Reveal Physician Fatigue**

cs.CL

**SubmitDate**: 2023-12-05    [abs](http://arxiv.org/abs/2312.03077v1) [paper-pdf](http://arxiv.org/pdf/2312.03077v1)

**Authors**: Chao-Chun Hsu, Ziad Obermeyer, Chenhao Tan

**Abstract**: Physicians write notes about patients. In doing so, they reveal much about themselves. Using data from 129,228 emergency room visits, we train a model to identify notes written by fatigued physicians -- those who worked 5 or more of the prior 7 days. In a hold-out set, the model accurately identifies notes written by these high-workload physicians, and also flags notes written in other high-fatigue settings: on overnight shifts, and after high patient volumes. Model predictions also correlate with worse decision-making on at least one important metric: yield of testing for heart attack is 18% lower with each standard deviation increase in model-predicted fatigue. Finally, the model indicates that notes written about Black and Hispanic patients have 12% and 21% higher predicted fatigue than Whites -- larger than overnight vs. daytime differences. These results have an important implication for large language models (LLMs). Our model indicates that fatigued doctors write more predictable notes. Perhaps unsurprisingly, because word prediction is the core of how LLMs work, we find that LLM-written notes have 17% higher predicted fatigue than real physicians' notes. This indicates that LLMs may introduce distortions in generated text that are not yet fully understood.



## **26. Tree of Attacks: Jailbreaking Black-Box LLMs Automatically**

cs.LG

An implementation of the presented method is available at  https://github.com/RICommunity/TAP

**SubmitDate**: 2023-12-04    [abs](http://arxiv.org/abs/2312.02119v1) [paper-pdf](http://arxiv.org/pdf/2312.02119v1)

**Authors**: Anay Mehrotra, Manolis Zampetakis, Paul Kassianik, Blaine Nelson, Hyrum Anderson, Yaron Singer, Amin Karbasi

**Abstract**: While Large Language Models (LLMs) display versatile functionality, they continue to generate harmful, biased, and toxic content, as demonstrated by the prevalence of human-designed jailbreaks. In this work, we present Tree of Attacks with Pruning (TAP), an automated method for generating jailbreaks that only requires black-box access to the target LLM. TAP utilizes an LLM to iteratively refine candidate (attack) prompts using tree-of-thoughts reasoning until one of the generated prompts jailbreaks the target. Crucially, before sending prompts to the target, TAP assesses them and prunes the ones unlikely to result in jailbreaks. Using tree-of-thought reasoning allows TAP to navigate a large search space of prompts and pruning reduces the total number of queries sent to the target. In empirical evaluations, we observe that TAP generates prompts that jailbreak state-of-the-art LLMs (including GPT4 and GPT4-Turbo) for more than 80% of the prompts using only a small number of queries. This significantly improves upon the previous state-of-the-art black-box method for generating jailbreaks.



## **27. A Survey on Large Language Model (LLM) Security and Privacy: The Good, the Bad, and the Ugly**

cs.CR

**SubmitDate**: 2023-12-04    [abs](http://arxiv.org/abs/2312.02003v1) [paper-pdf](http://arxiv.org/pdf/2312.02003v1)

**Authors**: Yifan Yao, Jinhao Duan, Kaidi Xu, Yuanfang Cai, Eric Sun, Yue Zhang

**Abstract**: Large Language Models (LLMs), such as GPT-3 and BERT, have revolutionized natural language understanding and generation. They possess deep language comprehension, human-like text generation capabilities, contextual awareness, and robust problem-solving skills, making them invaluable in various domains (e.g., search engines, customer support, translation). In the meantime, LLMs have also gained traction in the security community, revealing security vulnerabilities and showcasing their potential in security-related tasks. This paper explores the intersection of LLMs with security and privacy. Specifically, we investigate how LLMs positively impact security and privacy, potential risks and threats associated with their use, and inherent vulnerabilities within LLMs. Through a comprehensive literature review, the paper categorizes findings into "The Good" (beneficial LLM applications), "The Bad" (offensive applications), and "The Ugly" (vulnerabilities and their defenses). We have some interesting findings. For example, LLMs have proven to enhance code and data security, outperforming traditional methods. However, they can also be harnessed for various attacks (particularly user-level attacks) due to their human-like reasoning abilities. We have identified areas that require further research efforts. For example, research on model and parameter extraction attacks is limited and often theoretical, hindered by LLM parameter scale and confidentiality. Safe instruction tuning, a recent development, requires more exploration. We hope that our work can shed light on the LLMs' potential to both bolster and jeopardize cybersecurity.



## **28. Intrusion Detection System with Machine Learning and Multiple Datasets**

cs.CR

12 pages, 2 figures, 2 tables

**SubmitDate**: 2023-12-04    [abs](http://arxiv.org/abs/2312.01941v1) [paper-pdf](http://arxiv.org/pdf/2312.01941v1)

**Authors**: Haiyan Xuan, Mohith Manohar

**Abstract**: As Artificial Intelligence (AI) technologies continue to gain traction in the modern-day world, they ultimately pose an immediate threat to current cybersecurity systems via exploitative methods. Prompt engineering is a relatively new field that explores various prompt designs that can hijack large language models (LLMs). If used by an unethical attacker, it can enable an AI system to offer malicious insights and code to them. In this paper, an enhanced intrusion detection system (IDS) that utilizes machine learning (ML) and hyperparameter tuning is explored, which can improve a model's performance in terms of accuracy and efficacy. Ultimately, this improved system can be used to combat the attacks made by unethical hackers. A standard IDS is solely configured with pre-configured rules and patterns; however, with the utilization of machine learning, implicit and different patterns can be generated through the models' hyperparameter settings and parameters. In addition, the IDS will be equipped with multiple datasets so that the accuracy of the models improves. We evaluate the performance of multiple ML models and their respective hyperparameter settings through various metrics to compare their results to other models and past research work. The results of the proposed multi-dataset integration method yielded an accuracy score of 99.9% when equipped with the XGBoost and random forest classifiers and RandomizedSearchCV hyperparameter technique.



## **29. InstructTA: Instruction-Tuned Targeted Attack for Large Vision-Language Models**

cs.CV

**SubmitDate**: 2023-12-04    [abs](http://arxiv.org/abs/2312.01886v1) [paper-pdf](http://arxiv.org/pdf/2312.01886v1)

**Authors**: Xunguang Wang, Zhenlan Ji, Pingchuan Ma, Zongjie Li, Shuai Wang

**Abstract**: Large vision-language models (LVLMs) have demonstrated their incredible capability in image understanding and response generation. However, this rich visual interaction also makes LVLMs vulnerable to adversarial examples. In this paper, we formulate a novel and practical gray-box attack scenario that the adversary can only access the visual encoder of the victim LVLM, without the knowledge of its prompts (which are often proprietary for service providers and not publicly available) and its underlying large language model (LLM). This practical setting poses challenges to the cross-prompt and cross-model transferability of targeted adversarial attack, which aims to confuse the LVLM to output a response that is semantically similar to the attacker's chosen target text. To this end, we propose an instruction-tuned targeted attack (dubbed InstructTA) to deliver the targeted adversarial attack on LVLMs with high transferability. Initially, we utilize a public text-to-image generative model to "reverse" the target response into a target image, and employ GPT-4 to infer a reasonable instruction $\boldsymbol{p}^\prime$ from the target response. We then form a local surrogate model (sharing the same visual encoder with the victim LVLM) to extract instruction-aware features of an adversarial image example and the target image, and minimize the distance between these two features to optimize the adversarial example. To further improve the transferability, we augment the instruction $\boldsymbol{p}^\prime$ with instructions paraphrased from an LLM. Extensive experiments demonstrate the superiority of our proposed method in targeted attack performance and transferability.



## **30. Warfare:Breaking the Watermark Protection of AI-Generated Content**

cs.CV

**SubmitDate**: 2023-12-04    [abs](http://arxiv.org/abs/2310.07726v2) [paper-pdf](http://arxiv.org/pdf/2310.07726v2)

**Authors**: Guanlin Li, Yifei Chen, Jie Zhang, Jiwei Li, Shangwei Guo, Tianwei Zhang

**Abstract**: AI-Generated Content (AIGC) is gaining great popularity, with many emerging commercial services and applications. These services leverage advanced generative models, such as latent diffusion models and large language models, to generate creative content (e.g., realistic images and fluent sentences) for users. The usage of such generated content needs to be highly regulated, as the service providers need to ensure the users do not violate the usage policies (e.g., abuse for commercialization, generating and distributing unsafe content). A promising solution to achieve this goal is watermarking, which adds unique and imperceptible watermarks on the content for service verification and attribution. Numerous watermarking approaches have been proposed recently. However, in this paper, we show that an adversary can easily break these watermarking mechanisms. Specifically, we consider two possible attacks. (1) Watermark removal: the adversary can easily erase the embedded watermark from the generated content and then use it freely bypassing the regulation of the service provider. (2) Watermark forging: the adversary can create illegal content with forged watermarks from another user, causing the service provider to make wrong attributions. We propose Warfare, a unified methodology to achieve both attacks in a holistic way. The key idea is to leverage a pre-trained diffusion model for content processing and a generative adversarial network for watermark removal or forging. We evaluate Warfare on different datasets and embedding setups. The results prove that it can achieve high success rates while maintaining the quality of the generated content. Compared to existing diffusion model-based attacks, Warfare is 5,050~11,000x faster.



## **31. Unleashing Cheapfakes through Trojan Plugins of Large Language Models**

cs.CR

**SubmitDate**: 2023-12-01    [abs](http://arxiv.org/abs/2312.00374v1) [paper-pdf](http://arxiv.org/pdf/2312.00374v1)

**Authors**: Tian Dong, Guoxing Chen, Shaofeng Li, Minhui Xue, Rayne Holland, Yan Meng, Zhen Liu, Haojin Zhu

**Abstract**: Open-source Large Language Models (LLMs) have recently gained popularity because of their comparable performance to proprietary LLMs. To efficiently fulfill domain-specialized tasks, open-source LLMs can be refined, without expensive accelerators, using low-rank adapters. However, it is still unknown whether low-rank adapters can be exploited to control LLMs. To address this gap, we demonstrate that an infected adapter can induce, on specific triggers, an LLM to output content defined by an adversary and to even maliciously use tools. To train a Trojan adapter, we propose two novel attacks, POLISHED and FUSION, that improve over prior approaches. POLISHED uses LLM-enhanced paraphrasing to polish benchmark poisoned datasets. In contrast, in the absence of a dataset, FUSION leverages an over-poisoning procedure to transform a benign adaptor. Our experiments validate that our attacks provide higher attack effectiveness than the baseline and, for the purpose of attracting downloads, preserves or improves the adapter's utility. Finally, we provide two case studies to demonstrate that the Trojan adapter can lead a LLM-powered autonomous agent to execute unintended scripts or send phishing emails. Our novel attacks represent the first study of supply chain threats for LLMs through the lens of Trojan plugins.



## **32. Ignore This Title and HackAPrompt: Exposing Systemic Vulnerabilities of LLMs through a Global Scale Prompt Hacking Competition**

cs.CR

34 pages, 8 figures Codebase:  https://github.com/PromptLabs/hackaprompt Dataset:  https://huggingface.co/datasets/hackaprompt/hackaprompt-dataset/blob/main/README.md  Playground: https://huggingface.co/spaces/hackaprompt/playground

**SubmitDate**: 2023-11-30    [abs](http://arxiv.org/abs/2311.16119v2) [paper-pdf](http://arxiv.org/pdf/2311.16119v2)

**Authors**: Sander Schulhoff, Jeremy Pinto, Anaum Khan, Louis-François Bouchard, Chenglei Si, Svetlina Anati, Valen Tagliabue, Anson Liu Kost, Christopher Carnahan, Jordan Boyd-Graber

**Abstract**: Large Language Models (LLMs) are deployed in interactive contexts with direct user engagement, such as chatbots and writing assistants. These deployments are vulnerable to prompt injection and jailbreaking (collectively, prompt hacking), in which models are manipulated to ignore their original instructions and follow potentially malicious ones. Although widely acknowledged as a significant security threat, there is a dearth of large-scale resources and quantitative studies on prompt hacking. To address this lacuna, we launch a global prompt hacking competition, which allows for free-form human input attacks. We elicit 600K+ adversarial prompts against three state-of-the-art LLMs. We describe the dataset, which empirically verifies that current LLMs can indeed be manipulated via prompt hacking. We also present a comprehensive taxonomical ontology of the types of adversarial prompts.



## **33. Devising and Detecting Phishing: Large Language Models vs. Smaller Human Models**

cs.CR

**SubmitDate**: 2023-11-30    [abs](http://arxiv.org/abs/2308.12287v2) [paper-pdf](http://arxiv.org/pdf/2308.12287v2)

**Authors**: Fredrik Heiding, Bruce Schneier, Arun Vishwanath, Jeremy Bernstein, Peter S. Park

**Abstract**: AI programs, built using large language models, make it possible to automatically create phishing emails based on a few data points about a user. They stand in contrast to traditional phishing emails that hackers manually design using general rules gleaned from experience. The V-Triad is an advanced set of rules for manually designing phishing emails to exploit our cognitive heuristics and biases. In this study, we compare the performance of phishing emails created automatically by GPT-4 and manually using the V-Triad. We also combine GPT-4 with the V-Triad to assess their combined potential. A fourth group, exposed to generic phishing emails, was our control group. We utilized a factorial approach, sending emails to 112 randomly selected participants recruited for the study. The control group emails received a click-through rate between 19-28%, the GPT-generated emails 30-44%, emails generated by the V-Triad 69-79%, and emails generated by GPT and the V-Triad 43-81%. Each participant was asked to explain why they pressed or did not press a link in the email. These answers often contradict each other, highlighting the need for personalized content. The cues that make one person avoid phishing emails make another person fall for them. Next, we used four popular large language models (GPT, Claude, PaLM, and LLaMA) to detect the intention of phishing emails and compare the results to human detection. The language models demonstrated a strong ability to detect malicious intent, even in non-obvious phishing emails. They sometimes surpassed human detection, although often being slightly less accurate than humans. Finally, we make an analysis of the economic aspects of AI-enabled phishing attacks, showing how large language models can increase the incentives of phishing and spear phishing by reducing their costs.



## **34. Locally Differentially Private Document Generation Using Zero Shot Prompting**

cs.CL

Accepted at EMNLP 2023 (Findings)

**SubmitDate**: 2023-11-30    [abs](http://arxiv.org/abs/2310.16111v2) [paper-pdf](http://arxiv.org/pdf/2310.16111v2)

**Authors**: Saiteja Utpala, Sara Hooker, Pin Yu Chen

**Abstract**: Numerous studies have highlighted the privacy risks associated with pretrained large language models. In contrast, our research offers a unique perspective by demonstrating that pretrained large language models can effectively contribute to privacy preservation. We propose a locally differentially private mechanism called DP-Prompt, which leverages the power of pretrained large language models and zero-shot prompting to counter author de-anonymization attacks while minimizing the impact on downstream utility. When DP-Prompt is used with a powerful language model like ChatGPT (gpt-3.5), we observe a notable reduction in the success rate of de-anonymization attacks, showing that it surpasses existing approaches by a considerable margin despite its simpler design. For instance, in the case of the IMDB dataset, DP-Prompt (with ChatGPT) perfectly recovers the clean sentiment F1 score while achieving a 46\% reduction in author identification F1 score against static attackers and a 26\% reduction against adaptive attackers. We conduct extensive experiments across six open-source large language models, ranging up to 7 billion parameters, to analyze various effects of the privacy-utility tradeoff.



## **35. Towards Safer Generative Language Models: A Survey on Safety Risks, Evaluations, and Improvements**

cs.AI

**SubmitDate**: 2023-11-30    [abs](http://arxiv.org/abs/2302.09270v3) [paper-pdf](http://arxiv.org/pdf/2302.09270v3)

**Authors**: Jiawen Deng, Jiale Cheng, Hao Sun, Zhexin Zhang, Minlie Huang

**Abstract**: As generative large model capabilities advance, safety concerns become more pronounced in their outputs. To ensure the sustainable growth of the AI ecosystem, it's imperative to undertake a holistic evaluation and refinement of associated safety risks. This survey presents a framework for safety research pertaining to large models, delineating the landscape of safety risks as well as safety evaluation and improvement methods. We begin by introducing safety issues of wide concern, then delve into safety evaluation methods for large models, encompassing preference-based testing, adversarial attack approaches, issues detection, and other advanced evaluation methods. Additionally, we explore the strategies for enhancing large model safety from training to deployment, highlighting cutting-edge safety approaches for each stage in building large models. Finally, we discuss the core challenges in advancing towards more responsible AI, including the interpretability of safety mechanisms, ongoing safety issues, and robustness against malicious attacks. Through this survey, we aim to provide clear technical guidance for safety researchers and encourage further study on the safety of large models.



## **36. Leveraging a Randomized Key Matrix to Enhance the Security of Symmetric Substitution Ciphers**

cs.CR

In Proceedings of the 10th IEEE Asia-Pacific Conference on Computer  Science and Data Engineering 2023 (CSDE)

**SubmitDate**: 2023-11-29    [abs](http://arxiv.org/abs/2311.18085v1) [paper-pdf](http://arxiv.org/pdf/2311.18085v1)

**Authors**: Shubham Gandhi, Om Khare, Mihika Dravid, Mihika Sanghvi, Sunil Mane, Aadesh Gajaralwar, Saloni Gandhi

**Abstract**: An innovative strategy to enhance the security of symmetric substitution ciphers is presented, through the implementation of a randomized key matrix suitable for various file formats, including but not limited to binary and text files. Despite their historical relevance, symmetric substitution ciphers have been limited by vulnerabilities to cryptanalytic methods like frequency analysis and known plaintext attacks. The aim of our research is to mitigate these vulnerabilities by employing a polyalphabetic substitution strategy that incorporates a distinct randomized key matrix. This matrix plays a pivotal role in generating a unique random key, comprising characters, encompassing both uppercase and lowercase letters, numeric, and special characters, to derive the corresponding ciphertext. The effectiveness of the proposed methodology in enhancing the security of conventional substitution methods for file encryption and decryption is supported by comprehensive testing and analysis, which encompass computational speed, frequency analysis, keyspace examination, Kasiski test, entropy analysis, and the utilization of a large language model.



## **37. SmoothLLM: Defending Large Language Models Against Jailbreaking Attacks**

cs.LG

**SubmitDate**: 2023-11-29    [abs](http://arxiv.org/abs/2310.03684v3) [paper-pdf](http://arxiv.org/pdf/2310.03684v3)

**Authors**: Alexander Robey, Eric Wong, Hamed Hassani, George J. Pappas

**Abstract**: Despite efforts to align large language models (LLMs) with human values, widely-used LLMs such as GPT, Llama, Claude, and PaLM are susceptible to jailbreaking attacks, wherein an adversary fools a targeted LLM into generating objectionable content. To address this vulnerability, we propose SmoothLLM, the first algorithm designed to mitigate jailbreaking attacks on LLMs. Based on our finding that adversarially-generated prompts are brittle to character-level changes, our defense first randomly perturbs multiple copies of a given input prompt, and then aggregates the corresponding predictions to detect adversarial inputs. SmoothLLM reduces the attack success rate on numerous popular LLMs to below one percentage point, avoids unnecessary conservatism, and admits provable guarantees on attack mitigation. Moreover, our defense uses exponentially fewer queries than existing attacks and is compatible with any LLM. Our code is publicly available at the following link: https://github.com/arobey1/smooth-llm.



## **38. Query-Relevant Images Jailbreak Large Multi-Modal Models**

cs.CV

Technique report

**SubmitDate**: 2023-11-29    [abs](http://arxiv.org/abs/2311.17600v1) [paper-pdf](http://arxiv.org/pdf/2311.17600v1)

**Authors**: Xin Liu, Yichen Zhu, Yunshi Lan, Chao Yang, Yu Qiao

**Abstract**: Warning: This paper contains examples of harmful language and images, and reader discretion is recommended. The security concerns surrounding Large Language Models (LLMs) have been extensively explored, yet the safety of Large Multi-Modal Models (LMMs) remains understudied. In our study, we present a novel visual prompt attack that exploits query-relevant images to jailbreak the open-source LMMs. Our method creates a composite image from one image generated by diffusion models and another that displays the text as typography, based on keywords extracted from a malicious query. We show LLMs can be easily attacked by our approach, even if the employed Large Language Models are safely aligned. To evaluate the extent of this vulnerability in open-source LMMs, we have compiled a substantial dataset encompassing 13 scenarios with a total of 5,040 text-image pairs, using our presented attack technique. Our evaluation of 12 cutting-edge LMMs using this dataset shows the vulnerability of existing multi-modal models on adversarial attacks. This finding underscores the need for a concerted effort to strengthen and enhance the safety measures of open-source LMMs against potential malicious exploits. The resource is available at \href{this https URL}{https://github.com/isXinLiu/MM-SafetyBench}.



## **39. Unveiling the Implicit Toxicity in Large Language Models**

cs.CL

EMNLP 2023 Main Conference

**SubmitDate**: 2023-11-29    [abs](http://arxiv.org/abs/2311.17391v1) [paper-pdf](http://arxiv.org/pdf/2311.17391v1)

**Authors**: Jiaxin Wen, Pei Ke, Hao Sun, Zhexin Zhang, Chengfei Li, Jinfeng Bai, Minlie Huang

**Abstract**: The open-endedness of large language models (LLMs) combined with their impressive capabilities may lead to new safety issues when being exploited for malicious use. While recent studies primarily focus on probing toxic outputs that can be easily detected with existing toxicity classifiers, we show that LLMs can generate diverse implicit toxic outputs that are exceptionally difficult to detect via simply zero-shot prompting. Moreover, we propose a reinforcement learning (RL) based attacking method to further induce the implicit toxicity in LLMs. Specifically, we optimize the language model with a reward that prefers implicit toxic outputs to explicit toxic and non-toxic ones. Experiments on five widely-adopted toxicity classifiers demonstrate that the attack success rate can be significantly improved through RL fine-tuning. For instance, the RL-finetuned LLaMA-13B model achieves an attack success rate of 90.04% on BAD and 62.85% on Davinci003. Our findings suggest that LLMs pose a significant threat in generating undetectable implicit toxic outputs. We further show that fine-tuning toxicity classifiers on the annotated examples from our attacking method can effectively enhance their ability to detect LLM-generated implicit toxic language. The code is publicly available at https://github.com/thu-coai/Implicit-Toxicity.



## **40. Identifying and Mitigating Vulnerabilities in LLM-Integrated Applications**

cs.CR

**SubmitDate**: 2023-11-29    [abs](http://arxiv.org/abs/2311.16153v2) [paper-pdf](http://arxiv.org/pdf/2311.16153v2)

**Authors**: Fengqing Jiang, Zhangchen Xu, Luyao Niu, Boxin Wang, Jinyuan Jia, Bo Li, Radha Poovendran

**Abstract**: Large language models (LLMs) are increasingly deployed as the service backend for LLM-integrated applications such as code completion and AI-powered search. LLM-integrated applications serve as middleware to refine users' queries with domain-specific knowledge to better inform LLMs and enhance the responses. Despite numerous opportunities and benefits, LLM-integrated applications also introduce new attack surfaces. Understanding, minimizing, and eliminating these emerging attack surfaces is a new area of research. In this work, we consider a setup where the user and LLM interact via an LLM-integrated application in the middle. We focus on the communication rounds that begin with user's queries and end with LLM-integrated application returning responses to the queries, powered by LLMs at the service backend. For this query-response protocol, we identify potential vulnerabilities that can originate from the malicious application developer or from an outsider threat initiator that is able to control the database access, manipulate and poison data that are high-risk for the user. Successful exploits of the identified vulnerabilities result in the users receiving responses tailored to the intent of a threat initiator. We assess such threats against LLM-integrated applications empowered by OpenAI GPT-3.5 and GPT-4. Our empirical results show that the threats can effectively bypass the restrictions and moderation policies of OpenAI, resulting in users receiving responses that contain bias, toxic content, privacy risk, and disinformation. To mitigate those threats, we identify and define four key properties, namely integrity, source identification, attack detectability, and utility preservation, that need to be satisfied by a safe LLM-integrated application. Based on these properties, we develop a lightweight, threat-agnostic defense that mitigates both insider and outsider threats.



## **41. ZTCloudGuard: Zero Trust Context-Aware Access Management Framework to Avoid Misuse Cases in the Era of Generative AI and Cloud-based Health Information Ecosystem**

cs.CR

**SubmitDate**: 2023-11-28    [abs](http://arxiv.org/abs/2312.02993v1) [paper-pdf](http://arxiv.org/pdf/2312.02993v1)

**Authors**: Khalid Al-hammuri, Fayez Gebali, Awos Kanan

**Abstract**: Managing access between large numbers of distributed medical devices has become a crucial aspect of modern healthcare systems, enabling the establishment of smart hospitals and telehealth infrastructure. However, as telehealth technology continues to evolve and Internet of Things (IoT) devices become more widely used, they are also becoming increasingly exposed to various types of vulnerabilities and medical errors. In healthcare information systems, about 90\% of vulnerabilities emerged from misuse cases and human errors. As a result, there is a need for additional research and development of security tools to prevent such attacks. This article proposes a zero-trust-based context-aware framework for managing access to the main components of the cloud ecosystem, including users, devices and output data. The main goal and benefit of the proposed framework is to build a scoring system to prevent or alleviate misuse cases while using distributed medical devices in cloud-based healthcare information systems. The framework has two main scoring schemas to maintain the chain of trust. First, it proposes a critical trust score based on cloud-native micro-services of authentication, encryption, logging, and authorizations. Second, creating a bond trust scoring to assess the real-time semantic and syntactic analysis of attributes stored in a healthcare information system. The analysis is based on a pre-trained machine learning model to generate the semantic and syntactic scores. The framework also takes into account regulatory compliance and user consent to create a scoring system. The advantage of this method is that it is applicable to any language and adapts to all attributes as it relies on a language model, not just a set of predefined and limited attributes. The results show a high F1 score of 93.5%, which proves that it is valid for detecting misuse cases.



## **42. Certifying LLM Safety against Adversarial Prompting**

cs.CL

**SubmitDate**: 2023-11-28    [abs](http://arxiv.org/abs/2309.02705v2) [paper-pdf](http://arxiv.org/pdf/2309.02705v2)

**Authors**: Aounon Kumar, Chirag Agarwal, Suraj Srinivas, Aaron Jiaxun Li, Soheil Feizi, Himabindu Lakkaraju

**Abstract**: Large language models (LLMs) released for public use incorporate guardrails to ensure their output is safe, often referred to as "model alignment." An aligned language model should decline a user's request to produce harmful content. However, such safety measures are vulnerable to adversarial attacks, which add maliciously designed token sequences to a harmful prompt to bypass the model's safety guards. In this work, we introduce erase-and-check, the first framework to defend against adversarial prompts with verifiable safety guarantees. We defend against three attack modes: i) adversarial suffix, which appends an adversarial sequence at the end of the prompt; ii) adversarial insertion, where the adversarial sequence is inserted anywhere in the middle of the prompt; and iii) adversarial infusion, where adversarial tokens are inserted at arbitrary positions in the prompt, not necessarily as a contiguous block. Our experimental results demonstrate that this procedure can obtain strong certified safety guarantees on harmful prompts while maintaining good empirical performance on safe prompts. For example, against adversarial suffixes of length 20, it certifiably detects 92% of harmful prompts and labels 94% of safe prompts correctly using the open-source language model Llama 2 as the safety filter. We further improve the filter's performance, in terms of accuracy and speed, by replacing Llama 2 with a DistilBERT safety classifier fine-tuned on safe and harmful prompts. Additionally, we propose two efficient empirical defenses: i) RandEC, a randomized version of erase-and-check that evaluates the safety filter on a small subset of the erased subsequences, and ii) GradEC, a gradient-based version that optimizes the erased tokens to remove the adversarial sequence. The code for our experiments is available at https://github.com/aounon/certified-llm-safety.



## **43. C-ITS Environment Modeling and Attack Modeling**

cs.CR

in Korean Language, 14 Figures, 15 Pages

**SubmitDate**: 2023-11-27    [abs](http://arxiv.org/abs/2311.14327v2) [paper-pdf](http://arxiv.org/pdf/2311.14327v2)

**Authors**: Jaewoong Choi, Min Geun Song, Hyosun Lee, Chaeyeon Sagong, Sangbeom Park, Jaesung Lee, Jeong Do Yoo, Huy Kang Kim

**Abstract**: As technology advances, cities are evolving into smart cities, with the ability to process large amounts of data and the increasing complexity and diversification of various elements within urban areas. Among the core systems of a smart city is the Cooperative-Intelligent Transport Systems (C-ITS). C-ITS is a system where vehicles provide real-time information to drivers about surrounding traffic conditions, sudden stops, falling objects, and other accident risks through roadside base stations. It consists of road infrastructure, C-ITS centers, and vehicle terminals. However, as smart cities integrate many elements through networks and electronic control, they are susceptible to cybersecurity issues. In the case of cybersecurity problems in C-ITS, there is a significant risk of safety issues arising. This technical document aims to model the C-ITS environment and the services it provides, with the purpose of identifying the attack surface where security incidents could occur in a smart city environment. Subsequently, based on the identified attack surface, the document aims to construct attack scenarios and their respective stages. The document provides a description of the concept of C-ITS, followed by the description of the C-ITS environment model, service model, and attack scenario model defined by us.



## **44. Token-Level Adversarial Prompt Detection Based on Perplexity Measures and Contextual Information**

cs.CL

**SubmitDate**: 2023-11-27    [abs](http://arxiv.org/abs/2311.11509v2) [paper-pdf](http://arxiv.org/pdf/2311.11509v2)

**Authors**: Zhengmian Hu, Gang Wu, Saayan Mitra, Ruiyi Zhang, Tong Sun, Heng Huang, Viswanathan Swaminathan

**Abstract**: In recent years, Large Language Models (LLM) have emerged as pivotal tools in various applications. However, these models are susceptible to adversarial prompt attacks, where attackers can carefully curate input strings that lead to undesirable outputs. The inherent vulnerability of LLMs stems from their input-output mechanisms, especially when presented with intensely out-of-distribution (OOD) inputs. This paper proposes a token-level detection method to identify adversarial prompts, leveraging the LLM's capability to predict the next token's probability. We measure the degree of the model's perplexity and incorporate neighboring token information to encourage the detection of contiguous adversarial prompt sequences. As a result, we propose two methods: one that identifies each token as either being part of an adversarial prompt or not, and another that estimates the probability of each token being part of an adversarial prompt.



## **45. BadCLIP: Trigger-Aware Prompt Learning for Backdoor Attacks on CLIP**

cs.CV

13 pages, 5 figures

**SubmitDate**: 2023-11-26    [abs](http://arxiv.org/abs/2311.16194v1) [paper-pdf](http://arxiv.org/pdf/2311.16194v1)

**Authors**: Jiawang Bai, Kuofeng Gao, Shaobo Min, Shu-Tao Xia, Zhifeng Li, Wei Liu

**Abstract**: Contrastive Vision-Language Pre-training, known as CLIP, has shown promising effectiveness in addressing downstream image recognition tasks. However, recent works revealed that the CLIP model can be implanted with a downstream-oriented backdoor. On downstream tasks, one victim model performs well on clean samples but predicts a specific target class whenever a specific trigger is present. For injecting a backdoor, existing attacks depend on a large amount of additional data to maliciously fine-tune the entire pre-trained CLIP model, which makes them inapplicable to data-limited scenarios. In this work, motivated by the recent success of learnable prompts, we address this problem by injecting a backdoor into the CLIP model in the prompt learning stage. Our method named BadCLIP is built on a novel and effective mechanism in backdoor attacks on CLIP, i.e., influencing both the image and text encoders with the trigger. It consists of a learnable trigger applied to images and a trigger-aware context generator, such that the trigger can change text features via trigger-aware prompts, resulting in a powerful and generalizable attack. Extensive experiments conducted on 11 datasets verify that the clean accuracy of BadCLIP is similar to those of advanced prompt learning methods and the attack success rate is higher than 99% in most cases. BadCLIP is also generalizable to unseen classes, and shows a strong generalization capability under cross-dataset and cross-domain settings.



## **46. Evaluating the Instruction-Following Robustness of Large Language Models to Prompt Injection**

cs.CL

The data and code can be found at  https://github.com/Leezekun/instruction-following-robustness-eval

**SubmitDate**: 2023-11-25    [abs](http://arxiv.org/abs/2308.10819v3) [paper-pdf](http://arxiv.org/pdf/2308.10819v3)

**Authors**: Zekun Li, Baolin Peng, Pengcheng He, Xifeng Yan

**Abstract**: Large Language Models (LLMs) have demonstrated exceptional proficiency in instruction-following, becoming increasingly crucial across various applications. However, this capability brings with it the risk of prompt injection attacks, where attackers inject instructions into LLMs' input to elicit undesirable actions or content. Understanding the robustness of LLMs against such attacks is vital for their safe implementation. In this work, we establish a benchmark to evaluate the robustness of instruction-following LLMs against prompt injection attacks. Our objective is to determine the extent to which LLMs can be influenced by injected instructions and their ability to differentiate between these injected and original target instructions. Through extensive experiments with leading instruction-following LLMs, we uncover significant vulnerabilities in their robustness to such attacks. Our results indicate that some models are overly tuned to follow any embedded instructions in the prompt, overly focusing on the latter parts of the prompt without fully grasping the entire context. By contrast, models with a better grasp of the context and instruction-following capabilities will potentially be more susceptible to compromise by injected instructions. This underscores the need to shift the focus from merely enhancing LLMs' instruction-following capabilities to improving their overall comprehension of prompts and discernment of instructions that are appropriate to follow. We hope our in-depth analysis offers insights into the underlying causes of these vulnerabilities, aiding in the development of future solutions. Code and data are available at https://github.com/Leezekun/instruction-following-robustness-eval



## **47. Exploiting Large Language Models (LLMs) through Deception Techniques and Persuasion Principles**

cs.HC

10 pages, 16 tables, 5 figures, IEEE BigData 2023 (Workshops)

**SubmitDate**: 2023-11-24    [abs](http://arxiv.org/abs/2311.14876v1) [paper-pdf](http://arxiv.org/pdf/2311.14876v1)

**Authors**: Sonali Singh, Faranak Abri, Akbar Siami Namin

**Abstract**: With the recent advent of Large Language Models (LLMs), such as ChatGPT from OpenAI, BARD from Google, Llama2 from Meta, and Claude from Anthropic AI, gain widespread use, ensuring their security and robustness is critical. The widespread use of these language models heavily relies on their reliability and proper usage of this fascinating technology. It is crucial to thoroughly test these models to not only ensure its quality but also possible misuses of such models by potential adversaries for illegal activities such as hacking. This paper presents a novel study focusing on exploitation of such large language models against deceptive interactions. More specifically, the paper leverages widespread and borrows well-known techniques in deception theory to investigate whether these models are susceptible to deceitful interactions.   This research aims not only to highlight these risks but also to pave the way for robust countermeasures that enhance the security and integrity of language models in the face of sophisticated social engineering tactics. Through systematic experiments and analysis, we assess their performance in these critical security domains. Our results demonstrate a significant finding in that these large language models are susceptible to deception and social engineering attacks.



## **48. Backdoor Activation Attack: Attack Large Language Models using Activation Steering for Safety-Alignment**

cs.CR

**SubmitDate**: 2023-11-24    [abs](http://arxiv.org/abs/2311.09433v2) [paper-pdf](http://arxiv.org/pdf/2311.09433v2)

**Authors**: Haoran Wang, Kai Shu

**Abstract**: To ensure AI safety, instruction-tuned Large Language Models (LLMs) are specifically trained to ensure alignment, which refers to making models behave in accordance with human intentions. While these models have demonstrated commendable results on various safety benchmarks, the vulnerability of their safety alignment has not been extensively studied. This is particularly troubling given the potential harm that LLMs can inflict. Existing attack methods on LLMs often rely on poisoned training data or the injection of malicious prompts. These approaches compromise the stealthiness and generalizability of the attacks, making them susceptible to detection. Additionally, these models often demand substantial computational resources for implementation, making them less practical for real-world applications. Inspired by recent success in modifying model behavior through steering vectors without the need for optimization, and drawing on its effectiveness in red-teaming LLMs, we conducted experiments employing activation steering to target four key aspects of LLMs: truthfulness, toxicity, bias, and harmfulness - across a varied set of attack settings. To establish a universal attack strategy applicable to diverse target alignments without depending on manual analysis, we automatically select the intervention layer based on contrastive layer search. Our experiment results show that activation attacks are highly effective and add little or no overhead to attack efficiency. Additionally, we discuss potential countermeasures against such activation attacks. Our code and data are available at https://github.com/wang2226/Backdoor-Activation-Attack Warning: this paper contains content that can be offensive or upsetting.



## **49. Universal Jailbreak Backdoors from Poisoned Human Feedback**

cs.AI

**SubmitDate**: 2023-11-24    [abs](http://arxiv.org/abs/2311.14455v1) [paper-pdf](http://arxiv.org/pdf/2311.14455v1)

**Authors**: Javier Rando, Florian Tramèr

**Abstract**: Reinforcement Learning from Human Feedback (RLHF) is used to align large language models to produce helpful and harmless responses. Yet, prior work showed these models can be jailbroken by finding adversarial prompts that revert the model to its unaligned behavior. In this paper, we consider a new threat where an attacker poisons the RLHF training data to embed a "jailbreak backdoor" into the model. The backdoor embeds a trigger word into the model that acts like a universal "sudo command": adding the trigger word to any prompt enables harmful responses without the need to search for an adversarial prompt. Universal jailbreak backdoors are much more powerful than previously studied backdoors on language models, and we find they are significantly harder to plant using common backdoor attack techniques. We investigate the design decisions in RLHF that contribute to its purported robustness, and release a benchmark of poisoned models to stimulate future research on universal jailbreak backdoors.



## **50. Scalable and Transferable Black-Box Jailbreaks for Language Models via Persona Modulation**

cs.CL

**SubmitDate**: 2023-11-24    [abs](http://arxiv.org/abs/2311.03348v2) [paper-pdf](http://arxiv.org/pdf/2311.03348v2)

**Authors**: Rusheb Shah, Quentin Feuillade--Montixi, Soroush Pour, Arush Tagade, Stephen Casper, Javier Rando

**Abstract**: Despite efforts to align large language models to produce harmless responses, they are still vulnerable to jailbreak prompts that elicit unrestricted behaviour. In this work, we investigate persona modulation as a black-box jailbreaking method to steer a target model to take on personalities that are willing to comply with harmful instructions. Rather than manually crafting prompts for each persona, we automate the generation of jailbreaks using a language model assistant. We demonstrate a range of harmful completions made possible by persona modulation, including detailed instructions for synthesising methamphetamine, building a bomb, and laundering money. These automated attacks achieve a harmful completion rate of 42.5% in GPT-4, which is 185 times larger than before modulation (0.23%). These prompts also transfer to Claude 2 and Vicuna with harmful completion rates of 61.0% and 35.9%, respectively. Our work reveals yet another vulnerability in commercial large language models and highlights the need for more comprehensive safeguards.



