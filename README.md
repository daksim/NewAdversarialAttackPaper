# Latest Adversarial Attack Papers
**update at 2024-07-18 16:50:48**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. Investigating Adversarial Vulnerability and Implicit Bias through Frequency Analysis**

cs.LG

**SubmitDate**: 2024-07-17    [abs](http://arxiv.org/abs/2305.15203v2) [paper-pdf](http://arxiv.org/pdf/2305.15203v2)

**Authors**: Lorenzo Basile, Nikos Karantzas, Alberto D'Onofrio, Luca Bortolussi, Alex Rodriguez, Fabio Anselmi

**Abstract**: Despite their impressive performance in classification tasks, neural networks are known to be vulnerable to adversarial attacks, subtle perturbations of the input data designed to deceive the model. In this work, we investigate the relation between these perturbations and the implicit bias of neural networks trained with gradient-based algorithms. To this end, we analyse the network's implicit bias through the lens of the Fourier transform. Specifically, we identify the minimal and most critical frequencies necessary for accurate classification or misclassification respectively for each input image and its adversarially perturbed version, and uncover the correlation among those. To this end, among other methods, we use a newly introduced technique capable of detecting non-linear correlations between high-dimensional datasets. Our results provide empirical evidence that the network bias in Fourier space and the target frequencies of adversarial attacks are highly correlated and suggest new potential strategies for adversarial defence.



## **2. Muting Whisper: A Universal Acoustic Adversarial Attack on Speech Foundation Models**

cs.CL

**SubmitDate**: 2024-07-17    [abs](http://arxiv.org/abs/2405.06134v2) [paper-pdf](http://arxiv.org/pdf/2405.06134v2)

**Authors**: Vyas Raina, Rao Ma, Charles McGhee, Kate Knill, Mark Gales

**Abstract**: Recent developments in large speech foundation models like Whisper have led to their widespread use in many automatic speech recognition (ASR) applications. These systems incorporate `special tokens' in their vocabulary, such as $\texttt{<|endoftext|>}$, to guide their language generation process. However, we demonstrate that these tokens can be exploited by adversarial attacks to manipulate the model's behavior. We propose a simple yet effective method to learn a universal acoustic realization of Whisper's $\texttt{<|endoftext|>}$ token, which, when prepended to any speech signal, encourages the model to ignore the speech and only transcribe the special token, effectively `muting' the model. Our experiments demonstrate that the same, universal 0.64-second adversarial audio segment can successfully mute a target Whisper ASR model for over 97\% of speech samples. Moreover, we find that this universal adversarial audio segment often transfers to new datasets and tasks. Overall this work demonstrates the vulnerability of Whisper models to `muting' adversarial attacks, where such attacks can pose both risks and potential benefits in real-world settings: for example the attack can be used to bypass speech moderation systems, or conversely the attack can also be used to protect private speech data.



## **3. Similarity of Neural Architectures using Adversarial Attack Transferability**

cs.LG

ECCV 2024; 35pages, 2.56MB

**SubmitDate**: 2024-07-17    [abs](http://arxiv.org/abs/2210.11407v4) [paper-pdf](http://arxiv.org/pdf/2210.11407v4)

**Authors**: Jaehui Hwang, Dongyoon Han, Byeongho Heo, Song Park, Sanghyuk Chun, Jong-Seok Lee

**Abstract**: In recent years, many deep neural architectures have been developed for image classification. Whether they are similar or dissimilar and what factors contribute to their (dis)similarities remains curious. To address this question, we aim to design a quantitative and scalable similarity measure between neural architectures. We propose Similarity by Attack Transferability (SAT) from the observation that adversarial attack transferability contains information related to input gradients and decision boundaries widely used to understand model behaviors. We conduct a large-scale analysis on 69 state-of-the-art ImageNet classifiers using our proposed similarity function to answer the question. Moreover, we observe neural architecture-related phenomena using model similarity that model diversity can lead to better performance on model ensembles and knowledge distillation under specific conditions. Our results provide insights into why developing diverse neural architectures with distinct components is necessary.



## **4. Benchmarking Robust Self-Supervised Learning Across Diverse Downstream Tasks**

cs.CV

Accepted at the ICML 2024 Workshop on Foundation Models in the Wild

**SubmitDate**: 2024-07-17    [abs](http://arxiv.org/abs/2407.12588v1) [paper-pdf](http://arxiv.org/pdf/2407.12588v1)

**Authors**: Antoni Kowalczuk, Jan Dubiński, Atiyeh Ashari Ghomi, Yi Sui, George Stein, Jiapeng Wu, Jesse C. Cresswell, Franziska Boenisch, Adam Dziedzic

**Abstract**: Large-scale vision models have become integral in many applications due to their unprecedented performance and versatility across downstream tasks. However, the robustness of these foundation models has primarily been explored for a single task, namely image classification. The vulnerability of other common vision tasks, such as semantic segmentation and depth estimation, remains largely unknown. We present a comprehensive empirical evaluation of the adversarial robustness of self-supervised vision encoders across multiple downstream tasks. Our attacks operate in the encoder embedding space and at the downstream task output level. In both cases, current state-of-the-art adversarial fine-tuning techniques tested only for classification significantly degrade clean and robust performance on other tasks. Since the purpose of a foundation model is to cater to multiple applications at once, our findings reveal the need to enhance encoder robustness more broadly. %We discuss potential strategies for more robust foundation vision models across diverse downstream tasks. Our code is available at $\href{https://github.com/layer6ai-labs/ssl-robustness}{github.com/layer6ai-labs/ssl-robustness}$.



## **5. Open-Vocabulary Object Detectors: Robustness Challenges under Distribution Shifts**

cs.CV

14 + 3 single column pages

**SubmitDate**: 2024-07-17    [abs](http://arxiv.org/abs/2405.14874v3) [paper-pdf](http://arxiv.org/pdf/2405.14874v3)

**Authors**: Prakash Chandra Chhipa, Kanjar De, Meenakshi Subhash Chippa, Rajkumar Saini, Marcus Liwicki

**Abstract**: The challenge of Out-Of-Distribution (OOD) robustness remains a critical hurdle towards deploying deep vision models. Vision-Language Models (VLMs) have recently achieved groundbreaking results. VLM-based open-vocabulary object detection extends the capabilities of traditional object detection frameworks, enabling the recognition and classification of objects beyond predefined categories. Investigating OOD robustness in recent open-vocabulary object detection is essential to increase the trustworthiness of these models. This study presents a comprehensive robustness evaluation of the zero-shot capabilities of three recent open-vocabulary (OV) foundation object detection models: OWL-ViT, YOLO World, and Grounding DINO. Experiments carried out on the robustness benchmarks COCO-O, COCO-DC, and COCO-C encompassing distribution shifts due to information loss, corruption, adversarial attacks, and geometrical deformation, highlighting the challenges of the model's robustness to foster the research for achieving robustness. Source code shall be made available to the research community on GitHub.



## **6. Preventing Catastrophic Overfitting in Fast Adversarial Training: A Bi-level Optimization Perspective**

cs.LG

**SubmitDate**: 2024-07-17    [abs](http://arxiv.org/abs/2407.12443v1) [paper-pdf](http://arxiv.org/pdf/2407.12443v1)

**Authors**: Zhaoxin Wang, Handing Wang, Cong Tian, Yaochu Jin

**Abstract**: Adversarial training (AT) has become an effective defense method against adversarial examples (AEs) and it is typically framed as a bi-level optimization problem. Among various AT methods, fast AT (FAT), which employs a single-step attack strategy to guide the training process, can achieve good robustness against adversarial attacks at a low cost. However, FAT methods suffer from the catastrophic overfitting problem, especially on complex tasks or with large-parameter models. In this work, we propose a FAT method termed FGSM-PCO, which mitigates catastrophic overfitting by averting the collapse of the inner optimization problem in the bi-level optimization process. FGSM-PCO generates current-stage AEs from the historical AEs and incorporates them into the training process using an adaptive mechanism. This mechanism determines an appropriate fusion ratio according to the performance of the AEs on the training model. Coupled with a loss function tailored to the training framework, FGSM-PCO can alleviate catastrophic overfitting and help the recovery of an overfitted model to effective training. We evaluate our algorithm across three models and three datasets to validate its effectiveness. Comparative empirical studies against other FAT algorithms demonstrate that our proposed method effectively addresses unresolved overfitting issues in existing algorithms.



## **7. DIFFender: Diffusion-Based Adversarial Defense against Patch Attacks**

cs.CV

**SubmitDate**: 2024-07-17    [abs](http://arxiv.org/abs/2306.09124v4) [paper-pdf](http://arxiv.org/pdf/2306.09124v4)

**Authors**: Caixin Kang, Yinpeng Dong, Zhengyi Wang, Shouwei Ruan, Yubo Chen, Hang Su, Xingxing Wei

**Abstract**: Adversarial attacks, particularly patch attacks, pose significant threats to the robustness and reliability of deep learning models. Developing reliable defenses against patch attacks is crucial for real-world applications. This paper introduces DIFFender, a novel defense framework that harnesses the capabilities of a text-guided diffusion model to combat patch attacks. Central to our approach is the discovery of the Adversarial Anomaly Perception (AAP) phenomenon, which empowers the diffusion model to detect and localize adversarial patches through the analysis of distributional discrepancies. DIFFender integrates dual tasks of patch localization and restoration within a single diffusion model framework, utilizing their close interaction to enhance defense efficacy. Moreover, DIFFender utilizes vision-language pre-training coupled with an efficient few-shot prompt-tuning algorithm, which streamlines the adaptation of the pre-trained diffusion model to defense tasks, thus eliminating the need for extensive retraining. Our comprehensive evaluation spans image classification and face recognition tasks, extending to real-world scenarios, where DIFFender shows good robustness against adversarial attacks. The versatility and generalizability of DIFFender are evident across a variety of settings, classifiers, and attack methodologies, marking an advancement in adversarial patch defense strategies.



## **8. Bribe & Fork: Cheap Bribing Attacks via Forking Threat**

cs.CR

This is a full version of the paper Bribe & Fork: Cheap Bribing  Attacks via Forking Threat which was accepted to AFT'24

**SubmitDate**: 2024-07-17    [abs](http://arxiv.org/abs/2402.01363v2) [paper-pdf](http://arxiv.org/pdf/2402.01363v2)

**Authors**: Zeta Avarikioti, Paweł Kędzior, Tomasz Lizurej, Tomasz Michalak

**Abstract**: In this work, we reexamine the vulnerability of Payment Channel Networks (PCNs) to bribing attacks, where an adversary incentivizes blockchain miners to deliberately ignore a specific transaction to undermine the punishment mechanism of PCNs. While previous studies have posited a prohibitive cost for such attacks, we show that this cost may be dramatically reduced (to approximately \$125), thereby increasing the likelihood of these attacks. To this end, we introduce Bribe & Fork, a modified bribing attack that leverages the threat of a so-called feather fork which we analyze with a novel formal model for the mining game with forking. We empirically analyze historical data of some real-world blockchain implementations to evaluate the scale of this cost reduction. Our findings shed more light on the potential vulnerability of PCNs and highlight the need for robust solutions.



## **9. Revisiting the Adversarial Robustness of Vision Language Models: a Multimodal Perspective**

cs.CV

16 pages, 14 figures

**SubmitDate**: 2024-07-17    [abs](http://arxiv.org/abs/2404.19287v2) [paper-pdf](http://arxiv.org/pdf/2404.19287v2)

**Authors**: Wanqi Zhou, Shuanghao Bai, Qibin Zhao, Badong Chen

**Abstract**: Pretrained vision-language models (VLMs) like CLIP have shown impressive generalization performance across various downstream tasks, yet they remain vulnerable to adversarial attacks. While prior research has primarily concentrated on improving the adversarial robustness of image encoders to guard against attacks on images, the exploration of text-based and multimodal attacks has largely been overlooked. In this work, we initiate the first known and comprehensive effort to study adapting vision-language models for adversarial robustness under the multimodal attack. Firstly, we introduce a multimodal attack strategy and investigate the impact of different attacks. We then propose a multimodal contrastive adversarial training loss, aligning the clean and adversarial text embeddings with the adversarial and clean visual features, to enhance the adversarial robustness of both image and text encoders of CLIP. Extensive experiments on 15 datasets across two tasks demonstrate that our method significantly improves the adversarial robustness of CLIP. Interestingly, we find that the model fine-tuned against multimodal adversarial attacks exhibits greater robustness than its counterpart fine-tuned solely against image-based attacks, even in the context of image attacks, which may open up new possibilities for enhancing the security of VLMs.



## **10. Augmented Neural Fine-Tuning for Efficient Backdoor Purification**

cs.CV

Accepted to ECCV 2024

**SubmitDate**: 2024-07-17    [abs](http://arxiv.org/abs/2407.10052v2) [paper-pdf](http://arxiv.org/pdf/2407.10052v2)

**Authors**: Nazmul Karim, Abdullah Al Arafat, Umar Khalid, Zhishan Guo, Nazanin Rahnavard

**Abstract**: Recent studies have revealed the vulnerability of deep neural networks (DNNs) to various backdoor attacks, where the behavior of DNNs can be compromised by utilizing certain types of triggers or poisoning mechanisms. State-of-the-art (SOTA) defenses employ too-sophisticated mechanisms that require either a computationally expensive adversarial search module for reverse-engineering the trigger distribution or an over-sensitive hyper-parameter selection module. Moreover, they offer sub-par performance in challenging scenarios, e.g., limited validation data and strong attacks. In this paper, we propose Neural mask Fine-Tuning (NFT) with an aim to optimally re-organize the neuron activities in a way that the effect of the backdoor is removed. Utilizing a simple data augmentation like MixUp, NFT relaxes the trigger synthesis process and eliminates the requirement of the adversarial search module. Our study further reveals that direct weight fine-tuning under limited validation data results in poor post-purification clean test accuracy, primarily due to overfitting issue. To overcome this, we propose to fine-tune neural masks instead of model weights. In addition, a mask regularizer has been devised to further mitigate the model drift during the purification process. The distinct characteristics of NFT render it highly efficient in both runtime and sample usage, as it can remove the backdoor even when a single sample is available from each class. We validate the effectiveness of NFT through extensive experiments covering the tasks of image classification, object detection, video action recognition, 3D point cloud, and natural language processing. We evaluate our method against 14 different attacks (LIRA, WaNet, etc.) on 11 benchmark data sets such as ImageNet, UCF101, Pascal VOC, ModelNet, OpenSubtitles2012, etc.



## **11. Asymmetric Bias in Text-to-Image Generation with Adversarial Attacks**

cs.LG

camera-ready version

**SubmitDate**: 2024-07-17    [abs](http://arxiv.org/abs/2312.14440v3) [paper-pdf](http://arxiv.org/pdf/2312.14440v3)

**Authors**: Haz Sameen Shahgir, Xianghao Kong, Greg Ver Steeg, Yue Dong

**Abstract**: The widespread use of Text-to-Image (T2I) models in content generation requires careful examination of their safety, including their robustness to adversarial attacks. Despite extensive research on adversarial attacks, the reasons for their effectiveness remain underexplored. This paper presents an empirical study on adversarial attacks against T2I models, focusing on analyzing factors associated with attack success rates (ASR). We introduce a new attack objective - entity swapping using adversarial suffixes and two gradient-based attack algorithms. Human and automatic evaluations reveal the asymmetric nature of ASRs on entity swap: for example, it is easier to replace "human" with "robot" in the prompt "a human dancing in the rain." with an adversarial suffix, but the reverse replacement is significantly harder. We further propose probing metrics to establish indicative signals from the model's beliefs to the adversarial ASR. We identify conditions that result in a success probability of 60% for adversarial attacks and others where this likelihood drops below 5%.



## **12. Any Target Can be Offense: Adversarial Example Generation via Generalized Latent Infection**

cs.CV

ECCV 2024

**SubmitDate**: 2024-07-17    [abs](http://arxiv.org/abs/2407.12292v1) [paper-pdf](http://arxiv.org/pdf/2407.12292v1)

**Authors**: Youheng Sun, Shengming Yuan, Xuanhan Wang, Lianli Gao, Jingkuan Song

**Abstract**: Targeted adversarial attack, which aims to mislead a model to recognize any image as a target object by imperceptible perturbations, has become a mainstream tool for vulnerability assessment of deep neural networks (DNNs). Since existing targeted attackers only learn to attack known target classes, they cannot generalize well to unknown classes. To tackle this issue, we propose $\bf{G}$eneralized $\bf{A}$dversarial attac$\bf{KER}$ ($\bf{GAKer}$), which is able to construct adversarial examples to any target class. The core idea behind GAKer is to craft a latently infected representation during adversarial example generation. To this end, the extracted latent representations of the target object are first injected into intermediate features of an input image in an adversarial generator. Then, the generator is optimized to ensure visual consistency with the input image while being close to the target object in the feature space. Since the GAKer is class-agnostic yet model-agnostic, it can be regarded as a general tool that not only reveals the vulnerability of more DNNs but also identifies deficiencies of DNNs in a wider range of classes. Extensive experiments have demonstrated the effectiveness of our proposed method in generating adversarial examples for both known and unknown classes. Notably, compared with other generative methods, our method achieves an approximately $14.13\%$ higher attack success rate for unknown classes and an approximately $4.23\%$ higher success rate for known classes. Our code is available in https://github.com/VL-Group/GAKer.



## **13. Does Refusal Training in LLMs Generalize to the Past Tense?**

cs.CL

Code and jailbreak artifacts:  https://github.com/tml-epfl/llm-past-tense

**SubmitDate**: 2024-07-16    [abs](http://arxiv.org/abs/2407.11969v1) [paper-pdf](http://arxiv.org/pdf/2407.11969v1)

**Authors**: Maksym Andriushchenko, Nicolas Flammarion

**Abstract**: Refusal training is widely used to prevent LLMs from generating harmful, undesirable, or illegal outputs. We reveal a curious generalization gap in the current refusal training approaches: simply reformulating a harmful request in the past tense (e.g., "How to make a Molotov cocktail?" to "How did people make a Molotov cocktail?") is often sufficient to jailbreak many state-of-the-art LLMs. We systematically evaluate this method on Llama-3 8B, GPT-3.5 Turbo, Gemma-2 9B, Phi-3-Mini, GPT-4o, and R2D2 models using GPT-3.5 Turbo as a reformulation model. For example, the success rate of this simple attack on GPT-4o increases from 1% using direct requests to 88% using 20 past tense reformulation attempts on harmful requests from JailbreakBench with GPT-4 as a jailbreak judge. Interestingly, we also find that reformulations in the future tense are less effective, suggesting that refusal guardrails tend to consider past historical questions more benign than hypothetical future questions. Moreover, our experiments on fine-tuning GPT-3.5 Turbo show that defending against past reformulations is feasible when past tense examples are explicitly included in the fine-tuning data. Overall, our findings highlight that the widely used alignment techniques -- such as SFT, RLHF, and adversarial training -- employed to align the studied models can be brittle and do not always generalize as intended. We provide code and jailbreak artifacts at https://github.com/tml-epfl/llm-past-tense.



## **14. JailbreakBench: An Open Robustness Benchmark for Jailbreaking Large Language Models**

cs.CR

JailbreakBench v1.0: more attack artifacts, more test-time defenses,  a more accurate jailbreak judge (Llama-3-70B with a custom prompt), a larger  dataset of human preferences for selecting a jailbreak judge (300 examples),  an over-refusal evaluation dataset (100 benign/borderline behaviors), a  semantic refusal judge based on Llama-3-8B

**SubmitDate**: 2024-07-16    [abs](http://arxiv.org/abs/2404.01318v4) [paper-pdf](http://arxiv.org/pdf/2404.01318v4)

**Authors**: Patrick Chao, Edoardo Debenedetti, Alexander Robey, Maksym Andriushchenko, Francesco Croce, Vikash Sehwag, Edgar Dobriban, Nicolas Flammarion, George J. Pappas, Florian Tramer, Hamed Hassani, Eric Wong

**Abstract**: Jailbreak attacks cause large language models (LLMs) to generate harmful, unethical, or otherwise objectionable content. Evaluating these attacks presents a number of challenges, which the current collection of benchmarks and evaluation techniques do not adequately address. First, there is no clear standard of practice regarding jailbreaking evaluation. Second, existing works compute costs and success rates in incomparable ways. And third, numerous works are not reproducible, as they withhold adversarial prompts, involve closed-source code, or rely on evolving proprietary APIs. To address these challenges, we introduce JailbreakBench, an open-sourced benchmark with the following components: (1) an evolving repository of state-of-the-art adversarial prompts, which we refer to as jailbreak artifacts; (2) a jailbreaking dataset comprising 100 behaviors -- both original and sourced from prior work (Zou et al., 2023; Mazeika et al., 2023, 2024) -- which align with OpenAI's usage policies; (3) a standardized evaluation framework at https://github.com/JailbreakBench/jailbreakbench that includes a clearly defined threat model, system prompts, chat templates, and scoring functions; and (4) a leaderboard at https://jailbreakbench.github.io/ that tracks the performance of attacks and defenses for various LLMs. We have carefully considered the potential ethical implications of releasing this benchmark, and believe that it will be a net positive for the community.



## **15. Towards Reliable Evaluation and Fast Training of Robust Semantic Segmentation Models**

cs.CV

ECCV 2024

**SubmitDate**: 2024-07-16    [abs](http://arxiv.org/abs/2306.12941v2) [paper-pdf](http://arxiv.org/pdf/2306.12941v2)

**Authors**: Francesco Croce, Naman D Singh, Matthias Hein

**Abstract**: Adversarial robustness has been studied extensively in image classification, especially for the $\ell_\infty$-threat model, but significantly less so for related tasks such as object detection and semantic segmentation, where attacks turn out to be a much harder optimization problem than for image classification. We propose several problem-specific novel attacks minimizing different metrics in accuracy and mIoU. The ensemble of our attacks, SEA, shows that existing attacks severely overestimate the robustness of semantic segmentation models. Surprisingly, existing attempts of adversarial training for semantic segmentation models turn out to be weak or even completely non-robust. We investigate why previous adaptations of adversarial training to semantic segmentation failed and show how recently proposed robust ImageNet backbones can be used to obtain adversarially robust semantic segmentation models with up to six times less training time for PASCAL-VOC and the more challenging ADE20k. The associated code and robust models are available at https://github.com/nmndeep/robust-segmentation



## **16. Variational Randomized Smoothing for Sample-Wise Adversarial Robustness**

cs.LG

20 pages, under preparation

**SubmitDate**: 2024-07-16    [abs](http://arxiv.org/abs/2407.11844v1) [paper-pdf](http://arxiv.org/pdf/2407.11844v1)

**Authors**: Ryo Hase, Ye Wang, Toshiaki Koike-Akino, Jing Liu, Kieran Parsons

**Abstract**: Randomized smoothing is a defensive technique to achieve enhanced robustness against adversarial examples which are small input perturbations that degrade the performance of neural network models. Conventional randomized smoothing adds random noise with a fixed noise level for every input sample to smooth out adversarial perturbations. This paper proposes a new variational framework that uses a per-sample noise level suitable for each input by introducing a noise level selector. Our experimental results demonstrate enhancement of empirical robustness against adversarial attacks. We also provide and analyze the certified robustness for our sample-wise smoothing method.



## **17. Exploring the Robustness of Decision-Level Through Adversarial Attacks on LLM-Based Embodied Models**

cs.MM

**SubmitDate**: 2024-07-16    [abs](http://arxiv.org/abs/2405.19802v3) [paper-pdf](http://arxiv.org/pdf/2405.19802v3)

**Authors**: Shuyuan Liu, Jiawei Chen, Shouwei Ruan, Hang Su, Zhaoxia Yin

**Abstract**: Embodied intelligence empowers agents with a profound sense of perception, enabling them to respond in a manner closely aligned with real-world situations. Large Language Models (LLMs) delve into language instructions with depth, serving a crucial role in generating plans for intricate tasks. Thus, LLM-based embodied models further enhance the agent's capacity to comprehend and process information. However, this amalgamation also ushers in new challenges in the pursuit of heightened intelligence. Specifically, attackers can manipulate LLMs to produce irrelevant or even malicious outputs by altering their prompts. Confronted with this challenge, we observe a notable absence of multi-modal datasets essential for comprehensively evaluating the robustness of LLM-based embodied models. Consequently, we construct the Embodied Intelligent Robot Attack Dataset (EIRAD), tailored specifically for robustness evaluation. Additionally, two attack strategies are devised, including untargeted attacks and targeted attacks, to effectively simulate a range of diverse attack scenarios. At the same time, during the attack process, to more accurately ascertain whether our method is successful in attacking the LLM-based embodied model, we devise a new attack success evaluation method utilizing the BLIP2 model. Recognizing the time and cost-intensive nature of the GCG algorithm in attacks, we devise a scheme for prompt suffix initialization based on various target tasks, thus expediting the convergence process. Experimental results demonstrate that our method exhibits a superior attack success rate when targeting LLM-based embodied models, indicating a lower level of decision-level robustness in these models.



## **18. Relaxing Graph Transformers for Adversarial Attacks**

cs.LG

**SubmitDate**: 2024-07-16    [abs](http://arxiv.org/abs/2407.11764v1) [paper-pdf](http://arxiv.org/pdf/2407.11764v1)

**Authors**: Philipp Foth, Lukas Gosch, Simon Geisler, Leo Schwinn, Stephan Günnemann

**Abstract**: Existing studies have shown that Graph Neural Networks (GNNs) are vulnerable to adversarial attacks. Even though Graph Transformers (GTs) surpassed Message-Passing GNNs on several benchmarks, their adversarial robustness properties are unexplored. However, attacking GTs is challenging due to their Positional Encodings (PEs) and special attention mechanisms which can be difficult to differentiate. We overcome these challenges by targeting three representative architectures based on (1) random-walk PEs, (2) pair-wise-shortest-path PEs, and (3) spectral PEs - and propose the first adaptive attacks for GTs. We leverage our attacks to evaluate robustness to (a) structure perturbations on node classification; and (b) node injection attacks for (fake-news) graph classification. Our evaluation reveals that they can be catastrophically fragile and underlines our work's importance and the necessity for adaptive attacks.



## **19. Enhancing TinyML Security: Study of Adversarial Attack Transferability**

cs.CR

Accepted and presented at tinyML Foundation EMEA Innovation Forum  2024

**SubmitDate**: 2024-07-16    [abs](http://arxiv.org/abs/2407.11599v1) [paper-pdf](http://arxiv.org/pdf/2407.11599v1)

**Authors**: Parin Shah, Yuvaraj Govindarajulu, Pavan Kulkarni, Manojkumar Parmar

**Abstract**: The recent strides in artificial intelligence (AI) and machine learning (ML) have propelled the rise of TinyML, a paradigm enabling AI computations at the edge without dependence on cloud connections. While TinyML offers real-time data analysis and swift responses critical for diverse applications, its devices' intrinsic resource limitations expose them to security risks. This research delves into the adversarial vulnerabilities of AI models on resource-constrained embedded hardware, with a focus on Model Extraction and Evasion Attacks. Our findings reveal that adversarial attacks from powerful host machines could be transferred to smaller, less secure devices like ESP32 and Raspberry Pi. This illustrates that adversarial attacks could be extended to tiny devices, underscoring vulnerabilities, and emphasizing the necessity for reinforced security measures in TinyML deployments. This exploration enhances the comprehension of security challenges in TinyML and offers insights for safeguarding sensitive data and ensuring device dependability in AI-powered edge computing settings.



## **20. AEMIM: Adversarial Examples Meet Masked Image Modeling**

cs.CV

Under review of International Journal of Computer Vision (IJCV)

**SubmitDate**: 2024-07-16    [abs](http://arxiv.org/abs/2407.11537v1) [paper-pdf](http://arxiv.org/pdf/2407.11537v1)

**Authors**: Wenzhao Xiang, Chang Liu, Hang Su, Hongyang Yu

**Abstract**: Masked image modeling (MIM) has gained significant traction for its remarkable prowess in representation learning. As an alternative to the traditional approach, the reconstruction from corrupted images has recently emerged as a promising pretext task. However, the regular corrupted images are generated using generic generators, often lacking relevance to the specific reconstruction task involved in pre-training. Hence, reconstruction from regular corrupted images cannot ensure the difficulty of the pretext task, potentially leading to a performance decline. Moreover, generating corrupted images might introduce an extra generator, resulting in a notable computational burden. To address these issues, we propose to incorporate adversarial examples into masked image modeling, as the new reconstruction targets. Adversarial examples, generated online using only the trained models, can directly aim to disrupt tasks associated with pre-training. Therefore, the incorporation not only elevates the level of challenge in reconstruction but also enhances efficiency, contributing to the acquisition of superior representations by the model. In particular, we introduce a novel auxiliary pretext task that reconstructs the adversarial examples corresponding to the original images. We also devise an innovative adversarial attack to craft more suitable adversarial examples for MIM pre-training. It is noted that our method is not restricted to specific model architectures and MIM strategies, rendering it an adaptable plug-in capable of enhancing all MIM methods. Experimental findings substantiate the remarkable capability of our approach in amplifying the generalization and robustness of existing MIM methods. Notably, our method surpasses the performance of baselines on various tasks, including ImageNet, its variants, and other downstream tasks.



## **21. Learning on Graphs with Large Language Models(LLMs): A Deep Dive into Model Robustness**

cs.LG

**SubmitDate**: 2024-07-16    [abs](http://arxiv.org/abs/2407.12068v1) [paper-pdf](http://arxiv.org/pdf/2407.12068v1)

**Authors**: Kai Guo, Zewen Liu, Zhikai Chen, Hongzhi Wen, Wei Jin, Jiliang Tang, Yi Chang

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable performance across various natural language processing tasks. Recently, several LLMs-based pipelines have been developed to enhance learning on graphs with text attributes, showcasing promising performance. However, graphs are well-known to be susceptible to adversarial attacks and it remains unclear whether LLMs exhibit robustness in learning on graphs. To address this gap, our work aims to explore the potential of LLMs in the context of adversarial attacks on graphs. Specifically, we investigate the robustness against graph structural and textual perturbations in terms of two dimensions: LLMs-as-Enhancers and LLMs-as-Predictors. Through extensive experiments, we find that, compared to shallow models, both LLMs-as-Enhancers and LLMs-as-Predictors offer superior robustness against structural and textual attacks.Based on these findings, we carried out additional analyses to investigate the underlying causes. Furthermore, we have made our benchmark library openly available to facilitate quick and fair evaluations, and to encourage ongoing innovative research in this field.



## **22. Boosting the Transferability of Adversarial Attacks with Global Momentum Initialization**

cs.CV

Accepted by Expert Systems with Applications (ESWA)

**SubmitDate**: 2024-07-16    [abs](http://arxiv.org/abs/2211.11236v3) [paper-pdf](http://arxiv.org/pdf/2211.11236v3)

**Authors**: Jiafeng Wang, Zhaoyu Chen, Kaixun Jiang, Dingkang Yang, Lingyi Hong, Pinxue Guo, Haijing Guo, Wenqiang Zhang

**Abstract**: Deep Neural Networks (DNNs) are vulnerable to adversarial examples, which are crafted by adding human-imperceptible perturbations to the benign inputs. Simultaneously, adversarial examples exhibit transferability across models, enabling practical black-box attacks. However, existing methods are still incapable of achieving the desired transfer attack performance. In this work, focusing on gradient optimization and consistency, we analyse the gradient elimination phenomenon as well as the local momentum optimum dilemma. To tackle these challenges, we introduce Global Momentum Initialization (GI), providing global momentum knowledge to mitigate gradient elimination. Specifically, we perform gradient pre-convergence before the attack and a global search during this stage. GI seamlessly integrates with existing transfer methods, significantly improving the success rate of transfer attacks by an average of 6.4% under various advanced defense mechanisms compared to the state-of-the-art method. Ultimately, GI demonstrates strong transferability in both image and video attack domains. Particularly, when attacking advanced defense methods in the image domain, it achieves an average attack success rate of 95.4%. The code is available at $\href{https://github.com/Omenzychen/Global-Momentum-Initialization}{https://github.com/Omenzychen/Global-Momentum-Initialization}$.



## **23. Investigating Imperceptibility of Adversarial Attacks on Tabular Data: An Empirical Analysis**

cs.LG

33 pages

**SubmitDate**: 2024-07-16    [abs](http://arxiv.org/abs/2407.11463v1) [paper-pdf](http://arxiv.org/pdf/2407.11463v1)

**Authors**: Zhipeng He, Chun Ouyang, Laith Alzubaidi, Alistair Barros, Catarina Moreira

**Abstract**: Adversarial attacks are a potential threat to machine learning models, as they can cause the model to make incorrect predictions by introducing imperceptible perturbations to the input data. While extensively studied in unstructured data like images, their application to structured data like tabular data presents unique challenges due to the heterogeneity and intricate feature interdependencies of tabular data. Imperceptibility in tabular data involves preserving data integrity while potentially causing misclassification, underscoring the need for tailored imperceptibility criteria for tabular data. However, there is currently a lack of standardised metrics for assessing adversarial attacks specifically targeted at tabular data. To address this gap, we derive a set of properties for evaluating the imperceptibility of adversarial attacks on tabular data. These properties are defined to capture seven perspectives of perturbed data: proximity to original inputs, sparsity of alterations, deviation to datapoints in the original dataset, sensitivity of altering sensitive features, immutability of perturbation, feasibility of perturbed values and intricate feature interdepencies among tabular features. Furthermore, we conduct both quantitative empirical evaluation and case-based qualitative examples analysis for seven properties. The evaluation reveals a trade-off between attack success and imperceptibility, particularly concerning proximity, sensitivity, and deviation. Although no evaluated attacks can achieve optimal effectiveness and imperceptibility simultaneously, unbounded attacks prove to be more promised for tabular data in crafting imperceptible adversarial examples. The study also highlights the limitation of evaluated algorithms in controlling sparsity effectively. We suggest incorporating a sparsity metric in future attack design to regulate the number of perturbed features.



## **24. PromptRobust: Towards Evaluating the Robustness of Large Language Models on Adversarial Prompts**

cs.CL

Technical report; code is at:  https://github.com/microsoft/promptbench

**SubmitDate**: 2024-07-16    [abs](http://arxiv.org/abs/2306.04528v5) [paper-pdf](http://arxiv.org/pdf/2306.04528v5)

**Authors**: Kaijie Zhu, Jindong Wang, Jiaheng Zhou, Zichen Wang, Hao Chen, Yidong Wang, Linyi Yang, Wei Ye, Yue Zhang, Neil Zhenqiang Gong, Xing Xie

**Abstract**: The increasing reliance on Large Language Models (LLMs) across academia and industry necessitates a comprehensive understanding of their robustness to prompts. In response to this vital need, we introduce PromptRobust, a robustness benchmark designed to measure LLMs' resilience to adversarial prompts. This study uses a plethora of adversarial textual attacks targeting prompts across multiple levels: character, word, sentence, and semantic. The adversarial prompts, crafted to mimic plausible user errors like typos or synonyms, aim to evaluate how slight deviations can affect LLM outcomes while maintaining semantic integrity. These prompts are then employed in diverse tasks including sentiment analysis, natural language inference, reading comprehension, machine translation, and math problem-solving. Our study generates 4,788 adversarial prompts, meticulously evaluated over 8 tasks and 13 datasets. Our findings demonstrate that contemporary LLMs are not robust to adversarial prompts. Furthermore, we present a comprehensive analysis to understand the mystery behind prompt robustness and its transferability. We then offer insightful robustness analysis and pragmatic recommendations for prompt composition, beneficial to both researchers and everyday users.



## **25. Gradients Look Alike: Sensitivity is Often Overestimated in DP-SGD**

cs.LG

published in 33rd USENIX Security Symposium

**SubmitDate**: 2024-07-16    [abs](http://arxiv.org/abs/2307.00310v3) [paper-pdf](http://arxiv.org/pdf/2307.00310v3)

**Authors**: Anvith Thudi, Hengrui Jia, Casey Meehan, Ilia Shumailov, Nicolas Papernot

**Abstract**: Differentially private stochastic gradient descent (DP-SGD) is the canonical approach to private deep learning. While the current privacy analysis of DP-SGD is known to be tight in some settings, several empirical results suggest that models trained on common benchmark datasets leak significantly less privacy for many datapoints. Yet, despite past attempts, a rigorous explanation for why this is the case has not been reached. Is it because there exist tighter privacy upper bounds when restricted to these dataset settings, or are our attacks not strong enough for certain datapoints? In this paper, we provide the first per-instance (i.e., ``data-dependent") DP analysis of DP-SGD. Our analysis captures the intuition that points with similar neighbors in the dataset enjoy better data-dependent privacy than outliers. Formally, this is done by modifying the per-step privacy analysis of DP-SGD to introduce a dependence on the distribution of model updates computed from a training dataset. We further develop a new composition theorem to effectively use this new per-step analysis to reason about an entire training run. Put all together, our evaluation shows that this novel DP-SGD analysis allows us to now formally show that DP-SGD leaks significantly less privacy for many datapoints (when trained on common benchmarks) than the current data-independent guarantee. This implies privacy attacks will necessarily fail against many datapoints if the adversary does not have sufficient control over the possible training datasets.



## **26. Learning to Break Deep Perceptual Hashing: The Use Case NeuralHash**

cs.LG

Accepted by ACM FAccT 2022 as Oral

**SubmitDate**: 2024-07-16    [abs](http://arxiv.org/abs/2111.06628v5) [paper-pdf](http://arxiv.org/pdf/2111.06628v5)

**Authors**: Lukas Struppek, Dominik Hintersdorf, Daniel Neider, Kristian Kersting

**Abstract**: Apple recently revealed its deep perceptual hashing system NeuralHash to detect child sexual abuse material (CSAM) on user devices before files are uploaded to its iCloud service. Public criticism quickly arose regarding the protection of user privacy and the system's reliability. In this paper, we present the first comprehensive empirical analysis of deep perceptual hashing based on NeuralHash. Specifically, we show that current deep perceptual hashing may not be robust. An adversary can manipulate the hash values by applying slight changes in images, either induced by gradient-based approaches or simply by performing standard image transformations, forcing or preventing hash collisions. Such attacks permit malicious actors easily to exploit the detection system: from hiding abusive material to framing innocent users, everything is possible. Moreover, using the hash values, inferences can still be made about the data stored on user devices. In our view, based on our results, deep perceptual hashing in its current form is generally not ready for robust client-side scanning and should not be used from a privacy perspective.



## **27. Wicked Oddities: Selectively Poisoning for Effective Clean-Label Backdoor Attacks**

cs.LG

**SubmitDate**: 2024-07-16    [abs](http://arxiv.org/abs/2407.10825v2) [paper-pdf](http://arxiv.org/pdf/2407.10825v2)

**Authors**: Quang H. Nguyen, Nguyen Ngoc-Hieu, The-Anh Ta, Thanh Nguyen-Tang, Kok-Seng Wong, Hoang Thanh-Tung, Khoa D. Doan

**Abstract**: Deep neural networks are vulnerable to backdoor attacks, a type of adversarial attack that poisons the training data to manipulate the behavior of models trained on such data. Clean-label attacks are a more stealthy form of backdoor attacks that can perform the attack without changing the labels of poisoned data. Early works on clean-label attacks added triggers to a random subset of the training set, ignoring the fact that samples contribute unequally to the attack's success. This results in high poisoning rates and low attack success rates. To alleviate the problem, several supervised learning-based sample selection strategies have been proposed. However, these methods assume access to the entire labeled training set and require training, which is expensive and may not always be practical. This work studies a new and more practical (but also more challenging) threat model where the attacker only provides data for the target class (e.g., in face recognition systems) and has no knowledge of the victim model or any other classes in the training set. We study different strategies for selectively poisoning a small set of training samples in the target class to boost the attack success rate in this setting. Our threat model poses a serious threat in training machine learning models with third-party datasets, since the attack can be performed effectively with limited information. Experiments on benchmark datasets illustrate the effectiveness of our strategies in improving clean-label backdoor attacks.



## **28. Feature Inference Attack on Shapley Values**

cs.LG

This work was published in CCS 2022

**SubmitDate**: 2024-07-16    [abs](http://arxiv.org/abs/2407.11359v1) [paper-pdf](http://arxiv.org/pdf/2407.11359v1)

**Authors**: Xinjian Luo, Yangfan Jiang, Xiaokui Xiao

**Abstract**: As a solution concept in cooperative game theory, Shapley value is highly recognized in model interpretability studies and widely adopted by the leading Machine Learning as a Service (MLaaS) providers, such as Google, Microsoft, and IBM. However, as the Shapley value-based model interpretability methods have been thoroughly studied, few researchers consider the privacy risks incurred by Shapley values, despite that interpretability and privacy are two foundations of machine learning (ML) models.   In this paper, we investigate the privacy risks of Shapley value-based model interpretability methods using feature inference attacks: reconstructing the private model inputs based on their Shapley value explanations. Specifically, we present two adversaries. The first adversary can reconstruct the private inputs by training an attack model based on an auxiliary dataset and black-box access to the model interpretability services. The second adversary, even without any background knowledge, can successfully reconstruct most of the private features by exploiting the local linear correlations between the model inputs and outputs. We perform the proposed attacks on the leading MLaaS platforms, i.e., Google Cloud, Microsoft Azure, and IBM aix360. The experimental results demonstrate the vulnerability of the state-of-the-art Shapley value-based model interpretability methods used in the leading MLaaS platforms and highlight the significance and necessity of designing privacy-preserving model interpretability methods in future studies. To our best knowledge, this is also the first work that investigates the privacy risks of Shapley values.



## **29. Towards Adversarially Robust Vision-Language Models: Insights from Design Choices and Prompt Formatting Techniques**

cs.CV

**SubmitDate**: 2024-07-15    [abs](http://arxiv.org/abs/2407.11121v1) [paper-pdf](http://arxiv.org/pdf/2407.11121v1)

**Authors**: Rishika Bhagwatkar, Shravan Nayak, Reza Bayat, Alexis Roger, Daniel Z Kaplan, Pouya Bashivan, Irina Rish

**Abstract**: Vision-Language Models (VLMs) have witnessed a surge in both research and real-world applications. However, as they are becoming increasingly prevalent, ensuring their robustness against adversarial attacks is paramount. This work systematically investigates the impact of model design choices on the adversarial robustness of VLMs against image-based attacks. Additionally, we introduce novel, cost-effective approaches to enhance robustness through prompt formatting. By rephrasing questions and suggesting potential adversarial perturbations, we demonstrate substantial improvements in model robustness against strong image-based attacks such as Auto-PGD. Our findings provide important guidelines for developing more robust VLMs, particularly for deployment in safety-critical environments.



## **30. PartImageNet++ Dataset: Scaling up Part-based Models for Robust Recognition**

cs.CV

Accepted by ECCV2024

**SubmitDate**: 2024-07-15    [abs](http://arxiv.org/abs/2407.10918v1) [paper-pdf](http://arxiv.org/pdf/2407.10918v1)

**Authors**: Xiao Li, Yining Liu, Na Dong, Sitian Qin, Xiaolin Hu

**Abstract**: Deep learning-based object recognition systems can be easily fooled by various adversarial perturbations. One reason for the weak robustness may be that they do not have part-based inductive bias like the human recognition process. Motivated by this, several part-based recognition models have been proposed to improve the adversarial robustness of recognition. However, due to the lack of part annotations, the effectiveness of these methods is only validated on small-scale nonstandard datasets. In this work, we propose PIN++, short for PartImageNet++, a dataset providing high-quality part segmentation annotations for all categories of ImageNet-1K (IN-1K). With these annotations, we build part-based methods directly on the standard IN-1K dataset for robust recognition. Different from previous two-stage part-based models, we propose a Multi-scale Part-supervised Model (MPM), to learn a robust representation with part annotations. Experiments show that MPM yielded better adversarial robustness on the large-scale IN-1K over strong baselines across various attack settings. Furthermore, MPM achieved improved robustness on common corruptions and several out-of-distribution datasets. The dataset, together with these results, enables and encourages researchers to explore the potential of part-based models in more real applications.



## **31. Provable Robustness of (Graph) Neural Networks Against Data Poisoning and Backdoor Attacks**

cs.LG

**SubmitDate**: 2024-07-15    [abs](http://arxiv.org/abs/2407.10867v1) [paper-pdf](http://arxiv.org/pdf/2407.10867v1)

**Authors**: Lukas Gosch, Mahalakshmi Sabanayagam, Debarghya Ghoshdastidar, Stephan Günnemann

**Abstract**: Generalization of machine learning models can be severely compromised by data poisoning, where adversarial changes are applied to the training data, as well as backdoor attacks that additionally manipulate the test data. These vulnerabilities have led to interest in certifying (i.e., proving) that such changes up to a certain magnitude do not affect test predictions. We, for the first time, certify Graph Neural Networks (GNNs) against poisoning and backdoor attacks targeting the node features of a given graph. Our certificates are white-box and based upon $(i)$ the neural tangent kernel, which characterizes the training dynamics of sufficiently wide networks; and $(ii)$ a novel reformulation of the bilevel optimization problem describing poisoning as a mixed-integer linear program. Consequently, we leverage our framework to provide fundamental insights into the role of graph structure and its connectivity on the worst-case robustness behavior of convolution-based and PageRank-based GNNs. We note that our framework is more general and constitutes the first approach to derive white-box poisoning certificates for NNs, which can be of independent interest beyond graph-related tasks.



## **32. Secure Aggregation is Not Private Against Membership Inference Attacks**

cs.LG

accepted to the European Conference on Machine Learning and  Principles and Practice of Knowledge Discovery in Databases (ECML PKDD) 2024

**SubmitDate**: 2024-07-15    [abs](http://arxiv.org/abs/2403.17775v3) [paper-pdf](http://arxiv.org/pdf/2403.17775v3)

**Authors**: Khac-Hoang Ngo, Johan Östman, Giuseppe Durisi, Alexandre Graell i Amat

**Abstract**: Secure aggregation (SecAgg) is a commonly-used privacy-enhancing mechanism in federated learning, affording the server access only to the aggregate of model updates while safeguarding the confidentiality of individual updates. Despite widespread claims regarding SecAgg's privacy-preserving capabilities, a formal analysis of its privacy is lacking, making such presumptions unjustified. In this paper, we delve into the privacy implications of SecAgg by treating it as a local differential privacy (LDP) mechanism for each local update. We design a simple attack wherein an adversarial server seeks to discern which update vector a client submitted, out of two possible ones, in a single training round of federated learning under SecAgg. By conducting privacy auditing, we assess the success probability of this attack and quantify the LDP guarantees provided by SecAgg. Our numerical results unveil that, contrary to prevailing claims, SecAgg offers weak privacy against membership inference attacks even in a single training round. Indeed, it is difficult to hide a local update by adding other independent local updates when the updates are of high dimension. Our findings underscore the imperative for additional privacy-enhancing mechanisms, such as noise injection, in federated learning.



## **33. The Quantum Imitation Game: Reverse Engineering of Quantum Machine Learning Models**

quant-ph

11 pages, 12 figures

**SubmitDate**: 2024-07-15    [abs](http://arxiv.org/abs/2407.07237v2) [paper-pdf](http://arxiv.org/pdf/2407.07237v2)

**Authors**: Archisman Ghosh, Swaroop Ghosh

**Abstract**: Quantum Machine Learning (QML) amalgamates quantum computing paradigms with machine learning models, providing significant prospects for solving complex problems. However, with the expansion of numerous third-party vendors in the Noisy Intermediate-Scale Quantum (NISQ) era of quantum computing, the security of QML models is of prime importance, particularly against reverse engineering, which could expose trained parameters and algorithms of the models. We assume the untrusted quantum cloud provider is an adversary having white-box access to the transpiled user-designed trained QML model during inference. Reverse engineering (RE) to extract the pre-transpiled QML circuit will enable re-transpilation and usage of the model for various hardware with completely different native gate sets and even different qubit technology. Such flexibility may not be obtained from the transpiled circuit which is tied to a particular hardware and qubit technology. The information about the number of parameters, and optimized values can allow further training of the QML model to alter the QML model, tamper with the watermark, and/or embed their own watermark or refine the model for other purposes. In this first effort to investigate the RE of QML circuits, we perform RE and compare the training accuracy of original and reverse-engineered Quantum Neural Networks (QNNs) of various sizes. We note that multi-qubit classifiers can be reverse-engineered under specific conditions with a mean error of order 1e-2 in a reasonable time. We also propose adding dummy fixed parametric gates in the QML models to increase the RE overhead for defense. For instance, adding 2 dummy qubits and 2 layers increases the overhead by ~1.76 times for a classifier with 2 qubits and 3 layers with a performance overhead of less than 9%. We note that RE is a very powerful attack model which warrants further efforts on defenses.



## **34. Formal Verification of Object Detection**

cs.CV

**SubmitDate**: 2024-07-15    [abs](http://arxiv.org/abs/2407.01295v4) [paper-pdf](http://arxiv.org/pdf/2407.01295v4)

**Authors**: Avraham Raviv, Yizhak Y. Elboher, Michelle Aluf-Medina, Yael Leibovich Weiss, Omer Cohen, Roy Assa, Guy Katz, Hillel Kugler

**Abstract**: Deep Neural Networks (DNNs) are ubiquitous in real-world applications, yet they remain vulnerable to errors and adversarial attacks. This work tackles the challenge of applying formal verification to ensure the safety of computer vision models, extending verification beyond image classification to object detection. We propose a general formulation for certifying the robustness of object detection models using formal verification and outline implementation strategies compatible with state-of-the-art verification tools. Our approach enables the application of these tools, originally designed for verifying classification models, to object detection. We define various attacks for object detection, illustrating the diverse ways adversarial inputs can compromise neural network outputs. Our experiments, conducted on several common datasets and networks, reveal potential errors in object detection models, highlighting system vulnerabilities and emphasizing the need for expanding formal verification to these new domains. This work paves the way for further research in integrating formal verification across a broader range of computer vision applications.



## **35. TAPI: Towards Target-Specific and Adversarial Prompt Injection against Code LLMs**

cs.CR

**SubmitDate**: 2024-07-15    [abs](http://arxiv.org/abs/2407.09164v2) [paper-pdf](http://arxiv.org/pdf/2407.09164v2)

**Authors**: Yuchen Yang, Hongwei Yao, Bingrun Yang, Yiling He, Yiming Li, Tianwei Zhang, Zhan Qin, Kui Ren

**Abstract**: Recently, code-oriented large language models (Code LLMs) have been widely and successfully used to simplify and facilitate code programming. With these tools, developers can easily generate desired complete functional codes based on incomplete code and natural language prompts. However, a few pioneering works revealed that these Code LLMs are also vulnerable, e.g., against backdoor and adversarial attacks. The former could induce LLMs to respond to triggers to insert malicious code snippets by poisoning the training data or model parameters, while the latter can craft malicious adversarial input codes to reduce the quality of generated codes. However, both attack methods have underlying limitations: backdoor attacks rely on controlling the model training process, while adversarial attacks struggle with fulfilling specific malicious purposes.   To inherit the advantages of both backdoor and adversarial attacks, this paper proposes a new attack paradigm, i.e., target-specific and adversarial prompt injection (TAPI), against Code LLMs. TAPI generates unreadable comments containing information about malicious instructions and hides them as triggers in the external source code. When users exploit Code LLMs to complete codes containing the trigger, the models will generate attacker-specified malicious code snippets at specific locations. We evaluate our TAPI attack on four representative LLMs under three representative malicious objectives and seven cases. The results show that our method is highly threatening (achieving an attack success rate enhancement of up to 89.3%) and stealthy (saving an average of 53.1% of tokens in the trigger design). In particular, we successfully attack some famous deployed code completion integrated applications, including CodeGeex and Github Copilot. This further confirms the realistic threat of our attack.



## **36. Self-Evaluation as a Defense Against Adversarial Attacks on LLMs**

cs.LG

8 pages, 7 figures

**SubmitDate**: 2024-07-15    [abs](http://arxiv.org/abs/2407.03234v2) [paper-pdf](http://arxiv.org/pdf/2407.03234v2)

**Authors**: Hannah Brown, Leon Lin, Kenji Kawaguchi, Michael Shieh

**Abstract**: When LLMs are deployed in sensitive, human-facing settings, it is crucial that they do not output unsafe, biased, or privacy-violating outputs. For this reason, models are both trained and instructed to refuse to answer unsafe prompts such as "Tell me how to build a bomb." We find that, despite these safeguards, it is possible to break model defenses simply by appending a space to the end of a model's input. In a study of eight open-source models, we demonstrate that this acts as a strong enough attack to cause the majority of models to generate harmful outputs with very high success rates. We examine the causes of this behavior, finding that the contexts in which single spaces occur in tokenized training data encourage models to generate lists when prompted, overriding training signals to refuse to answer unsafe requests. Our findings underscore the fragile state of current model alignment and promote the importance of developing more robust alignment methods. Code and data will be made available at https://github.com/Linlt-leon/self-eval.



## **37. Backdoor Attacks against Image-to-Image Networks**

cs.CV

**SubmitDate**: 2024-07-15    [abs](http://arxiv.org/abs/2407.10445v1) [paper-pdf](http://arxiv.org/pdf/2407.10445v1)

**Authors**: Wenbo Jiang, Hongwei Li, Jiaming He, Rui Zhang, Guowen Xu, Tianwei Zhang, Rongxing Lu

**Abstract**: Recently, deep learning-based Image-to-Image (I2I) networks have become the predominant choice for I2I tasks such as image super-resolution and denoising. Despite their remarkable performance, the backdoor vulnerability of I2I networks has not been explored. To fill this research gap, we conduct a comprehensive investigation on the susceptibility of I2I networks to backdoor attacks. Specifically, we propose a novel backdoor attack technique, where the compromised I2I network behaves normally on clean input images, yet outputs a predefined image of the adversary for malicious input images containing the trigger. To achieve this I2I backdoor attack, we propose a targeted universal adversarial perturbation (UAP) generation algorithm for I2I networks, where the generated UAP is used as the backdoor trigger. Additionally, in the backdoor training process that contains the main task and the backdoor task, multi-task learning (MTL) with dynamic weighting methods is employed to accelerate convergence rates. In addition to attacking I2I tasks, we extend our I2I backdoor to attack downstream tasks, including image classification and object detection. Extensive experiments demonstrate the effectiveness of the I2I backdoor on state-of-the-art I2I network architectures, as well as the robustness against different mainstream backdoor defenses.



## **38. Pandora's White-Box: Precise Training Data Detection and Extraction in Large Language Models**

cs.CR

Found software bug in experiments, withdrawing in order to address  and update results

**SubmitDate**: 2024-07-15    [abs](http://arxiv.org/abs/2402.17012v4) [paper-pdf](http://arxiv.org/pdf/2402.17012v4)

**Authors**: Jeffrey G. Wang, Jason Wang, Marvin Li, Seth Neel

**Abstract**: In this paper we develop state-of-the-art privacy attacks against Large Language Models (LLMs), where an adversary with some access to the model tries to learn something about the underlying training data. Our headline results are new membership inference attacks (MIAs) against pretrained LLMs that perform hundreds of times better than baseline attacks, and a pipeline showing that over 50% (!) of the fine-tuning dataset can be extracted from a fine-tuned LLM in natural settings. We consider varying degrees of access to the underlying model, pretraining and fine-tuning data, and both MIAs and training data extraction. For pretraining data, we propose two new MIAs: a supervised neural network classifier that predicts training data membership on the basis of (dimensionality-reduced) model gradients, as well as a variant of this attack that only requires logit access to the model by leveraging recent model-stealing work on LLMs. To our knowledge this is the first MIA that explicitly incorporates model-stealing information. Both attacks outperform existing black-box baselines, and our supervised attack closes the gap between MIA attack success against LLMs and the strongest known attacks for other machine learning models. In fine-tuning, we find that a simple attack based on the ratio of the loss between the base and fine-tuned models is able to achieve near-perfect MIA performance; we then leverage our MIA to extract a large fraction of the fine-tuning dataset from fine-tuned Pythia and Llama models. Our code is available at github.com/safr-ai-lab/pandora-llm.



## **39. MultiDelete for Multimodal Machine Unlearning**

cs.AI

ECCV 2024

**SubmitDate**: 2024-07-15    [abs](http://arxiv.org/abs/2311.12047v2) [paper-pdf](http://arxiv.org/pdf/2311.12047v2)

**Authors**: Jiali Cheng, Hadi Amiri

**Abstract**: Machine Unlearning removes specific knowledge about training data samples from an already trained model. It has significant practical benefits, such as purging private, inaccurate, or outdated information from trained models without the need for complete re-training. Unlearning within a multimodal setting presents unique challenges due to the complex dependencies between different data modalities and the expensive cost of training on large multimodal datasets and architectures. This paper presents the first machine unlearning approach for multimodal data and models, titled MultiDelete, which is designed to decouple associations between unimodal data points during unlearning without losing the overall representation strength of the trained model. MultiDelete advocates for three key properties for effective multimodal unlearning: (a): modality decoupling, which effectively decouples the association between individual unimodal data points marked for deletion, rendering them as unrelated data points, (b): multimodal knowledge retention, which retains the multimodal representation post-unlearning, and (c): unimodal knowledge retention, which retains the unimodal representation postunlearning. MultiDelete is efficient to train and is not constrained by using a strongly convex loss -- a common restriction among existing baselines. Experiments on two architectures and four datasets, including image-text and graph-text datasets, show that MultiDelete gains an average improvement of 17.6 points over best performing baseline in unlearning multimodal samples, can maintain the multimodal and unimodal knowledge of the original model post unlearning, and can provide better protection to unlearned data against adversarial attacks.



## **40. SENTINEL: Securing Indoor Localization against Adversarial Attacks with Capsule Neural Networks**

eess.SP

**SubmitDate**: 2024-07-14    [abs](http://arxiv.org/abs/2407.11091v1) [paper-pdf](http://arxiv.org/pdf/2407.11091v1)

**Authors**: Danish Gufran, Pooja Anandathirtha, Sudeep Pasricha

**Abstract**: With the increasing demand for edge device powered location-based services in indoor environments, Wi-Fi received signal strength (RSS) fingerprinting has become popular, given the unavailability of GPS indoors. However, achieving robust and efficient indoor localization faces several challenges, due to RSS fluctuations from dynamic changes in indoor environments and heterogeneity of edge devices, leading to diminished localization accuracy. While advances in machine learning (ML) have shown promise in mitigating these phenomena, it remains an open problem. Additionally, emerging threats from adversarial attacks on ML-enhanced indoor localization systems, especially those introduced by malicious or rogue access points (APs), can deceive ML models to further increase localization errors. To address these challenges, we present SENTINEL, a novel embedded ML framework utilizing modified capsule neural networks to bolster the resilience of indoor localization solutions against adversarial attacks, device heterogeneity, and dynamic RSS fluctuations. We also introduce RSSRogueLoc, a novel dataset capturing the effects of rogue APs from several real-world indoor environments. Experimental evaluations demonstrate that SENTINEL achieves significant improvements, with up to 3.5x reduction in mean error and 3.4x reduction in worst-case error compared to state-of-the-art frameworks using simulated adversarial attacks. SENTINEL also achieves improvements of up to 2.8x in mean error and 2.7x in worst-case error compared to state-of-the-art frameworks when evaluated with the real-world RSSRogueLoc dataset.



## **41. Proof-of-Learning with Incentive Security**

cs.CR

17 pages

**SubmitDate**: 2024-07-14    [abs](http://arxiv.org/abs/2404.09005v6) [paper-pdf](http://arxiv.org/pdf/2404.09005v6)

**Authors**: Zishuo Zhao, Zhixuan Fang, Xuechao Wang, Xi Chen, Yuan Zhou

**Abstract**: Most concurrent blockchain systems rely heavily on the Proof-of-Work (PoW) or Proof-of-Stake (PoS) mechanisms for decentralized consensus and security assurance. However, the substantial energy expenditure stemming from computationally intensive yet meaningless tasks has raised considerable concerns surrounding traditional PoW approaches, The PoS mechanism, while free of energy consumption, is subject to security and economic issues. Addressing these issues, the paradigm of Proof-of-Useful-Work (PoUW) seeks to employ challenges of practical significance as PoW, thereby imbuing energy consumption with tangible value. While previous efforts in Proof of Learning (PoL) explored the utilization of deep learning model training SGD tasks as PoUW challenges, recent research has revealed its vulnerabilities to adversarial attacks and the theoretical hardness in crafting a byzantine-secure PoL mechanism. In this paper, we introduce the concept of incentive-security that incentivizes rational provers to behave honestly for their best interest, bypassing the existing hardness to design a PoL mechanism with computational efficiency, a provable incentive-security guarantee and controllable difficulty. Particularly, our work is secure against two attacks to the recent work of Jia et al. [2021], and also improves the computational overhead from $\Theta(1)$ to $O(\frac{\log E}{E})$. Furthermore, while most recent research assumes trusted problem providers and verifiers, our design also guarantees frontend incentive-security even when problem providers are untrusted, and verifier incentive-security that bypasses the Verifier's Dilemma. By incorporating ML training into blockchain consensus mechanisms with provable guarantees, our research not only proposes an eco-friendly solution to blockchain systems, but also provides a proposal for a completely decentralized computing power market in the new AI age.



## **42. Merging Improves Self-Critique Against Jailbreak Attacks**

cs.CL

Published at ICML 2024 Workshop on Foundation Models in the Wild

**SubmitDate**: 2024-07-14    [abs](http://arxiv.org/abs/2406.07188v2) [paper-pdf](http://arxiv.org/pdf/2406.07188v2)

**Authors**: Victor Gallego

**Abstract**: The robustness of large language models (LLMs) against adversarial manipulations, such as jailbreak attacks, remains a significant challenge. In this work, we propose an approach that enhances the self-critique capability of the LLM and further fine-tunes it over sanitized synthetic data. This is done with the addition of an external critic model that can be merged with the original, thus bolstering self-critique capabilities and improving the robustness of the LLMs response to adversarial prompts. Our results demonstrate that the combination of merging and self-critique can reduce the attack success rate of adversaries significantly, thus offering a promising defense mechanism against jailbreak attacks. Code, data and models released at https://github.com/vicgalle/merging-self-critique-jailbreaks .



## **43. Boosting Transferability in Vision-Language Attacks via Diversification along the Intersection Region of Adversarial Trajectory**

cs.CV

ECCV2024. Code is available at  https://github.com/SensenGao/VLPTransferAttack

**SubmitDate**: 2024-07-14    [abs](http://arxiv.org/abs/2403.12445v3) [paper-pdf](http://arxiv.org/pdf/2403.12445v3)

**Authors**: Sensen Gao, Xiaojun Jia, Xuhong Ren, Ivor Tsang, Qing Guo

**Abstract**: Vision-language pre-training (VLP) models exhibit remarkable capabilities in comprehending both images and text, yet they remain susceptible to multimodal adversarial examples (AEs). Strengthening attacks and uncovering vulnerabilities, especially common issues in VLP models (e.g., high transferable AEs), can advance reliable and practical VLP models. A recent work (i.e., Set-level guidance attack) indicates that augmenting image-text pairs to increase AE diversity along the optimization path enhances the transferability of adversarial examples significantly. However, this approach predominantly emphasizes diversity around the online adversarial examples (i.e., AEs in the optimization period), leading to the risk of overfitting the victim model and affecting the transferability. In this study, we posit that the diversity of adversarial examples towards the clean input and online AEs are both pivotal for enhancing transferability across VLP models. Consequently, we propose using diversification along the intersection region of adversarial trajectory to expand the diversity of AEs. To fully leverage the interaction between modalities, we introduce text-guided adversarial example selection during optimization. Furthermore, to further mitigate the potential overfitting, we direct the adversarial text deviating from the last intersection region along the optimization path, rather than adversarial images as in existing methods. Extensive experiments affirm the effectiveness of our method in improving transferability across various VLP models and downstream vision-and-language tasks.



## **44. CLIP-Guided Networks for Transferable Targeted Attacks**

cs.CV

ECCV 2024

**SubmitDate**: 2024-07-14    [abs](http://arxiv.org/abs/2407.10179v1) [paper-pdf](http://arxiv.org/pdf/2407.10179v1)

**Authors**: Hao Fang, Jiawei Kong, Bin Chen, Tao Dai, Hao Wu, Shu-Tao Xia

**Abstract**: Transferable targeted adversarial attacks aim to mislead models into outputting adversary-specified predictions in black-box scenarios. Recent studies have introduced \textit{single-target} generative attacks that train a generator for each target class to generate highly transferable perturbations, resulting in substantial computational overhead when handling multiple classes. \textit{Multi-target} attacks address this by training only one class-conditional generator for multiple classes. However, the generator simply uses class labels as conditions, failing to leverage the rich semantic information of the target class. To this end, we design a \textbf{C}LIP-guided \textbf{G}enerative \textbf{N}etwork with \textbf{C}ross-attention modules (CGNC) to enhance multi-target attacks by incorporating textual knowledge of CLIP into the generator. Extensive experiments demonstrate that CGNC yields significant improvements over previous multi-target generative attacks, e.g., a 21.46\% improvement in success rate from ResNet-152 to DenseNet-121. Moreover, we propose a masked fine-tuning mechanism to further strengthen our method in attacking a single class, which surpasses existing single-target methods.



## **45. Can Adversarial Examples Be Parsed to Reveal Victim Model Information?**

cs.CV

**SubmitDate**: 2024-07-14    [abs](http://arxiv.org/abs/2303.07474v3) [paper-pdf](http://arxiv.org/pdf/2303.07474v3)

**Authors**: Yuguang Yao, Jiancheng Liu, Yifan Gong, Xiaoming Liu, Yanzhi Wang, Xue Lin, Sijia Liu

**Abstract**: Numerous adversarial attack methods have been developed to generate imperceptible image perturbations that can cause erroneous predictions of state-of-the-art machine learning (ML) models, in particular, deep neural networks (DNNs). Despite intense research on adversarial attacks, little effort was made to uncover 'arcana' carried in adversarial attacks. In this work, we ask whether it is possible to infer data-agnostic victim model (VM) information (i.e., characteristics of the ML model or DNN used to generate adversarial attacks) from data-specific adversarial instances. We call this 'model parsing of adversarial attacks' - a task to uncover 'arcana' in terms of the concealed VM information in attacks. We approach model parsing via supervised learning, which correctly assigns classes of VM's model attributes (in terms of architecture type, kernel size, activation function, and weight sparsity) to an attack instance generated from this VM. We collect a dataset of adversarial attacks across 7 attack types generated from 135 victim models (configured by 5 architecture types, 3 kernel size setups, 3 activation function types, and 3 weight sparsity ratios). We show that a simple, supervised model parsing network (MPN) is able to infer VM attributes from unseen adversarial attacks if their attack settings are consistent with the training setting (i.e., in-distribution generalization assessment). We also provide extensive experiments to justify the feasibility of VM parsing from adversarial attacks, and the influence of training and evaluation factors in the parsing performance (e.g., generalization challenge raised in out-of-distribution evaluation). We further demonstrate how the proposed MPN can be used to uncover the source VM attributes from transfer attacks, and shed light on a potential connection between model parsing and attack transferability.



## **46. Transferable 3D Adversarial Shape Completion using Diffusion Models**

cs.CV

ECCV 2024

**SubmitDate**: 2024-07-14    [abs](http://arxiv.org/abs/2407.10077v1) [paper-pdf](http://arxiv.org/pdf/2407.10077v1)

**Authors**: Xuelong Dai, Bin Xiao

**Abstract**: Recent studies that incorporate geometric features and transformers into 3D point cloud feature learning have significantly improved the performance of 3D deep-learning models. However, their robustness against adversarial attacks has not been thoroughly explored. Existing attack methods primarily focus on white-box scenarios and struggle to transfer to recently proposed 3D deep-learning models. Even worse, these attacks introduce perturbations to 3D coordinates, generating unrealistic adversarial examples and resulting in poor performance against 3D adversarial defenses. In this paper, we generate high-quality adversarial point clouds using diffusion models. By using partial points as prior knowledge, we generate realistic adversarial examples through shape completion with adversarial guidance. The proposed adversarial shape completion allows for a more reliable generation of adversarial point clouds. To enhance attack transferability, we delve into the characteristics of 3D point clouds and employ model uncertainty for better inference of model classification through random down-sampling of point clouds. We adopt ensemble adversarial guidance for improved transferability across different network architectures. To maintain the generation quality, we limit our adversarial guidance solely to the critical points of the point clouds by calculating saliency scores. Extensive experiments demonstrate that our proposed attacks outperform state-of-the-art adversarial attack methods against both black-box models and defenses. Our black-box attack establishes a new baseline for evaluating the robustness of various 3D point cloud classification models.



## **47. AdvDiff: Generating Unrestricted Adversarial Examples using Diffusion Models**

cs.LG

ECCV 2024

**SubmitDate**: 2024-07-14    [abs](http://arxiv.org/abs/2307.12499v4) [paper-pdf](http://arxiv.org/pdf/2307.12499v4)

**Authors**: Xuelong Dai, Kaisheng Liang, Bin Xiao

**Abstract**: Unrestricted adversarial attacks present a serious threat to deep learning models and adversarial defense techniques. They pose severe security problems for deep learning applications because they can effectively bypass defense mechanisms. However, previous attack methods often directly inject Projected Gradient Descent (PGD) gradients into the sampling of generative models, which are not theoretically provable and thus generate unrealistic examples by incorporating adversarial objectives, especially for GAN-based methods on large-scale datasets like ImageNet. In this paper, we propose a new method, called AdvDiff, to generate unrestricted adversarial examples with diffusion models. We design two novel adversarial guidance techniques to conduct adversarial sampling in the reverse generation process of diffusion models. These two techniques are effective and stable in generating high-quality, realistic adversarial examples by integrating gradients of the target classifier interpretably. Experimental results on MNIST and ImageNet datasets demonstrate that AdvDiff is effective in generating unrestricted adversarial examples, which outperforms state-of-the-art unrestricted adversarial attack methods in terms of attack performance and generation quality.



## **48. Harvesting Private Medical Images in Federated Learning Systems with Crafted Models**

cs.LG

**SubmitDate**: 2024-07-13    [abs](http://arxiv.org/abs/2407.09972v1) [paper-pdf](http://arxiv.org/pdf/2407.09972v1)

**Authors**: Shanghao Shi, Md Shahedul Haque, Abhijeet Parida, Marius George Linguraru, Y. Thomas Hou, Syed Muhammad Anwar, Wenjing Lou

**Abstract**: Federated learning (FL) allows a set of clients to collaboratively train a machine-learning model without exposing local training samples. In this context, it is considered to be privacy-preserving and hence has been adopted by medical centers to train machine-learning models over private data. However, in this paper, we propose a novel attack named MediLeak that enables a malicious parameter server to recover high-fidelity patient images from the model updates uploaded by the clients. MediLeak requires the server to generate an adversarial model by adding a crafted module in front of the original model architecture. It is published to the clients in the regular FL training process and each client conducts local training on it to generate corresponding model updates. Then, based on the FL protocol, the model updates are sent back to the server and our proposed analytical method recovers private data from the parameter updates of the crafted module. We provide a comprehensive analysis for MediLeak and show that it can successfully break the state-of-the-art cryptographic secure aggregation protocols, designed to protect the FL systems from privacy inference attacks. We implement MediLeak on the MedMNIST and COVIDx CXR-4 datasets. The results show that MediLeak can nearly perfectly recover private images with high recovery rates and quantitative scores. We further perform downstream tasks such as disease classification with the recovered data, where our results show no significant performance degradation compared to using the original training samples.



## **49. Black-Box Detection of Language Model Watermarks**

cs.CR

**SubmitDate**: 2024-07-13    [abs](http://arxiv.org/abs/2405.20777v2) [paper-pdf](http://arxiv.org/pdf/2405.20777v2)

**Authors**: Thibaud Gloaguen, Nikola Jovanović, Robin Staab, Martin Vechev

**Abstract**: Watermarking has emerged as a promising way to detect LLM-generated text. To apply a watermark an LLM provider, given a secret key, augments generations with a signal that is later detectable by any party with the same key. Recent work has proposed three main families of watermarking schemes, two of which focus on the property of preserving the LLM distribution. This is motivated by it being a tractable proxy for maintaining LLM capabilities, but also by the idea that concealing a watermark deployment makes it harder for malicious actors to hide misuse by avoiding a certain LLM or attacking its watermark. Yet, despite much discourse around detectability, no prior work has investigated if any of these scheme families are detectable in a realistic black-box setting. We tackle this for the first time, developing rigorous statistical tests to detect the presence of all three most popular watermarking scheme families using only a limited number of black-box queries. We experimentally confirm the effectiveness of our methods on a range of schemes and a diverse set of open-source models. Our findings indicate that current watermarking schemes are more detectable than previously believed, and that obscuring the fact that a watermark was deployed may not be a viable way for providers to protect against adversaries. We further apply our methods to test for watermark presence behind the most popular public APIs: GPT4, Claude 3, Gemini 1.0 Pro, finding no strong evidence of a watermark at this point in time.



## **50. SpecFormer: Guarding Vision Transformer Robustness via Maximum Singular Value Penalization**

cs.CV

Accepted by ECCV 2024; 27 pages; code is at:  https://github.com/microsoft/robustlearn

**SubmitDate**: 2024-07-13    [abs](http://arxiv.org/abs/2402.03317v2) [paper-pdf](http://arxiv.org/pdf/2402.03317v2)

**Authors**: Xixu Hu, Runkai Zheng, Jindong Wang, Cheuk Hang Leung, Qi Wu, Xing Xie

**Abstract**: Vision Transformers (ViTs) are increasingly used in computer vision due to their high performance, but their vulnerability to adversarial attacks is a concern. Existing methods lack a solid theoretical basis, focusing mainly on empirical training adjustments. This study introduces SpecFormer, tailored to fortify ViTs against adversarial attacks, with theoretical underpinnings. We establish local Lipschitz bounds for the self-attention layer and propose the Maximum Singular Value Penalization (MSVP) to precisely manage these bounds By incorporating MSVP into ViTs' attention layers, we enhance the model's robustness without compromising training efficiency. SpecFormer, the resulting model, outperforms other state-of-the-art models in defending against adversarial attacks, as proven by experiments on CIFAR and ImageNet datasets. Code is released at https://github.com/microsoft/robustlearn.



