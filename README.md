# Latest Adversarial Attack Papers
**update at 2024-10-11 09:42:03**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. Protecting Your LLMs with Information Bottleneck**

cs.CL

Accepted by Neural Information Processing Systems (NeurIPS 2024)

**SubmitDate**: 2024-10-10    [abs](http://arxiv.org/abs/2404.13968v3) [paper-pdf](http://arxiv.org/pdf/2404.13968v3)

**Authors**: Zichuan Liu, Zefan Wang, Linjie Xu, Jinyu Wang, Lei Song, Tianchun Wang, Chunlin Chen, Wei Cheng, Jiang Bian

**Abstract**: The advent of large language models (LLMs) has revolutionized the field of natural language processing, yet they might be attacked to produce harmful content. Despite efforts to ethically align LLMs, these are often fragile and can be circumvented by jailbreaking attacks through optimized or manual adversarial prompts. To address this, we introduce the Information Bottleneck Protector (IBProtector), a defense mechanism grounded in the information bottleneck principle, and we modify the objective to avoid trivial solutions. The IBProtector selectively compresses and perturbs prompts, facilitated by a lightweight and trainable extractor, preserving only essential information for the target LLMs to respond with the expected answer. Moreover, we further consider a situation where the gradient is not visible to be compatible with any LLM. Our empirical evaluations show that IBProtector outperforms current defense methods in mitigating jailbreak attempts, without overly affecting response quality or inference speed. Its effectiveness and adaptability across various attack methods and target LLMs underscore the potential of IBProtector as a novel, transferable defense that bolsters the security of LLMs without requiring modifications to the underlying models.



## **2. MGMD-GAN: Generalization Improvement of Generative Adversarial Networks with Multiple Generator Multiple Discriminator Framework Against Membership Inference Attacks**

cs.LG

**SubmitDate**: 2024-10-10    [abs](http://arxiv.org/abs/2410.07803v1) [paper-pdf](http://arxiv.org/pdf/2410.07803v1)

**Authors**: Nirob Arefin

**Abstract**: Generative Adversarial Networks (GAN) are among the widely used Generative models in various applications. However, the original GAN architecture may memorize the distribution of the training data and, therefore, poses a threat to Membership Inference Attacks. In this work, we propose a new GAN framework that consists of Multiple Generators and Multiple Discriminators (MGMD-GAN). Disjoint partitions of the training data are used to train this model and it learns the mixture distribution of all the training data partitions. In this way, our proposed model reduces the generalization gap which makes our MGMD-GAN less vulnerable to Membership Inference Attacks. We provide an experimental analysis of our model and also a comparison with other GAN frameworks.



## **3. Invisibility Cloak: Disappearance under Human Pose Estimation via Backdoor Attacks**

cs.CR

**SubmitDate**: 2024-10-10    [abs](http://arxiv.org/abs/2410.07670v1) [paper-pdf](http://arxiv.org/pdf/2410.07670v1)

**Authors**: Minxing Zhang, Michael Backes, Xiao Zhang

**Abstract**: Human Pose Estimation (HPE) has been widely applied in autonomous systems such as self-driving cars. However, the potential risks of HPE to adversarial attacks have not received comparable attention with image classification or segmentation tasks. Existing works on HPE robustness focus on misleading an HPE system to provide wrong predictions that still indicate some human poses. In this paper, we study the vulnerability of HPE systems to disappearance attacks, where the attacker aims to subtly alter the HPE training process via backdoor techniques so that any input image with some specific trigger will not be recognized as involving any human pose. As humans are typically at the center of HPE systems, such attacks can induce severe security hazards, e.g., pedestrians' lives will be threatened if a self-driving car incorrectly understands the front scene due to disappearance attacks.   To achieve the adversarial goal of disappearance, we propose IntC, a general framework to craft Invisibility Cloak in the HPE domain. The core of our work lies in the design of target HPE labels that do not represent any human pose. In particular, we propose three specific backdoor attacks based on our IntC framework with different label designs. IntC-S and IntC-E, respectively designed for regression- and heatmap-based HPE techniques, concentrate the keypoints of triggered images in a tiny, imperceptible region. Further, to improve the attack's stealthiness, IntC-L designs the target poisons to capture the label outputs of typical landscape images without a human involved, achieving disappearance and reducing detectability simultaneously. Extensive experiments demonstrate the effectiveness and generalizability of our IntC methods in achieving the disappearance goal. By revealing the vulnerability of HPE to disappearance and backdoor attacks, we hope our work can raise awareness of the potential risks ...



## **4. Universally Optimal Watermarking Schemes for LLMs: from Theory to Practice**

cs.CR

**SubmitDate**: 2024-10-10    [abs](http://arxiv.org/abs/2410.02890v2) [paper-pdf](http://arxiv.org/pdf/2410.02890v2)

**Authors**: Haiyun He, Yepeng Liu, Ziqiao Wang, Yongyi Mao, Yuheng Bu

**Abstract**: Large Language Models (LLMs) boosts human efficiency but also poses misuse risks, with watermarking serving as a reliable method to differentiate AI-generated content from human-created text. In this work, we propose a novel theoretical framework for watermarking LLMs. Particularly, we jointly optimize both the watermarking scheme and detector to maximize detection performance, while controlling the worst-case Type-I error and distortion in the watermarked text. Within our framework, we characterize the universally minimum Type-II error, showing a fundamental trade-off between detection performance and distortion. More importantly, we identify the optimal type of detectors and watermarking schemes. Building upon our theoretical analysis, we introduce a practical, model-agnostic and computationally efficient token-level watermarking algorithm that invokes a surrogate model and the Gumbel-max trick. Empirical results on Llama-13B and Mistral-8$\times$7B demonstrate the effectiveness of our method. Furthermore, we also explore how robustness can be integrated into our theoretical framework, which provides a foundation for designing future watermarking systems with improved resilience to adversarial attacks.



## **5. Prompt-Agnostic Adversarial Perturbation for Customized Diffusion Models**

cs.CV

Accepted by NIPS 2024

**SubmitDate**: 2024-10-10    [abs](http://arxiv.org/abs/2408.10571v4) [paper-pdf](http://arxiv.org/pdf/2408.10571v4)

**Authors**: Cong Wan, Yuhang He, Xiang Song, Yihong Gong

**Abstract**: Diffusion models have revolutionized customized text-to-image generation, allowing for efficient synthesis of photos from personal data with textual descriptions. However, these advancements bring forth risks including privacy breaches and unauthorized replication of artworks. Previous researches primarily center around using prompt-specific methods to generate adversarial examples to protect personal images, yet the effectiveness of existing methods is hindered by constrained adaptability to different prompts. In this paper, we introduce a Prompt-Agnostic Adversarial Perturbation (PAP) method for customized diffusion models. PAP first models the prompt distribution using a Laplace Approximation, and then produces prompt-agnostic perturbations by maximizing a disturbance expectation based on the modeled distribution. This approach effectively tackles the prompt-agnostic attacks, leading to improved defense stability. Extensive experiments in face privacy and artistic style protection, demonstrate the superior generalization of PAP in comparison to existing techniques. Our project page is available at https://github.com/vancyland/Prompt-Agnostic-Adversarial-Perturbation-for-Customized-Diffusion-Models.github.io.



## **6. MorCode: Face Morphing Attack Generation using Generative Codebooks**

cs.CV

**SubmitDate**: 2024-10-10    [abs](http://arxiv.org/abs/2410.07625v1) [paper-pdf](http://arxiv.org/pdf/2410.07625v1)

**Authors**: Aravinda Reddy PN, Raghavendra Ramachandra, Sushma Venkatesh, Krothapalli Sreenivasa Rao, Pabitra Mitra, Rakesh Krishna

**Abstract**: Face recognition systems (FRS) can be compromised by face morphing attacks, which blend textural and geometric information from multiple facial images. The rapid evolution of generative AI, especially Generative Adversarial Networks (GAN) or Diffusion models, where encoded images are interpolated to generate high-quality face morphing images. In this work, we present a novel method for the automatic face morphing generation method \textit{MorCode}, which leverages a contemporary encoder-decoder architecture conditioned on codebook learning to generate high-quality morphing images. Extensive experiments were performed on the newly constructed morphing dataset using five state-of-the-art morphing generation techniques using both digital and print-scan data. The attack potential of the proposed morphing generation technique, \textit{MorCode}, was benchmarked using three different face recognition systems. The obtained results indicate the highest attack potential of the proposed \textit{MorCode} when compared with five state-of-the-art morphing generation methods on both digital and print scan data.



## **7. An undetectable watermark for generative image models**

cs.CR

**SubmitDate**: 2024-10-09    [abs](http://arxiv.org/abs/2410.07369v1) [paper-pdf](http://arxiv.org/pdf/2410.07369v1)

**Authors**: Sam Gunn, Xuandong Zhao, Dawn Song

**Abstract**: We present the first undetectable watermarking scheme for generative image models. Undetectability ensures that no efficient adversary can distinguish between watermarked and un-watermarked images, even after making many adaptive queries. In particular, an undetectable watermark does not degrade image quality under any efficiently computable metric. Our scheme works by selecting the initial latents of a diffusion model using a pseudorandom error-correcting code (Christ and Gunn, 2024), a strategy which guarantees undetectability and robustness. We experimentally demonstrate that our watermarks are quality-preserving and robust using Stable Diffusion 2.1. Our experiments verify that, in contrast to every prior scheme we tested, our watermark does not degrade image quality. Our experiments also demonstrate robustness: existing watermark removal attacks fail to remove our watermark from images without significantly degrading the quality of the images. Finally, we find that we can robustly encode 512 bits in our watermark, and up to 2500 bits when the images are not subjected to watermark removal attacks. Our code is available at https://github.com/XuandongZhao/PRC-Watermark.



## **8. JPEG Inspired Deep Learning**

cs.CV

**SubmitDate**: 2024-10-09    [abs](http://arxiv.org/abs/2410.07081v1) [paper-pdf](http://arxiv.org/pdf/2410.07081v1)

**Authors**: Ahmed H. Salamah, Kaixiang Zheng, Yiwen Liu, En-Hui Yang

**Abstract**: Although it is traditionally believed that lossy image compression, such as JPEG compression, has a negative impact on the performance of deep neural networks (DNNs), it is shown by recent works that well-crafted JPEG compression can actually improve the performance of deep learning (DL). Inspired by this, we propose JPEG-DL, a novel DL framework that prepends any underlying DNN architecture with a trainable JPEG compression layer. To make the quantization operation in JPEG compression trainable, a new differentiable soft quantizer is employed at the JPEG layer, and then the quantization operation and underlying DNN are jointly trained. Extensive experiments show that in comparison with the standard DL, JPEG-DL delivers significant accuracy improvements across various datasets and model architectures while enhancing robustness against adversarial attacks. Particularly, on some fine-grained image classification datasets, JPEG-DL can increase prediction accuracy by as much as 20.9%. Our code is available on https://github.com/JpegInspiredDl/JPEG-Inspired-DL.git.



## **9. SAGMAN: Stability Analysis of Graph Neural Networks on the Manifolds**

cs.LG

**SubmitDate**: 2024-10-09    [abs](http://arxiv.org/abs/2402.08653v4) [paper-pdf](http://arxiv.org/pdf/2402.08653v4)

**Authors**: Wuxinlin Cheng, Chenhui Deng, Ali Aghdaei, Zhiru Zhang, Zhuo Feng

**Abstract**: Modern graph neural networks (GNNs) can be sensitive to changes in the input graph structure and node features, potentially resulting in unpredictable behavior and degraded performance. In this work, we introduce a spectral framework known as SAGMAN for examining the stability of GNNs. This framework assesses the distance distortions that arise from the nonlinear mappings of GNNs between the input and output manifolds: when two nearby nodes on the input manifold are mapped (through a GNN model) to two distant ones on the output manifold, it implies a large distance distortion and thus a poor GNN stability. We propose a distance-preserving graph dimension reduction (GDR) approach that utilizes spectral graph embedding and probabilistic graphical models (PGMs) to create low-dimensional input/output graph-based manifolds for meaningful stability analysis. Our empirical evaluations show that SAGMAN effectively assesses the stability of each node when subjected to various edge or feature perturbations, offering a scalable approach for evaluating the stability of GNNs, extending to applications within recommendation systems. Furthermore, we illustrate its utility in downstream tasks, notably in enhancing GNN stability and facilitating adversarial targeted attacks.



## **10. Defensive Unlearning with Adversarial Training for Robust Concept Erasure in Diffusion Models**

cs.CV

Accepted by NeurIPS'24. Codes are available at  https://github.com/OPTML-Group/AdvUnlearn

**SubmitDate**: 2024-10-09    [abs](http://arxiv.org/abs/2405.15234v3) [paper-pdf](http://arxiv.org/pdf/2405.15234v3)

**Authors**: Yimeng Zhang, Xin Chen, Jinghan Jia, Yihua Zhang, Chongyu Fan, Jiancheng Liu, Mingyi Hong, Ke Ding, Sijia Liu

**Abstract**: Diffusion models (DMs) have achieved remarkable success in text-to-image generation, but they also pose safety risks, such as the potential generation of harmful content and copyright violations. The techniques of machine unlearning, also known as concept erasing, have been developed to address these risks. However, these techniques remain vulnerable to adversarial prompt attacks, which can prompt DMs post-unlearning to regenerate undesired images containing concepts (such as nudity) meant to be erased. This work aims to enhance the robustness of concept erasing by integrating the principle of adversarial training (AT) into machine unlearning, resulting in the robust unlearning framework referred to as AdvUnlearn. However, achieving this effectively and efficiently is highly nontrivial. First, we find that a straightforward implementation of AT compromises DMs' image generation quality post-unlearning. To address this, we develop a utility-retaining regularization on an additional retain set, optimizing the trade-off between concept erasure robustness and model utility in AdvUnlearn. Moreover, we identify the text encoder as a more suitable module for robustification compared to UNet, ensuring unlearning effectiveness. And the acquired text encoder can serve as a plug-and-play robust unlearner for various DM types. Empirically, we perform extensive experiments to demonstrate the robustness advantage of AdvUnlearn across various DM unlearning scenarios, including the erasure of nudity, objects, and style concepts. In addition to robustness, AdvUnlearn also achieves a balanced tradeoff with model utility. To our knowledge, this is the first work to systematically explore robust DM unlearning through AT, setting it apart from existing methods that overlook robustness in concept erasing. Codes are available at: https://github.com/OPTML-Group/AdvUnlearn



## **11. The Vital Role of Gradient Clipping in Byzantine-Resilient Distributed Learning**

cs.LG

**SubmitDate**: 2024-10-09    [abs](http://arxiv.org/abs/2405.14432v4) [paper-pdf](http://arxiv.org/pdf/2405.14432v4)

**Authors**: Youssef Allouah, Rachid Guerraoui, Nirupam Gupta, Ahmed Jellouli, Geovani Rizk, John Stephan

**Abstract**: Byzantine-resilient distributed machine learning seeks to achieve robust learning performance in the presence of misbehaving or adversarial workers. While state-of-the-art (SOTA) robust distributed gradient descent (Robust-DGD) methods were proven theoretically optimal, their empirical success has often relied on pre-aggregation gradient clipping. However, the currently considered static clipping strategy exhibits mixed results: improving robustness against some attacks while being ineffective or detrimental against others. We address this gap by proposing a principled adaptive clipping strategy, termed Adaptive Robust Clipping (ARC). We show that ARC consistently enhances the empirical robustness of SOTA Robust-DGD methods, while preserving the theoretical robustness guarantees. Our analysis shows that ARC provably improves the asymptotic convergence guarantee of Robust-DGD in the case when the model is well-initialized. We validate this theoretical insight through an exhaustive set of experiments on benchmark image classification tasks. We observe that the improvement induced by ARC is more pronounced in highly heterogeneous and adversarial settings.



## **12. Average Certified Radius is a Poor Metric for Randomized Smoothing**

cs.LG

**SubmitDate**: 2024-10-09    [abs](http://arxiv.org/abs/2410.06895v1) [paper-pdf](http://arxiv.org/pdf/2410.06895v1)

**Authors**: Chenhao Sun, Yuhao Mao, Mark Niklas Müller, Martin Vechev

**Abstract**: Randomized smoothing is a popular approach for providing certified robustness guarantees against adversarial attacks, and has become a very active area of research. Over the past years, the average certified radius (ACR) has emerged as the single most important metric for comparing methods and tracking progress in the field. However, in this work, we show that ACR is an exceptionally poor metric for evaluating robustness guarantees provided by randomized smoothing. We theoretically show not only that a trivial classifier can have arbitrarily large ACR, but also that ACR is much more sensitive to improvements on easy samples than on hard ones. Empirically, we confirm that existing training strategies that improve ACR reduce the model's robustness on hard samples. Further, we show that by focusing on easy samples, we can effectively replicate the increase in ACR. We develop strategies, including explicitly discarding hard samples, reweighing the dataset with certified radius, and extreme optimization for easy samples, to achieve state-of-the-art ACR, although these strategies ignore robustness for the general data distribution. Overall, our results suggest that ACR has introduced a strong undesired bias to the field, and better metrics are required to holistically evaluate randomized smoothing.



## **13. Secure Video Quality Assessment Resisting Adversarial Attacks**

cs.CV

**SubmitDate**: 2024-10-09    [abs](http://arxiv.org/abs/2410.06866v1) [paper-pdf](http://arxiv.org/pdf/2410.06866v1)

**Authors**: Ao-Xiang Zhang, Yu Ran, Weixuan Tang, Yuan-Gen Wang, Qingxiao Guan, Chunsheng Yang

**Abstract**: The exponential surge in video traffic has intensified the imperative for Video Quality Assessment (VQA). Leveraging cutting-edge architectures, current VQA models have achieved human-comparable accuracy. However, recent studies have revealed the vulnerability of existing VQA models against adversarial attacks. To establish a reliable and practical assessment system, a secure VQA model capable of resisting such malicious attacks is urgently demanded. Unfortunately, no attempt has been made to explore this issue. This paper first attempts to investigate general adversarial defense principles, aiming at endowing existing VQA models with security. Specifically, we first introduce random spatial grid sampling on the video frame for intra-frame defense. Then, we design pixel-wise randomization through a guardian map, globally neutralizing adversarial perturbations. Meanwhile, we extract temporal information from the video sequence as compensation for inter-frame defense. Building upon these principles, we present a novel VQA framework from the security-oriented perspective, termed SecureVQA. Extensive experiments indicate that SecureVQA sets a new benchmark in security while achieving competitive VQA performance compared with state-of-the-art models. Ablation studies delve deeper into analyzing the principles of SecureVQA, demonstrating their generalization and contributions to the security of leading VQA models.



## **14. Understanding Model Ensemble in Transferable Adversarial Attack**

cs.LG

**SubmitDate**: 2024-10-09    [abs](http://arxiv.org/abs/2410.06851v1) [paper-pdf](http://arxiv.org/pdf/2410.06851v1)

**Authors**: Wei Yao, Zeliang Zhang, Huayi Tang, Yong Liu

**Abstract**: Model ensemble adversarial attack has become a powerful method for generating transferable adversarial examples that can target even unknown models, but its theoretical foundation remains underexplored. To address this gap, we provide early theoretical insights that serve as a roadmap for advancing model ensemble adversarial attack. We first define transferability error to measure the error in adversarial transferability, alongside concepts of diversity and empirical model ensemble Rademacher complexity. We then decompose the transferability error into vulnerability, diversity, and a constant, which rigidly explains the origin of transferability error in model ensemble attack: the vulnerability of an adversarial example to ensemble components, and the diversity of ensemble components. Furthermore, we apply the latest mathematical tools in information theory to bound the transferability error using complexity and generalization terms, contributing to three practical guidelines for reducing transferability error: (1) incorporating more surrogate models, (2) increasing their diversity, and (3) reducing their complexity in cases of overfitting. Finally, extensive experiments with 54 models validate our theoretical framework, representing a significant step forward in understanding transferable model ensemble adversarial attacks.



## **15. On the Byzantine-Resilience of Distillation-Based Federated Learning**

cs.LG

**SubmitDate**: 2024-10-09    [abs](http://arxiv.org/abs/2402.12265v2) [paper-pdf](http://arxiv.org/pdf/2402.12265v2)

**Authors**: Christophe Roux, Max Zimmer, Sebastian Pokutta

**Abstract**: Federated Learning (FL) algorithms using Knowledge Distillation (KD) have received increasing attention due to their favorable properties with respect to privacy, non-i.i.d. data and communication cost. These methods depart from transmitting model parameters and instead communicate information about a learning task by sharing predictions on a public dataset. In this work, we study the performance of such approaches in the byzantine setting, where a subset of the clients act in an adversarial manner aiming to disrupt the learning process. We show that KD-based FL algorithms are remarkably resilient and analyze how byzantine clients can influence the learning process. Based on these insights, we introduce two new byzantine attacks and demonstrate their ability to break existing byzantine-resilient methods. Additionally, we propose a novel defence method which enhances the byzantine resilience of KD-based FL algorithms. Finally, we provide a general framework to obfuscate attacks, making them significantly harder to detect, thereby improving their effectiveness. Our findings serve as an important building block in the analysis of byzantine FL, contributing through the development of new attacks and new defence mechanisms, further advancing the robustness of KD-based FL algorithms.



## **16. Read Over the Lines: Attacking LLMs and Toxicity Detection Systems with ASCII Art to Mask Profanity**

cs.CL

**SubmitDate**: 2024-10-09    [abs](http://arxiv.org/abs/2409.18708v4) [paper-pdf](http://arxiv.org/pdf/2409.18708v4)

**Authors**: Sergey Berezin, Reza Farahbakhsh, Noel Crespi

**Abstract**: We introduce a novel family of adversarial attacks that exploit the inability of language models to interpret ASCII art. To evaluate these attacks, we propose the ToxASCII benchmark and develop two custom ASCII art fonts: one leveraging special tokens and another using text-filled letter shapes. Our attacks achieve a perfect 1.0 Attack Success Rate across ten models, including OpenAI's o1-preview and LLaMA 3.1.   Warning: this paper contains examples of toxic language used for research purposes.



## **17. Faithfulness and the Notion of Adversarial Sensitivity in NLP Explanations**

cs.CL

Accepted as a Full Paper at EMNLP 2024 Workshop BlackBoxNLP

**SubmitDate**: 2024-10-09    [abs](http://arxiv.org/abs/2409.17774v2) [paper-pdf](http://arxiv.org/pdf/2409.17774v2)

**Authors**: Supriya Manna, Niladri Sett

**Abstract**: Faithfulness is arguably the most critical metric to assess the reliability of explainable AI. In NLP, current methods for faithfulness evaluation are fraught with discrepancies and biases, often failing to capture the true reasoning of models. We introduce Adversarial Sensitivity as a novel approach to faithfulness evaluation, focusing on the explainer's response when the model is under adversarial attack. Our method accounts for the faithfulness of explainers by capturing sensitivity to adversarial input changes. This work addresses significant limitations in existing evaluation techniques, and furthermore, quantifies faithfulness from a crucial yet underexplored paradigm.



## **18. TASAR: Transfer-based Attack on Skeletal Action Recognition**

cs.CV

**SubmitDate**: 2024-10-09    [abs](http://arxiv.org/abs/2409.02483v2) [paper-pdf](http://arxiv.org/pdf/2409.02483v2)

**Authors**: Yunfeng Diao, Baiqi Wu, Ruixuan Zhang, Ajian Liu, Xingxing Wei, Meng Wang, He Wang

**Abstract**: Skeletal sequences, as well-structured representations of human behaviors, play a vital role in Human Activity Recognition (HAR). The transferability of adversarial skeletal sequences enables attacks in real-world HAR scenarios, such as autonomous driving, intelligent surveillance, and human-computer interactions. However, most existing skeleton-based HAR (S-HAR) attacks are primarily designed for white-box scenarios and exhibit weak adversarial transferability. Therefore, they cannot be considered true transfer-based S-HAR attacks. More importantly, the reason for this failure remains unclear. In this paper, we study this phenomenon through the lens of loss surface, and find that its sharpness contributes to the weak transferability in S-HAR. Inspired by this observation, we assume and empirically validate that smoothening the rugged loss landscape could potentially improve adversarial transferability in S-HAR. To this end, we propose the first \textbf{T}ransfer-based \textbf{A}ttack on \textbf{S}keletal \textbf{A}ction \textbf{R}ecognition, TASAR. TASAR explores the smoothed model posterior without requiring surrogate re-training, which is achieved by a new post-train Dual Bayesian optimization strategy. Furthermore, unlike previous transfer-based attacks that treat each frame independently and overlook temporal coherence within sequences, TASAR incorporates motion dynamics into the Bayesian attack gradient, effectively disrupting the spatial-temporal coherence of S-HARs. To exhaustively evaluate the effectiveness of existing methods and our method, we build the first large-scale robust S-HAR benchmark, comprising 7 S-HAR models, 10 attack methods, 3 S-HAR datasets and 2 defense methods. Extensive results demonstrate the superiority of TASAR. Our benchmark enables easy comparisons for future studies, with the code available in the supplementary material.



## **19. PII-Scope: A Benchmark for Training Data PII Leakage Assessment in LLMs**

cs.CL

**SubmitDate**: 2024-10-09    [abs](http://arxiv.org/abs/2410.06704v1) [paper-pdf](http://arxiv.org/pdf/2410.06704v1)

**Authors**: Krishna Kanth Nakka, Ahmed Frikha, Ricardo Mendes, Xue Jiang, Xuebing Zhou

**Abstract**: In this work, we introduce PII-Scope, a comprehensive benchmark designed to evaluate state-of-the-art methodologies for PII extraction attacks targeting LLMs across diverse threat settings. Our study provides a deeper understanding of these attacks by uncovering several hyperparameters (e.g., demonstration selection) crucial to their effectiveness. Building on this understanding, we extend our study to more realistic attack scenarios, exploring PII attacks that employ advanced adversarial strategies, including repeated and diverse querying, and leveraging iterative learning for continual PII extraction. Through extensive experimentation, our results reveal a notable underestimation of PII leakage in existing single-query attacks. In fact, we show that with sophisticated adversarial capabilities and a limited query budget, PII extraction rates can increase by up to fivefold when targeting the pretrained model. Moreover, we evaluate PII leakage on finetuned models, showing that they are more vulnerable to leakage than pretrained models. Overall, our work establishes a rigorous empirical benchmark for PII extraction attacks in realistic threat scenarios and provides a strong foundation for developing effective mitigation strategies.



## **20. Break the Visual Perception: Adversarial Attacks Targeting Encoded Visual Tokens of Large Vision-Language Models**

cs.CV

Accepted to ACMMM 2024

**SubmitDate**: 2024-10-09    [abs](http://arxiv.org/abs/2410.06699v1) [paper-pdf](http://arxiv.org/pdf/2410.06699v1)

**Authors**: Yubo Wang, Chaohu Liu, Yanqiu Qu, Haoyu Cao, Deqiang Jiang, Linli Xu

**Abstract**: Large vision-language models (LVLMs) integrate visual information into large language models, showcasing remarkable multi-modal conversational capabilities. However, the visual modules introduces new challenges in terms of robustness for LVLMs, as attackers can craft adversarial images that are visually clean but may mislead the model to generate incorrect answers. In general, LVLMs rely on vision encoders to transform images into visual tokens, which are crucial for the language models to perceive image contents effectively. Therefore, we are curious about one question: Can LVLMs still generate correct responses when the encoded visual tokens are attacked and disrupting the visual information? To this end, we propose a non-targeted attack method referred to as VT-Attack (Visual Tokens Attack), which constructs adversarial examples from multiple perspectives, with the goal of comprehensively disrupting feature representations and inherent relationships as well as the semantic properties of visual tokens output by image encoders. Using only access to the image encoder in the proposed attack, the generated adversarial examples exhibit transferability across diverse LVLMs utilizing the same image encoder and generality across different tasks. Extensive experiments validate the superior attack performance of the VT-Attack over baseline methods, demonstrating its effectiveness in attacking LVLMs with image encoders, which in turn can provide guidance on the robustness of LVLMs, particularly in terms of the stability of the visual feature space.



## **21. Does Vec2Text Pose a New Corpus Poisoning Threat?**

cs.IR

arXiv admin note: substantial text overlap with arXiv:2402.12784

**SubmitDate**: 2024-10-09    [abs](http://arxiv.org/abs/2410.06628v1) [paper-pdf](http://arxiv.org/pdf/2410.06628v1)

**Authors**: Shengyao Zhuang, Bevan Koopman, Guido Zuccon

**Abstract**: The emergence of Vec2Text -- a method for text embedding inversion -- has raised serious privacy concerns for dense retrieval systems which use text embeddings. This threat comes from the ability for an attacker with access to embeddings to reconstruct the original text. In this paper, we take a new look at Vec2Text and investigate how much of a threat it poses to the different attacks of corpus poisoning, whereby an attacker injects adversarial passages into a retrieval corpus with the intention of misleading dense retrievers. Theoretically, Vec2Text is far more dangerous than previous attack methods because it does not need access to the embedding model's weights and it can efficiently generate many adversarial passages. We show that under certain conditions, corpus poisoning with Vec2Text can pose a serious threat to dense retriever system integrity and user experience by injecting adversarial passaged into top ranked positions. Code and data are made available at https://github.com/ielab/vec2text-corpus-poisoning



## **22. ETA: Evaluating Then Aligning Safety of Vision Language Models at Inference Time**

cs.CV

27pages

**SubmitDate**: 2024-10-09    [abs](http://arxiv.org/abs/2410.06625v1) [paper-pdf](http://arxiv.org/pdf/2410.06625v1)

**Authors**: Yi Ding, Bolian Li, Ruqi Zhang

**Abstract**: Vision Language Models (VLMs) have become essential backbones for multimodal intelligence, yet significant safety challenges limit their real-world application. While textual inputs are often effectively safeguarded, adversarial visual inputs can easily bypass VLM defense mechanisms. Existing defense methods are either resource-intensive, requiring substantial data and compute, or fail to simultaneously ensure safety and usefulness in responses. To address these limitations, we propose a novel two-phase inference-time alignment framework, Evaluating Then Aligning (ETA): 1) Evaluating input visual contents and output responses to establish a robust safety awareness in multimodal settings, and 2) Aligning unsafe behaviors at both shallow and deep levels by conditioning the VLMs' generative distribution with an interference prefix and performing sentence-level best-of-N to search the most harmless and helpful generation paths. Extensive experiments show that ETA outperforms baseline methods in terms of harmlessness, helpfulness, and efficiency, reducing the unsafe rate by 87.5% in cross-modality attacks and achieving 96.6% win-ties in GPT-4 helpfulness evaluation. The code is publicly available at https://github.com/DripNowhy/ETA.



## **23. Can DeepFake Speech be Reliably Detected?**

cs.SD

**SubmitDate**: 2024-10-09    [abs](http://arxiv.org/abs/2410.06572v1) [paper-pdf](http://arxiv.org/pdf/2410.06572v1)

**Authors**: Hongbin Liu, Youzheng Chen, Arun Narayanan, Athula Balachandran, Pedro J. Moreno, Lun Wang

**Abstract**: Recent advances in text-to-speech (TTS) systems, particularly those with voice cloning capabilities, have made voice impersonation readily accessible, raising ethical and legal concerns due to potential misuse for malicious activities like misinformation campaigns and fraud. While synthetic speech detectors (SSDs) exist to combat this, they are vulnerable to ``test domain shift", exhibiting decreased performance when audio is altered through transcoding, playback, or background noise. This vulnerability is further exacerbated by deliberate manipulation of synthetic speech aimed at deceiving detectors. This work presents the first systematic study of such active malicious attacks against state-of-the-art open-source SSDs. White-box attacks, black-box attacks, and their transferability are studied from both attack effectiveness and stealthiness, using both hardcoded metrics and human ratings. The results highlight the urgent need for more robust detection methods in the face of evolving adversarial threats.



## **24. Hallucinating AI Hijacking Attack: Large Language Models and Malicious Code Recommenders**

cs.CR

**SubmitDate**: 2024-10-09    [abs](http://arxiv.org/abs/2410.06462v1) [paper-pdf](http://arxiv.org/pdf/2410.06462v1)

**Authors**: David Noever, Forrest McKee

**Abstract**: The research builds and evaluates the adversarial potential to introduce copied code or hallucinated AI recommendations for malicious code in popular code repositories. While foundational large language models (LLMs) from OpenAI, Google, and Anthropic guard against both harmful behaviors and toxic strings, previous work on math solutions that embed harmful prompts demonstrate that the guardrails may differ between expert contexts. These loopholes would appear in mixture of expert's models when the context of the question changes and may offer fewer malicious training examples to filter toxic comments or recommended offensive actions. The present work demonstrates that foundational models may refuse to propose destructive actions correctly when prompted overtly but may unfortunately drop their guard when presented with a sudden change of context, like solving a computer programming challenge. We show empirical examples with trojan-hosting repositories like GitHub, NPM, NuGet, and popular content delivery networks (CDN) like jsDelivr which amplify the attack surface. In the LLM's directives to be helpful, example recommendations propose application programming interface (API) endpoints which a determined domain-squatter could acquire and setup attack mobile infrastructure that triggers from the naively copied code. We compare this attack to previous work on context-shifting and contrast the attack surface as a novel version of "living off the land" attacks in the malware literature. In the latter case, foundational language models can hijack otherwise innocent user prompts to recommend actions that violate their owners' safety policies when posed directly without the accompanying coding support request.



## **25. Filtered Randomized Smoothing: A New Defense for Robust Modulation Classification**

cs.LG

IEEE Milcom 2024

**SubmitDate**: 2024-10-08    [abs](http://arxiv.org/abs/2410.06339v1) [paper-pdf](http://arxiv.org/pdf/2410.06339v1)

**Authors**: Wenhan Zhang, Meiyu Zhong, Ravi Tandon, Marwan Krunz

**Abstract**: Deep Neural Network (DNN) based classifiers have recently been used for the modulation classification of RF signals. These classifiers have shown impressive performance gains relative to conventional methods, however, they are vulnerable to imperceptible (low-power) adversarial attacks. Some of the prominent defense approaches include adversarial training (AT) and randomized smoothing (RS). While AT increases robustness in general, it fails to provide resilience against previously unseen adaptive attacks. Other approaches, such as Randomized Smoothing (RS), which injects noise into the input, address this shortcoming by providing provable certified guarantees against arbitrary attacks, however, they tend to sacrifice accuracy.   In this paper, we study the problem of designing robust DNN-based modulation classifiers that can provide provable defense against arbitrary attacks without significantly sacrificing accuracy. To this end, we first analyze the spectral content of commonly studied attacks on modulation classifiers for the benchmark RadioML dataset. We observe that spectral signatures of un-perturbed RF signals are highly localized, whereas attack signals tend to be spread out in frequency. To exploit this spectral heterogeneity, we propose Filtered Randomized Smoothing (FRS), a novel defense which combines spectral filtering together with randomized smoothing. FRS can be viewed as a strengthening of RS by leveraging the specificity (spectral Heterogeneity) inherent to the modulation classification problem. In addition to providing an approach to compute the certified accuracy of FRS, we also provide a comprehensive set of simulations on the RadioML dataset to show the effectiveness of FRS and show that it significantly outperforms existing defenses including AT and RS in terms of accuracy on both attacked and benign signals.



## **26. Evaluating and Safeguarding the Adversarial Robustness of Retrieval-Based In-Context Learning**

cs.CL

COLM 2024, 31 pages, 6 figures

**SubmitDate**: 2024-10-08    [abs](http://arxiv.org/abs/2405.15984v4) [paper-pdf](http://arxiv.org/pdf/2405.15984v4)

**Authors**: Simon Yu, Jie He, Pasquale Minervini, Jeff Z. Pan

**Abstract**: With the emergence of large language models, such as LLaMA and OpenAI GPT-3, In-Context Learning (ICL) gained significant attention due to its effectiveness and efficiency. However, ICL is very sensitive to the choice, order, and verbaliser used to encode the demonstrations in the prompt. Retrieval-Augmented ICL methods try to address this problem by leveraging retrievers to extract semantically related examples as demonstrations. While this approach yields more accurate results, its robustness against various types of adversarial attacks, including perturbations on test samples, demonstrations, and retrieved data, remains under-explored. Our study reveals that retrieval-augmented models can enhance robustness against test sample attacks, outperforming vanilla ICL with a 4.87% reduction in Attack Success Rate (ASR); however, they exhibit overconfidence in the demonstrations, leading to a 2% increase in ASR for demonstration attacks. Adversarial training can help improve the robustness of ICL methods to adversarial attacks; however, such a training scheme can be too costly in the context of LLMs. As an alternative, we introduce an effective training-free adversarial defence method, DARD, which enriches the example pool with those attacked samples. We show that DARD yields improvements in performance and robustness, achieving a 15% reduction in ASR over the baselines. Code and data are released to encourage further research: https://github.com/simonucl/adv-retreival-icl



## **27. The Last Iterate Advantage: Empirical Auditing and Principled Heuristic Analysis of Differentially Private SGD**

cs.CR

**SubmitDate**: 2024-10-10    [abs](http://arxiv.org/abs/2410.06186v2) [paper-pdf](http://arxiv.org/pdf/2410.06186v2)

**Authors**: Thomas Steinke, Milad Nasr, Arun Ganesh, Borja Balle, Christopher A. Choquette-Choo, Matthew Jagielski, Jamie Hayes, Abhradeep Guha Thakurta, Adam Smith, Andreas Terzis

**Abstract**: We propose a simple heuristic privacy analysis of noisy clipped stochastic gradient descent (DP-SGD) in the setting where only the last iterate is released and the intermediate iterates remain hidden. Namely, our heuristic assumes a linear structure for the model.   We show experimentally that our heuristic is predictive of the outcome of privacy auditing applied to various training procedures. Thus it can be used prior to training as a rough estimate of the final privacy leakage. We also probe the limitations of our heuristic by providing some artificial counterexamples where it underestimates the privacy leakage.   The standard composition-based privacy analysis of DP-SGD effectively assumes that the adversary has access to all intermediate iterates, which is often unrealistic. However, this analysis remains the state of the art in practice. While our heuristic does not replace a rigorous privacy analysis, it illustrates the large gap between the best theoretical upper bounds and the privacy auditing lower bounds and sets a target for further work to improve the theoretical privacy analyses. We also empirically support our heuristic and show existing privacy auditing attacks are bounded by our heuristic analysis in both vision and language tasks.



## **28. Position: Towards Resilience Against Adversarial Examples**

cs.LG

**SubmitDate**: 2024-10-08    [abs](http://arxiv.org/abs/2405.01349v2) [paper-pdf](http://arxiv.org/pdf/2405.01349v2)

**Authors**: Sihui Dai, Chong Xiang, Tong Wu, Prateek Mittal

**Abstract**: Current research on defending against adversarial examples focuses primarily on achieving robustness against a single attack type such as $\ell_2$ or $\ell_{\infty}$-bounded attacks. However, the space of possible perturbations is much larger than considered by many existing defenses and is difficult to mathematically model, so the attacker can easily bypass the defense by using a type of attack that is not covered by the defense. In this position paper, we argue that in addition to robustness, we should also aim to develop defense algorithms that are adversarially resilient -- defense algorithms should specify a means to quickly adapt the defended model to be robust against new attacks. We provide a definition of adversarial resilience and outline considerations of designing an adversarially resilient defense. We then introduce a subproblem of adversarial resilience which we call continual adaptive robustness, in which the defender gains knowledge of the formulation of possible perturbation spaces over time and can then update their model based on this information. Additionally, we demonstrate the connection between continual adaptive robustness and previously studied problems of multiattack robustness and unforeseen attack robustness and outline open directions within these fields which can contribute to improving continual adaptive robustness and adversarial resilience.



## **29. One Perturbation is Enough: On Generating Universal Adversarial Perturbations against Vision-Language Pre-training Models**

cs.CV

**SubmitDate**: 2024-10-08    [abs](http://arxiv.org/abs/2406.05491v2) [paper-pdf](http://arxiv.org/pdf/2406.05491v2)

**Authors**: Hao Fang, Jiawei Kong, Wenbo Yu, Bin Chen, Jiawei Li, Shutao Xia, Ke Xu

**Abstract**: Vision-Language Pre-training (VLP) models have exhibited unprecedented capability in many applications by taking full advantage of the multimodal alignment. However, previous studies have shown they are vulnerable to maliciously crafted adversarial samples. Despite recent success, these methods are generally instance-specific and require generating perturbations for each input sample. In this paper, we reveal that VLP models are also vulnerable to the instance-agnostic universal adversarial perturbation (UAP). Specifically, we design a novel Contrastive-training Perturbation Generator with Cross-modal conditions (C-PGC) to achieve the attack. In light that the pivotal multimodal alignment is achieved through the advanced contrastive learning technique, we devise to turn this powerful weapon against themselves, i.e., employ a malicious version of contrastive learning to train the C-PGC based on our carefully crafted positive and negative image-text pairs for essentially destroying the alignment relationship learned by VLP models. Besides, C-PGC fully utilizes the characteristics of Vision-and-Language (V+L) scenarios by incorporating both unimodal and cross-modal information as effective guidance. Extensive experiments show that C-PGC successfully forces adversarial samples to move away from their original area in the VLP model's feature space, thus essentially enhancing attacks across various victim models and V+L tasks. The GitHub repository is available at https://github.com/ffhibnese/CPGC_VLP_Universal_Attacks.



## **30. Transferability Bound Theory: Exploring Relationship between Adversarial Transferability and Flatness**

cs.LG

Accepted by NeurIPS 2024

**SubmitDate**: 2024-10-08    [abs](http://arxiv.org/abs/2311.06423v3) [paper-pdf](http://arxiv.org/pdf/2311.06423v3)

**Authors**: Mingyuan Fan, Xiaodan Li, Cen Chen, Wenmeng Zhou, Yaliang Li

**Abstract**: A prevailing belief in attack and defense community is that the higher flatness of adversarial examples enables their better cross-model transferability, leading to a growing interest in employing sharpness-aware minimization and its variants. However, the theoretical relationship between the transferability of adversarial examples and their flatness has not been well established, making the belief questionable. To bridge this gap, we embark on a theoretical investigation and, for the first time, derive a theoretical bound for the transferability of adversarial examples with few practical assumptions. Our analysis challenges this belief by demonstrating that the increased flatness of adversarial examples does not necessarily guarantee improved transferability. Moreover, building upon the theoretical analysis, we propose TPA, a Theoretically Provable Attack that optimizes a surrogate of the derived bound to craft adversarial examples. Extensive experiments across widely used benchmark datasets and various real-world applications show that TPA can craft more transferable adversarial examples compared to state-of-the-art baselines. We hope that these results can recalibrate preconceived impressions within the community and facilitate the development of stronger adversarial attack and defense mechanisms. The source codes are available in <https://github.com/fmy266/TPA>.



## **31. Partially Recentralization Softmax Loss for Vision-Language Models Robustness**

cs.CL

**SubmitDate**: 2024-10-08    [abs](http://arxiv.org/abs/2402.03627v2) [paper-pdf](http://arxiv.org/pdf/2402.03627v2)

**Authors**: Hao Wang, Jinzhe Jiang, Xin Zhang, Chen Li

**Abstract**: As Large Language Models make a breakthrough in natural language processing tasks (NLP), multimodal technique becomes extremely popular. However, it has been shown that multimodal NLP are vulnerable to adversarial attacks, where the outputs of a model can be dramatically changed by a perturbation to the input. While several defense techniques have been proposed both in computer vision and NLP models, the multimodal robustness of models have not been fully explored. In this paper, we study the adversarial robustness provided by modifying loss function of pre-trained multimodal models, by restricting top K softmax outputs. Based on the evaluation and scoring, our experiments show that after a fine-tuning, adversarial robustness of pre-trained models can be significantly improved, against popular attacks. Further research should be studying, such as output diversity, generalization and the robustness-performance trade-off of this kind of loss functions. Our code will be available after this paper is accepted



## **32. ATM: Adversarial Tuning Multi-agent System Makes a Robust Retrieval-Augmented Generator**

cs.CL

18 pages

**SubmitDate**: 2024-10-08    [abs](http://arxiv.org/abs/2405.18111v3) [paper-pdf](http://arxiv.org/pdf/2405.18111v3)

**Authors**: Junda Zhu, Lingyong Yan, Haibo Shi, Dawei Yin, Lei Sha

**Abstract**: Large language models (LLMs) are proven to benefit a lot from retrieval-augmented generation (RAG) in alleviating hallucinations confronted with knowledge-intensive questions. RAG adopts information retrieval techniques to inject external knowledge from semantic-relevant documents as input contexts. However, since today's Internet is flooded with numerous noisy and fabricating content, it is inevitable that RAG systems are vulnerable to these noises and prone to respond incorrectly. To this end, we propose to optimize the retrieval-augmented Generator with an Adversarial Tuning Multi-agent system (ATM). The ATM steers the Generator to have a robust perspective of useful documents for question answering with the help of an auxiliary Attacker agent through adversarially tuning the agents for several iterations. After rounds of multi-agent iterative tuning, the Generator can eventually better discriminate useful documents amongst fabrications. The experimental results verify the effectiveness of ATM and we also observe that the Generator can achieve better performance compared to the state-of-the-art baselines.



## **33. TaeBench: Improving Quality of Toxic Adversarial Examples**

cs.CR

**SubmitDate**: 2024-10-08    [abs](http://arxiv.org/abs/2410.05573v1) [paper-pdf](http://arxiv.org/pdf/2410.05573v1)

**Authors**: Xuan Zhu, Dmitriy Bespalov, Liwen You, Ninad Kulkarni, Yanjun Qi

**Abstract**: Toxicity text detectors can be vulnerable to adversarial examples - small perturbations to input text that fool the systems into wrong detection. Existing attack algorithms are time-consuming and often produce invalid or ambiguous adversarial examples, making them less useful for evaluating or improving real-world toxicity content moderators. This paper proposes an annotation pipeline for quality control of generated toxic adversarial examples (TAE). We design model-based automated annotation and human-based quality verification to assess the quality requirements of TAE. Successful TAE should fool a target toxicity model into making benign predictions, be grammatically reasonable, appear natural like human-generated text, and exhibit semantic toxicity. When applying these requirements to more than 20 state-of-the-art (SOTA) TAE attack recipes, we find many invalid samples from a total of 940k raw TAE attack generations. We then utilize the proposed pipeline to filter and curate a high-quality TAE dataset we call TaeBench (of size 264k). Empirically, we demonstrate that TaeBench can effectively transfer-attack SOTA toxicity content moderation models and services. Our experiments also show that TaeBench with adversarial training achieve significant improvements of the robustness of two toxicity detectors.



## **34. Cyber Threats to Canadian Federal Election: Emerging Threats, Assessment, and Mitigation Strategies**

cs.CR

**SubmitDate**: 2024-10-07    [abs](http://arxiv.org/abs/2410.05560v1) [paper-pdf](http://arxiv.org/pdf/2410.05560v1)

**Authors**: Nazmul Islam, Soomin Kim, Mohammad Pirooz, Sasha Shvetsov

**Abstract**: As Canada prepares for the 2025 federal election, ensuring the integrity and security of the electoral process against cyber threats is crucial. Recent foreign interference in elections globally highlight the increasing sophistication of adversaries in exploiting technical and human vulnerabilities. Such vulnerabilities also exist in Canada's electoral system that relies on a complex network of IT systems, vendors, and personnel. To mitigate these vulnerabilities, a threat assessment is crucial to identify emerging threats, develop incident response capabilities, and build public trust and resilience against cyber threats. Therefore, this paper presents a comprehensive national cyber threat assessment, following the NIST Special Publication 800-30 framework, focusing on identifying and mitigating cybersecurity risks to the upcoming 2025 Canadian federal election. The research identifies three major threats: misinformation, disinformation, and malinformation (MDM) campaigns; attacks on critical infrastructure and election support systems; and espionage by malicious actors. Through detailed analysis, the assessment offers insights into the capabilities, intent, and potential impact of these threats. The paper also discusses emerging technologies and their influence on election security and proposes a multi-faceted approach to risk mitigation ahead of the election.



## **35. Kick Bad Guys Out! Conditionally Activated Anomaly Detection in Federated Learning with Zero-Knowledge Proof Verification**

cs.CR

**SubmitDate**: 2024-10-07    [abs](http://arxiv.org/abs/2310.04055v4) [paper-pdf](http://arxiv.org/pdf/2310.04055v4)

**Authors**: Shanshan Han, Wenxuan Wu, Baturalp Buyukates, Weizhao Jin, Qifan Zhang, Yuhang Yao, Salman Avestimehr, Chaoyang He

**Abstract**: Federated Learning (FL) systems are susceptible to adversarial attacks, where malicious clients submit poisoned models to disrupt the convergence or plant backdoors that cause the global model to misclassify some samples. Current defense methods are often impractical for real-world FL systems, as they either rely on unrealistic prior knowledge or cause accuracy loss even in the absence of attacks. Furthermore, these methods lack a protocol for verifying execution, leaving participants uncertain about the correct execution of the mechanism. To address these challenges, we propose a novel anomaly detection strategy that is designed for real-world FL systems. Our approach activates the defense only when potential attacks are detected, and enables the removal of malicious models without affecting the benign ones. Additionally, we incorporate zero-knowledge proofs to ensure the integrity of the proposed defense mechanism. Experimental results demonstrate the effectiveness of our approach in enhancing FL system security against a comprehensive set of adversarial attacks in various ML tasks.



## **36. Aligning LLMs to Be Robust Against Prompt Injection**

cs.CR

Key words: prompt injection defense, LLM security, LLM-integrated  applications. Alignment training makes LLMs robust against even the strongest  prompt injection attacks

**SubmitDate**: 2024-10-07    [abs](http://arxiv.org/abs/2410.05451v1) [paper-pdf](http://arxiv.org/pdf/2410.05451v1)

**Authors**: Sizhe Chen, Arman Zharmagambetov, Saeed Mahloujifar, Kamalika Chaudhuri, Chuan Guo

**Abstract**: Large language models (LLMs) are becoming increasingly prevalent in modern software systems, interfacing between the user and the internet to assist with tasks that require advanced language understanding. To accomplish these tasks, the LLM often uses external data sources such as user documents, web retrieval, results from API calls, etc. This opens up new avenues for attackers to manipulate the LLM via prompt injection. Adversarial prompts can be carefully crafted and injected into external data sources to override the user's intended instruction and instead execute a malicious instruction. Prompt injection attacks constitute a major threat to LLM security, making the design and implementation of practical countermeasures of paramount importance. To this end, we show that alignment can be a powerful tool to make LLMs more robust against prompt injection. Our method -- SecAlign -- first builds an alignment dataset by simulating prompt injection attacks and constructing pairs of desirable and undesirable responses. Then, we apply existing alignment techniques to fine-tune the LLM to be robust against these simulated attacks. Our experiments show that SecAlign robustifies the LLM substantially with a negligible hurt on model utility. Moreover, SecAlign's protection generalizes to strong attacks unseen in training. Specifically, the success rate of state-of-the-art GCG-based prompt injections drops from 56% to 2% in Mistral-7B after our alignment process. Our code is released at https://github.com/facebookresearch/SecAlign



## **37. STOP! Camera Spoofing via the in-Vehicle IP Network**

cs.CR

**SubmitDate**: 2024-10-07    [abs](http://arxiv.org/abs/2410.05417v1) [paper-pdf](http://arxiv.org/pdf/2410.05417v1)

**Authors**: Dror Peri, Avishai Wool

**Abstract**: Autonomous driving and advanced driver assistance systems (ADAS) rely on cameras to control the driving. In many prior approaches an attacker aiming to stop the vehicle had to send messages on the specialized and better-defended CAN bus. We suggest an easier alternative: manipulate the IP-based network communication between the camera and the ADAS logic, inject fake images of stop signs or red lights into the video stream, and let the ADAS stop the car safely. We created an attack tool that successfully exploits the GigE Vision protocol.   Then we analyze two classes of passive anomaly detectors to identify such attacks: protocol-based detectors and video-based detectors. We implemented multiple detectors of both classes and evaluated them on data collected from our test vehicle and also on data from the public BDD corpus. Our results show that such detectors are effective against naive adversaries, but sophisticated adversaries can evade detection.   Finally, we propose a novel class of active defense mechanisms that randomly adjust camera parameters during the video transmission, and verify that the received images obey the requested adjustments. Within this class we focus on a specific implementation, the width-varying defense, which randomly modifies the width of every frame. Beyond its function as an anomaly detector, this defense is also a protective measure against certain attacks: by distorting injected image patches it prevents their recognition by the ADAS logic. We demonstrate the effectiveness of the width-varying defense through theoretical analysis and by an extensive evaluation of several types of attack in a wide range of realistic road driving conditions. The best the attack was able to achieve against this defense was injecting a stop sign for a duration of 0.2 seconds, with a success probability of 0.2%, whereas stopping a vehicle requires about 2.5 seconds.



## **38. Random-Set Neural Networks (RS-NN)**

cs.LG

**SubmitDate**: 2024-10-07    [abs](http://arxiv.org/abs/2307.05772v2) [paper-pdf](http://arxiv.org/pdf/2307.05772v2)

**Authors**: Shireen Kudukkil Manchingal, Muhammad Mubashar, Kaizheng Wang, Keivan Shariatmadar, Fabio Cuzzolin

**Abstract**: Machine learning is increasingly deployed in safety-critical domains where robustness against adversarial attacks is crucial and erroneous predictions could lead to potentially catastrophic consequences. This highlights the need for learning systems to be equipped with the means to determine a model's confidence in its prediction and the epistemic uncertainty associated with it, 'to know when a model does not know'. In this paper, we propose a novel Random-Set Neural Network (RS-NN) for classification. RS-NN predicts belief functions rather than probability vectors over a set of classes using the mathematics of random sets, i.e., distributions over the power set of the sample space. RS-NN encodes the 'epistemic' uncertainty induced in machine learning by limited training sets via the size of the credal sets associated with the predicted belief functions. Our approach outperforms state-of-the-art Bayesian (LB-BNN, BNN-R) and Ensemble (ENN) methods in a classical evaluation setting in terms of performance, uncertainty estimation and out-of-distribution (OoD) detection on several benchmarks (CIFAR-10 vs SVHN/Intel-Image, MNIST vs FMNIST/KMNIST, ImageNet vs ImageNet-O) and scales effectively to large-scale architectures such as WideResNet-28-10, VGG16, Inception V3, EfficientNetB2, and ViT-Base.



## **39. $$\mathbf{L^2\cdot M = C^2}$$ Large Language Models are Covert Channels**

cs.CR

**SubmitDate**: 2024-10-07    [abs](http://arxiv.org/abs/2405.15652v2) [paper-pdf](http://arxiv.org/pdf/2405.15652v2)

**Authors**: Simen Gaure, Stefanos Koffas, Stjepan Picek, Sondre Rønjom

**Abstract**: Large Language Models (LLMs) have gained significant popularity recently. LLMs are susceptible to various attacks but can also improve the security of diverse systems. However, besides enabling more secure systems, how well do open source LLMs behave as covertext distributions to, e.g., facilitate censorship-resistant communication? In this paper, we explore open-source LLM-based covert channels. We empirically measure the security vs. capacity of an open-source LLM model (Llama-7B) to assess its performance as a covert channel. Although our results indicate that such channels are not likely to achieve high practical bitrates, we also show that the chance for an adversary to detect covert communication is low. To ensure our results can be used with the least effort as a general reference, we employ a conceptually simple and concise scheme and only assume public models.



## **40. Jailbreaking Leading Safety-Aligned LLMs with Simple Adaptive Attacks**

cs.CR

Updates in the v3: GPT-4o and Claude 3.5 Sonnet results, improved  writing. Updates in the v2: more models (Llama3, Phi-3, Nemotron-4-340B),  jailbreak artifacts for all attacks are available, evaluation with different  judges (Llama-3-70B and Llama Guard 2), more experiments (convergence plots  over iterations, ablation on the suffix length for random search), examples  of jailbroken generation

**SubmitDate**: 2024-10-07    [abs](http://arxiv.org/abs/2404.02151v3) [paper-pdf](http://arxiv.org/pdf/2404.02151v3)

**Authors**: Maksym Andriushchenko, Francesco Croce, Nicolas Flammarion

**Abstract**: We show that even the most recent safety-aligned LLMs are not robust to simple adaptive jailbreaking attacks. First, we demonstrate how to successfully leverage access to logprobs for jailbreaking: we initially design an adversarial prompt template (sometimes adapted to the target LLM), and then we apply random search on a suffix to maximize a target logprob (e.g., of the token "Sure"), potentially with multiple restarts. In this way, we achieve 100% attack success rate -- according to GPT-4 as a judge -- on Vicuna-13B, Mistral-7B, Phi-3-Mini, Nemotron-4-340B, Llama-2-Chat-7B/13B/70B, Llama-3-Instruct-8B, Gemma-7B, GPT-3.5, GPT-4o, and R2D2 from HarmBench that was adversarially trained against the GCG attack. We also show how to jailbreak all Claude models -- that do not expose logprobs -- via either a transfer or prefilling attack with a 100% success rate. In addition, we show how to use random search on a restricted set of tokens for finding trojan strings in poisoned models -- a task that shares many similarities with jailbreaking -- which is the algorithm that brought us the first place in the SaTML'24 Trojan Detection Competition. The common theme behind these attacks is that adaptivity is crucial: different models are vulnerable to different prompting templates (e.g., R2D2 is very sensitive to in-context learning prompts), some models have unique vulnerabilities based on their APIs (e.g., prefilling for Claude), and in some settings, it is crucial to restrict the token search space based on prior knowledge (e.g., for trojan detection). For reproducibility purposes, we provide the code, logs, and jailbreak artifacts in the JailbreakBench format at https://github.com/tml-epfl/llm-adaptive-attacks.



## **41. Dr. Jekyll and Mr. Hyde: Two Faces of LLMs**

cs.CR

**SubmitDate**: 2024-10-07    [abs](http://arxiv.org/abs/2312.03853v5) [paper-pdf](http://arxiv.org/pdf/2312.03853v5)

**Authors**: Matteo Gioele Collu, Tom Janssen-Groesbeek, Stefanos Koffas, Mauro Conti, Stjepan Picek

**Abstract**: Recently, we have witnessed a rise in the use of Large Language Models (LLMs), especially in applications like chatbots. Safety mechanisms are implemented to prevent improper responses from these chatbots. In this work, we bypass these measures for ChatGPT and Gemini by making them impersonate complex personas with personality characteristics that are not aligned with a truthful assistant. First, we create elaborate biographies of these personas, which we then use in a new session with the same chatbots. Our conversations then follow a role-play style to elicit prohibited responses. Using personas, we show that prohibited responses are provided, making it possible to obtain unauthorized, illegal, or harmful information in both ChatGPT and Gemini. We also introduce several ways of activating such adversarial personas, showing that both chatbots are vulnerable to this attack. With the same principle, we introduce two defenses that push the model to interpret trustworthy personalities and make it more robust against such attacks.



## **42. LOTOS: Layer-wise Orthogonalization for Training Robust Ensembles**

cs.LG

**SubmitDate**: 2024-10-07    [abs](http://arxiv.org/abs/2410.05136v1) [paper-pdf](http://arxiv.org/pdf/2410.05136v1)

**Authors**: Ali Ebrahimpour-Boroojeny, Hari Sundaram, Varun Chandrasekaran

**Abstract**: Transferability of adversarial examples is a well-known property that endangers all classification models, even those that are only accessible through black-box queries. Prior work has shown that an ensemble of models is more resilient to transferability: the probability that an adversarial example is effective against most models of the ensemble is low. Thus, most ongoing research focuses on improving ensemble diversity. Another line of prior work has shown that Lipschitz continuity of the models can make models more robust since it limits how a model's output changes with small input perturbations. In this paper, we study the effect of Lipschitz continuity on transferability rates. We show that although a lower Lipschitz constant increases the robustness of a single model, it is not as beneficial in training robust ensembles as it increases the transferability rate of adversarial examples across models in the ensemble. Therefore, we introduce LOTOS, a new training paradigm for ensembles, which counteracts this adverse effect. It does so by promoting orthogonality among the top-$k$ sub-spaces of the transformations of the corresponding affine layers of any pair of models in the ensemble. We theoretically show that $k$ does not need to be large for convolutional layers, which makes the computational overhead negligible. Through various experiments, we show LOTOS increases the robust accuracy of ensembles of ResNet-18 models by $6$ percentage points (p.p) against black-box attacks on CIFAR-10. It is also capable of combining with the robustness of prior state-of-the-art methods for training robust ensembles to enhance their robust accuracy by $10.7$ p.p.



## **43. A test suite of prompt injection attacks for LLM-based machine translation**

cs.CL

**SubmitDate**: 2024-10-07    [abs](http://arxiv.org/abs/2410.05047v1) [paper-pdf](http://arxiv.org/pdf/2410.05047v1)

**Authors**: Antonio Valerio Miceli-Barone, Zhifan Sun

**Abstract**: LLM-based NLP systems typically work by embedding their input data into prompt templates which contain instructions and/or in-context examples, creating queries which are submitted to a LLM, and then parsing the LLM response in order to generate the system outputs. Prompt Injection Attacks (PIAs) are a type of subversion of these systems where a malicious user crafts special inputs which interfere with the prompt templates, causing the LLM to respond in ways unintended by the system designer.   Recently, Sun and Miceli-Barone proposed a class of PIAs against LLM-based machine translation. Specifically, the task is to translate questions from the TruthfulQA test suite, where an adversarial prompt is prepended to the questions, instructing the system to ignore the translation instruction and answer the questions instead.   In this test suite, we extend this approach to all the language pairs of the WMT 2024 General Machine Translation task. Moreover, we include additional attack formats in addition to the one originally studied.



## **44. Collaboration! Towards Robust Neural Methods for Routing Problems**

cs.AI

Accepted at NeurIPS 2024

**SubmitDate**: 2024-10-07    [abs](http://arxiv.org/abs/2410.04968v1) [paper-pdf](http://arxiv.org/pdf/2410.04968v1)

**Authors**: Jianan Zhou, Yaoxin Wu, Zhiguang Cao, Wen Song, Jie Zhang, Zhiqi Shen

**Abstract**: Despite enjoying desirable efficiency and reduced reliance on domain expertise, existing neural methods for vehicle routing problems (VRPs) suffer from severe robustness issues -- their performance significantly deteriorates on clean instances with crafted perturbations. To enhance robustness, we propose an ensemble-based Collaborative Neural Framework (CNF) w.r.t. the defense of neural VRP methods, which is crucial yet underexplored in the literature. Given a neural VRP method, we adversarially train multiple models in a collaborative manner to synergistically promote robustness against attacks, while boosting standard generalization on clean instances. A neural router is designed to adeptly distribute training instances among models, enhancing overall load balancing and collaborative efficacy. Extensive experiments verify the effectiveness and versatility of CNF in defending against various attacks across different neural VRP methods. Notably, our approach also achieves impressive out-of-distribution generalization on benchmark instances.



## **45. Reconstruct Your Previous Conversations! Comprehensively Investigating Privacy Leakage Risks in Conversations with GPT Models**

cs.CR

Accepted in EMNLP 2024. 14 pages, 10 figures

**SubmitDate**: 2024-10-07    [abs](http://arxiv.org/abs/2402.02987v2) [paper-pdf](http://arxiv.org/pdf/2402.02987v2)

**Authors**: Junjie Chu, Zeyang Sha, Michael Backes, Yang Zhang

**Abstract**: Significant advancements have recently been made in large language models represented by GPT models. Users frequently have multi-round private conversations with cloud-hosted GPT models for task optimization. Yet, this operational paradigm introduces additional attack surfaces, particularly in custom GPTs and hijacked chat sessions. In this paper, we introduce a straightforward yet potent Conversation Reconstruction Attack. This attack targets the contents of previous conversations between GPT models and benign users, i.e., the benign users' input contents during their interaction with GPT models. The adversary could induce GPT models to leak such contents by querying them with designed malicious prompts. Our comprehensive examination of privacy risks during the interactions with GPT models under this attack reveals GPT-4's considerable resilience. We present two advanced attacks targeting improved reconstruction of past conversations, demonstrating significant privacy leakage across all models under these advanced techniques. Evaluating various defense mechanisms, we find them ineffective against these attacks. Our findings highlight the ease with which privacy can be compromised in interactions with GPT models, urging the community to safeguard against potential abuses of these models' capabilities.



## **46. ResTNet: Defense against Adversarial Policies via Transformer in Computer Go**

cs.LG

**SubmitDate**: 2024-10-07    [abs](http://arxiv.org/abs/2410.05347v1) [paper-pdf](http://arxiv.org/pdf/2410.05347v1)

**Authors**: Tai-Lin Wu, Ti-Rong Wu, Chung-Chin Shih, Yan-Ru Ju, I-Chen Wu

**Abstract**: Although AlphaZero has achieved superhuman levels in Go, recent research has highlighted its vulnerability in particular situations requiring a more comprehensive understanding of the entire board. To address this challenge, this paper introduces ResTNet, a network that interleaves residual networks and Transformer. Our empirical experiments demonstrate several advantages of using ResTNet. First, it not only improves playing strength but also enhances the ability of global information. Second, it defends against an adversary Go program, called cyclic-adversary, tailor-made for attacking AlphaZero algorithms, significantly reducing the average probability of being attacked rate from 70.44% to 23.91%. Third, it improves the accuracy from 59.15% to 80.01% in correctly recognizing ladder patterns, which are one of the challenging patterns for Go AIs. Finally, ResTNet offers a potential explanation of the decision-making process and can also be applied to other games like Hex. To the best of our knowledge, ResTNet is the first to integrate residual networks and Transformer in the context of AlphaZero for board games, suggesting a promising direction for enhancing AlphaZero's global understanding.



## **47. Patch is Enough: Naturalistic Adversarial Patch against Vision-Language Pre-training Models**

cs.CV

accepted by Visual Intelligence

**SubmitDate**: 2024-10-07    [abs](http://arxiv.org/abs/2410.04884v1) [paper-pdf](http://arxiv.org/pdf/2410.04884v1)

**Authors**: Dehong Kong, Siyuan Liang, Xiaopeng Zhu, Yuansheng Zhong, Wenqi Ren

**Abstract**: Visual language pre-training (VLP) models have demonstrated significant success across various domains, yet they remain vulnerable to adversarial attacks. Addressing these adversarial vulnerabilities is crucial for enhancing security in multimodal learning. Traditionally, adversarial methods targeting VLP models involve simultaneously perturbing images and text. However, this approach faces notable challenges: first, adversarial perturbations often fail to translate effectively into real-world scenarios; second, direct modifications to the text are conspicuously visible. To overcome these limitations, we propose a novel strategy that exclusively employs image patches for attacks, thus preserving the integrity of the original text. Our method leverages prior knowledge from diffusion models to enhance the authenticity and naturalness of the perturbations. Moreover, to optimize patch placement and improve the efficacy of our attacks, we utilize the cross-attention mechanism, which encapsulates intermodal interactions by generating attention maps to guide strategic patch placements. Comprehensive experiments conducted in a white-box setting for image-to-text scenarios reveal that our proposed method significantly outperforms existing techniques, achieving a 100% attack success rate. Additionally, it demonstrates commendable performance in transfer tasks involving text-to-image configurations.



## **48. AnyAttack: Towards Large-scale Self-supervised Generation of Targeted Adversarial Examples for Vision-Language Models**

cs.LG

**SubmitDate**: 2024-10-07    [abs](http://arxiv.org/abs/2410.05346v1) [paper-pdf](http://arxiv.org/pdf/2410.05346v1)

**Authors**: Jiaming Zhang, Junhong Ye, Xingjun Ma, Yige Li, Yunfan Yang, Jitao Sang, Dit-Yan Yeung

**Abstract**: Due to their multimodal capabilities, Vision-Language Models (VLMs) have found numerous impactful applications in real-world scenarios. However, recent studies have revealed that VLMs are vulnerable to image-based adversarial attacks, particularly targeted adversarial images that manipulate the model to generate harmful content specified by the adversary. Current attack methods rely on predefined target labels to create targeted adversarial attacks, which limits their scalability and applicability for large-scale robustness evaluations. In this paper, we propose AnyAttack, a self-supervised framework that generates targeted adversarial images for VLMs without label supervision, allowing any image to serve as a target for the attack. To address the limitation of existing methods that require label supervision, we introduce a contrastive loss that trains a generator on a large-scale unlabeled image dataset, LAION-400M dataset, for generating targeted adversarial noise. This large-scale pre-training endows our method with powerful transferability across a wide range of VLMs. Extensive experiments on five mainstream open-source VLMs (CLIP, BLIP, BLIP2, InstructBLIP, and MiniGPT-4) across three multimodal tasks (image-text retrieval, multimodal classification, and image captioning) demonstrate the effectiveness of our attack. Additionally, we successfully transfer AnyAttack to multiple commercial VLMs, including Google's Gemini, Claude's Sonnet, and Microsoft's Copilot. These results reveal an unprecedented risk to VLMs, highlighting the need for effective countermeasures.



## **49. A Differentially Private Energy Trading Mechanism Approaching Social Optimum**

cs.GT

11 pages, 8 figures

**SubmitDate**: 2024-10-07    [abs](http://arxiv.org/abs/2410.04787v1) [paper-pdf](http://arxiv.org/pdf/2410.04787v1)

**Authors**: Yuji Cao, Yue Chen

**Abstract**: This paper proposes a differentially private energy trading mechanism for prosumers in peer-to-peer (P2P) markets, offering provable privacy guarantees while approaching the Nash equilibrium with nearly socially optimal efficiency. We first model the P2P energy trading as a (generalized) Nash game and prove the vulnerability of traditional distributed algorithms to privacy attacks through an adversarial inference model. To address this challenge, we develop a privacy-preserving Nash equilibrium seeking algorithm incorporating carefully calibrated Laplacian noise. We prove that the proposed algorithm achieves $\epsilon$-differential privacy while converging in expectation to the Nash equilibrium with a suitable stepsize. Numerical experiments are conducted to evaluate the algorithm's robustness against privacy attacks, convergence behavior, and optimality compared to the non-private solution. Results demonstrate that our mechanism effectively protects prosumers' sensitive information while maintaining near-optimal market outcomes, offering a practical approach for privacy-preserving coordination in P2P markets.



## **50. Double Oracle Neural Architecture Search for Game Theoretic Deep Learning Models**

cs.LG

**SubmitDate**: 2024-10-07    [abs](http://arxiv.org/abs/2410.04764v1) [paper-pdf](http://arxiv.org/pdf/2410.04764v1)

**Authors**: Aye Phyu Phyu Aung, Xinrun Wang, Ruiyu Wang, Hau Chan, Bo An, Xiaoli Li, J. Senthilnath

**Abstract**: In this paper, we propose a new approach to train deep learning models using game theory concepts including Generative Adversarial Networks (GANs) and Adversarial Training (AT) where we deploy a double-oracle framework using best response oracles. GAN is essentially a two-player zero-sum game between the generator and the discriminator. The same concept can be applied to AT with attacker and classifier as players. Training these models is challenging as a pure Nash equilibrium may not exist and even finding the mixed Nash equilibrium is difficult as training algorithms for both GAN and AT have a large-scale strategy space. Extending our preliminary model DO-GAN, we propose the methods to apply the double oracle framework concept to Adversarial Neural Architecture Search (NAS for GAN) and Adversarial Training (NAS for AT) algorithms. We first generalize the players' strategies as the trained models of generator and discriminator from the best response oracles. We then compute the meta-strategies using a linear program. For scalability of the framework where multiple network models of best responses are stored in the memory, we prune the weakly-dominated players' strategies to keep the oracles from becoming intractable. Finally, we conduct experiments on MNIST, CIFAR-10 and TinyImageNet for DONAS-GAN. We also evaluate the robustness under FGSM and PGD attacks on CIFAR-10, SVHN and TinyImageNet for DONAS-AT. We show that all our variants have significant improvements in both subjective qualitative evaluation and quantitative metrics, compared with their respective base architectures.



