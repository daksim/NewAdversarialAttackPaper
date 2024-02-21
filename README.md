# Latest Adversarial Attack Papers
**update at 2024-02-21 10:59:58**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. Unfolding Local Growth Rate Estimates for (Almost) Perfect Adversarial Detection**

cs.CV

accepted at VISAPP23

**SubmitDate**: 2024-02-20    [abs](http://arxiv.org/abs/2212.06776v4) [paper-pdf](http://arxiv.org/pdf/2212.06776v4)

**Authors**: Peter Lorenz, Margret Keuper, Janis Keuper

**Abstract**: Convolutional neural networks (CNN) define the state-of-the-art solution on many perceptual tasks. However, current CNN approaches largely remain vulnerable against adversarial perturbations of the input that have been crafted specifically to fool the system while being quasi-imperceptible to the human eye. In recent years, various approaches have been proposed to defend CNNs against such attacks, for example by model hardening or by adding explicit defence mechanisms. Thereby, a small "detector" is included in the network and trained on the binary classification task of distinguishing genuine data from data containing adversarial perturbations. In this work, we propose a simple and light-weight detector, which leverages recent findings on the relation between networks' local intrinsic dimensionality (LID) and adversarial attacks. Based on a re-interpretation of the LID measure and several simple adaptations, we surpass the state-of-the-art on adversarial detection by a significant margin and reach almost perfect results in terms of F1-score for several networks and datasets. Sources available at: https://github.com/adverML/multiLID



## **2. Is RobustBench/AutoAttack a suitable Benchmark for Adversarial Robustness?**

cs.CV

AAAI-22 AdvML Workshop

**SubmitDate**: 2024-02-20    [abs](http://arxiv.org/abs/2112.01601v4) [paper-pdf](http://arxiv.org/pdf/2112.01601v4)

**Authors**: Peter Lorenz, Dominik Strassel, Margret Keuper, Janis Keuper

**Abstract**: Recently, RobustBench (Croce et al. 2020) has become a widely recognized benchmark for the adversarial robustness of image classification networks. In its most commonly reported sub-task, RobustBench evaluates and ranks the adversarial robustness of trained neural networks on CIFAR10 under AutoAttack (Croce and Hein 2020b) with l-inf perturbations limited to eps = 8/255. With leading scores of the currently best performing models of around 60% of the baseline, it is fair to characterize this benchmark to be quite challenging. Despite its general acceptance in recent literature, we aim to foster discussion about the suitability of RobustBench as a key indicator for robustness which could be generalized to practical applications. Our line of argumentation against this is two-fold and supported by excessive experiments presented in this paper: We argue that I) the alternation of data by AutoAttack with l-inf, eps = 8/255 is unrealistically strong, resulting in close to perfect detection rates of adversarial samples even by simple detection algorithms and human observers. We also show that other attack methods are much harder to detect while achieving similar success rates. II) That results on low-resolution data sets like CIFAR10 do not generalize well to higher resolution images as gradient-based attacks appear to become even more detectable with increasing resolutions.



## **3. Detecting AutoAttack Perturbations in the Frequency Domain**

cs.CV

accepted at ICML 2021 workshop for robustness

**SubmitDate**: 2024-02-20    [abs](http://arxiv.org/abs/2111.08785v3) [paper-pdf](http://arxiv.org/pdf/2111.08785v3)

**Authors**: Peter Lorenz, Paula Harder, Dominik Strassel, Margret Keuper, Janis Keuper

**Abstract**: Recently, adversarial attacks on image classification networks by the AutoAttack (Croce and Hein, 2020b) framework have drawn a lot of attention. While AutoAttack has shown a very high attack success rate, most defense approaches are focusing on network hardening and robustness enhancements, like adversarial training. This way, the currently best-reported method can withstand about 66% of adversarial examples on CIFAR10. In this paper, we investigate the spatial and frequency domain properties of AutoAttack and propose an alternative defense. Instead of hardening a network, we detect adversarial attacks during inference, rejecting manipulated inputs. Based on a rather simple and fast analysis in the frequency domain, we introduce two different detection algorithms. First, a black box detector that only operates on the input images and achieves a detection accuracy of 100% on the AutoAttack CIFAR10 benchmark and 99.3% on ImageNet, for epsilon = 8/255 in both cases. Second, a whitebox detector using an analysis of CNN feature maps, leading to a detection rate of also 100% and 98.7% on the same benchmarks.



## **4. AICAttack: Adversarial Image Captioning Attack with Attention-Based Optimization**

cs.CV

**SubmitDate**: 2024-02-20    [abs](http://arxiv.org/abs/2402.11940v2) [paper-pdf](http://arxiv.org/pdf/2402.11940v2)

**Authors**: Jiyao Li, Mingze Ni, Yifei Dong, Tianqing Zhu, Wei Liu

**Abstract**: Recent advances in deep learning research have shown remarkable achievements across many tasks in computer vision (CV) and natural language processing (NLP). At the intersection of CV and NLP is the problem of image captioning, where the related models' robustness against adversarial attacks has not been well studied. In this paper, we present a novel adversarial attack strategy, which we call AICAttack (Attention-based Image Captioning Attack), designed to attack image captioning models through subtle perturbations on images. Operating within a black-box attack scenario, our algorithm requires no access to the target model's architecture, parameters, or gradient information. We introduce an attention-based candidate selection mechanism that identifies the optimal pixels to attack, followed by Differential Evolution (DE) for perturbing pixels' RGB values. We demonstrate AICAttack's effectiveness through extensive experiments on benchmark datasets with multiple victim models. The experimental results demonstrate that our method surpasses current leading-edge techniques by effectively distributing the alignment and semantics of words in the output.



## **5. Bounding Reconstruction Attack Success of Adversaries Without Data Priors**

cs.LG

**SubmitDate**: 2024-02-20    [abs](http://arxiv.org/abs/2402.12861v1) [paper-pdf](http://arxiv.org/pdf/2402.12861v1)

**Authors**: Alexander Ziller, Anneliese Riess, Kristian Schwethelm, Tamara T. Mueller, Daniel Rueckert, Georgios Kaissis

**Abstract**: Reconstruction attacks on machine learning (ML) models pose a strong risk of leakage of sensitive data. In specific contexts, an adversary can (almost) perfectly reconstruct training data samples from a trained model using the model's gradients. When training ML models with differential privacy (DP), formal upper bounds on the success of such reconstruction attacks can be provided. So far, these bounds have been formulated under worst-case assumptions that might not hold high realistic practicality. In this work, we provide formal upper bounds on reconstruction success under realistic adversarial settings against ML models trained with DP and support these bounds with empirical results. With this, we show that in realistic scenarios, (a) the expected reconstruction success can be bounded appropriately in different contexts and by different metrics, which (b) allows for a more educated choice of a privacy parameter.



## **6. SWAP: Sparse Entropic Wasserstein Regression for Robust Network Pruning**

cs.AI

Published as a conference paper at ICLR 2024

**SubmitDate**: 2024-02-20    [abs](http://arxiv.org/abs/2310.04918v4) [paper-pdf](http://arxiv.org/pdf/2310.04918v4)

**Authors**: Lei You, Hei Victor Cheng

**Abstract**: This study addresses the challenge of inaccurate gradients in computing the empirical Fisher Information Matrix during neural network pruning. We introduce SWAP, a formulation of Entropic Wasserstein regression (EWR) for pruning, capitalizing on the geometric properties of the optimal transport problem. The ``swap'' of the commonly used linear regression with the EWR in optimization is analytically demonstrated to offer noise mitigation effects by incorporating neighborhood interpolation across data points with only marginal additional computational cost. The unique strength of SWAP is its intrinsic ability to balance noise reduction and covariance information preservation effectively. Extensive experiments performed on various networks and datasets show comparable performance of SWAP with state-of-the-art (SoTA) network pruning algorithms. Our proposed method outperforms the SoTA when the network size or the target sparsity is large, the gain is even larger with the existence of noisy gradients, possibly from noisy data, analog memory, or adversarial attacks. Notably, our proposed method achieves a gain of 6% improvement in accuracy and 8% improvement in testing loss for MobileNetV1 with less than one-fourth of the network parameters remaining.



## **7. Understanding and Mitigating the Threat of Vec2Text to Dense Retrieval Systems**

cs.IR

**SubmitDate**: 2024-02-20    [abs](http://arxiv.org/abs/2402.12784v1) [paper-pdf](http://arxiv.org/pdf/2402.12784v1)

**Authors**: Shengyao Zhuang, Bevan Koopman, Xiaoran Chu, Guido Zuccon

**Abstract**: The introduction of Vec2Text, a technique for inverting text embeddings, has raised serious privacy concerns within dense retrieval systems utilizing text embeddings, including those provided by OpenAI and Cohere. This threat comes from the ability for a malicious attacker with access to text embeddings to reconstruct the original text.   In this paper, we investigate various aspects of embedding models that could influence the recoverability of text using Vec2Text. Our exploration involves factors such as distance metrics, pooling functions, bottleneck pre-training, training with noise addition, embedding quantization, and embedding dimensions -- aspects not previously addressed in the original Vec2Text paper. Through a thorough analysis of these factors, our aim is to gain a deeper understanding of the critical elements impacting the trade-offs between text recoverability and retrieval effectiveness in dense retrieval systems. This analysis provides valuable insights for practitioners involved in designing privacy-aware dense retrieval systems. Additionally, we propose a straightforward fix for embedding transformation that ensures equal ranking effectiveness while mitigating the risk of text recoverability.   Furthermore, we extend the application of Vec2Text to the separate task of corpus poisoning, where, theoretically, Vec2Text presents a more potent threat compared to previous attack methods. Notably, Vec2Text does not require access to the dense retriever's model parameters and can efficiently generate numerous adversarial passages.   In summary, this study highlights the potential threat posed by Vec2Text to existing dense retrieval systems, while also presenting effective methods to patch and strengthen such systems against such risks.



## **8. Revisiting the Information Capacity of Neural Network Watermarks: Upper Bound Estimation and Beyond**

cs.CR

Accepted by AAAI 2024

**SubmitDate**: 2024-02-20    [abs](http://arxiv.org/abs/2402.12720v1) [paper-pdf](http://arxiv.org/pdf/2402.12720v1)

**Authors**: Fangqi Li, Haodong Zhao, Wei Du, Shilin Wang

**Abstract**: To trace the copyright of deep neural networks, an owner can embed its identity information into its model as a watermark. The capacity of the watermark quantify the maximal volume of information that can be verified from the watermarked model. Current studies on capacity focus on the ownership verification accuracy under ordinary removal attacks and fail to capture the relationship between robustness and fidelity. This paper studies the capacity of deep neural network watermarks from an information theoretical perspective. We propose a new definition of deep neural network watermark capacity analogous to channel capacity, analyze its properties, and design an algorithm that yields a tight estimation of its upper bound under adversarial overwriting. We also propose a universal non-invasive method to secure the transmission of the identity message beyond capacity by multiple rounds of ownership verification. Our observations provide evidence for neural network owners and defenders that are curious about the tradeoff between the integrity of their ownership and the performance degradation of their products.



## **9. Beyond Worst-case Attacks: Robust RL with Adaptive Defense via Non-dominated Policies**

cs.LG

International Conference on Learning Representations (ICLR) 2024,  spotlight

**SubmitDate**: 2024-02-20    [abs](http://arxiv.org/abs/2402.12673v1) [paper-pdf](http://arxiv.org/pdf/2402.12673v1)

**Authors**: Xiangyu Liu, Chenghao Deng, Yanchao Sun, Yongyuan Liang, Furong Huang

**Abstract**: In light of the burgeoning success of reinforcement learning (RL) in diverse real-world applications, considerable focus has been directed towards ensuring RL policies are robust to adversarial attacks during test time. Current approaches largely revolve around solving a minimax problem to prepare for potential worst-case scenarios. While effective against strong attacks, these methods often compromise performance in the absence of attacks or the presence of only weak attacks. To address this, we study policy robustness under the well-accepted state-adversarial attack model, extending our focus beyond only worst-case attacks. We first formalize this task at test time as a regret minimization problem and establish its intrinsic hardness in achieving sublinear regret when the baseline policy is from a general continuous policy class, $\Pi$. This finding prompts us to \textit{refine} the baseline policy class $\Pi$ prior to test time, aiming for efficient adaptation within a finite policy class $\Tilde{\Pi}$, which can resort to an adversarial bandit subroutine. In light of the importance of a small, finite $\Tilde{\Pi}$, we propose a novel training-time algorithm to iteratively discover \textit{non-dominated policies}, forming a near-optimal and minimal $\Tilde{\Pi}$, thereby ensuring both robustness and test-time efficiency. Empirical validation on the Mujoco corroborates the superiority of our approach in terms of natural and robust performance, as well as adaptability to various attack scenarios.



## **10. Citadel: Enclaves with Microarchitectural Isolation and Secure Shared Memory on a Speculative Out-of-Order Processor**

cs.CR

**SubmitDate**: 2024-02-20    [abs](http://arxiv.org/abs/2306.14882v3) [paper-pdf](http://arxiv.org/pdf/2306.14882v3)

**Authors**: Jules Drean, Miguel Gomez-Garcia, Fisher Jepsen, Thomas Bourgeat, Srinivas Devadas

**Abstract**: Enclaves or Trusted Execution Environments are trusted-hardware primitives that make it possible to isolate and protect a sensitive program from an untrusted operating system. Unfortunately, almost all existing enclave platforms are vulnerable to microarchitectural side channels and transient execution attacks, and the one academic proposal that is not does not allow programs to interact with the outside world. We present Citadel, to our knowledge, the first enclave platform with microarchitectural isolation to run realistic secure programs on a speculative out-of-order multicore processor. We show how to leverage hardware/software co-design to enable shared memory between an enclave and an untrusted operating system while preventing speculative transmitters between the enclave and a potential adversary. We then evaluate our secure baseline and present further mechanisms to achieve reasonable performance for out-of-the-box programs. Our multicore processor runs on an FPGA and boots untrusted Linux from which users can securely launch and interact with enclaves. To demonstrate our platform capabilities, we run a private inference enclave that embed a small neural network trained on MNIST. A remote user can remotely attest the enclave integrity, perform key exchange and send encrypted input for secure evaluation. We open-source our end-to-end hardware and software infrastructure, hoping to spark more research and bridge the gap between conceptual proposals and FPGA prototypes.



## **11. Emulated Disalignment: Safety Alignment for Large Language Models May Backfire!**

cs.CL

**SubmitDate**: 2024-02-19    [abs](http://arxiv.org/abs/2402.12343v1) [paper-pdf](http://arxiv.org/pdf/2402.12343v1)

**Authors**: Zhanhui Zhou, Jie Liu, Zhichen Dong, Jiaheng Liu, Chao Yang, Wanli Ouyang, Yu Qiao

**Abstract**: Large language models (LLMs) need to undergo safety alignment to ensure safe conversations with humans. However, in this work, we introduce an inference-time attack framework, demonstrating that safety alignment can also unintentionally facilitate harmful outcomes under adversarial manipulation. This framework, named Emulated Disalignment (ED), adversely combines a pair of open-source pre-trained and safety-aligned language models in the output space to produce a harmful language model without any training. Our experiments with ED across three datasets and four model families (Llama-1, Llama-2, Mistral, and Alpaca) show that ED doubles the harmfulness of pre-trained models and outperforms strong baselines, achieving the highest harmful rate in 43 out of 48 evaluation subsets by a large margin. Crucially, our findings highlight the importance of reevaluating the practice of open-sourcing language models even after safety alignment.



## **12. An Adversarial Approach to Evaluating the Robustness of Event Identification Models**

eess.SY

**SubmitDate**: 2024-02-19    [abs](http://arxiv.org/abs/2402.12338v1) [paper-pdf](http://arxiv.org/pdf/2402.12338v1)

**Authors**: Obai Bahwal, Oliver Kosut, Lalitha Sankar

**Abstract**: Intelligent machine learning approaches are finding active use for event detection and identification that allow real-time situational awareness. Yet, such machine learning algorithms have been shown to be susceptible to adversarial attacks on the incoming telemetry data. This paper considers a physics-based modal decomposition method to extract features for event classification and focuses on interpretable classifiers including logistic regression and gradient boosting to distinguish two types of events: load loss and generation loss. The resulting classifiers are then tested against an adversarial algorithm to evaluate their robustness. The adversarial attack is tested in two settings: the white box setting, wherein the attacker knows exactly the classification model; and the gray box setting, wherein the attacker has access to historical data from the same network as was used to train the classifier, but does not know the classification model. Thorough experiments on the synthetic South Carolina 500-bus system highlight that a relatively simpler model such as logistic regression is more susceptible to adversarial attacks than gradient boosting.



## **13. Robust CLIP: Unsupervised Adversarial Fine-Tuning of Vision Embeddings for Robust Large Vision-Language Models**

cs.LG

**SubmitDate**: 2024-02-19    [abs](http://arxiv.org/abs/2402.12336v1) [paper-pdf](http://arxiv.org/pdf/2402.12336v1)

**Authors**: Christian Schlarmann, Naman Deep Singh, Francesco Croce, Matthias Hein

**Abstract**: Multi-modal foundation models like OpenFlamingo, LLaVA, and GPT-4 are increasingly used for various real-world tasks. Prior work has shown that these models are highly vulnerable to adversarial attacks on the vision modality. These attacks can be leveraged to spread fake information or defraud users, and thus pose a significant risk, which makes the robustness of large multi-modal foundation models a pressing problem. The CLIP model, or one of its variants, is used as a frozen vision encoder in many vision-language models (VLMs), e.g. LLaVA and OpenFlamingo. We propose an unsupervised adversarial fine-tuning scheme to obtain a robust CLIP vision encoder, which yields robustness on all vision down-stream tasks (VLMs, zero-shot classification) that rely on CLIP. In particular, we show that stealth-attacks on users of VLMs by a malicious third party providing manipulated images are no longer possible once one replaces the original CLIP model with our robust one. No retraining or fine-tuning of the VLM is required. The code and robust models are available at https://github.com/chs20/RobustVLM



## **14. Query-Based Adversarial Prompt Generation**

cs.CL

**SubmitDate**: 2024-02-19    [abs](http://arxiv.org/abs/2402.12329v1) [paper-pdf](http://arxiv.org/pdf/2402.12329v1)

**Authors**: Jonathan Hayase, Ema Borevkovic, Nicholas Carlini, Florian Tramèr, Milad Nasr

**Abstract**: Recent work has shown it is possible to construct adversarial examples that cause an aligned language model to emit harmful strings or perform harmful behavior. Existing attacks work either in the white-box setting (with full access to the model weights), or through transferability: the phenomenon that adversarial examples crafted on one model often remain effective on other models. We improve on prior work with a query-based attack that leverages API access to a remote language model to construct adversarial examples that cause the model to emit harmful strings with (much) higher probability than with transfer-only attacks. We validate our attack on GPT-3.5 and OpenAI's safety classifier; we can cause GPT-3.5 to emit harmful strings that current transfer attacks fail at, and we can evade the safety classifier with nearly 100% probability.



## **15. Attacks on Node Attributes in Graph Neural Networks**

cs.SI

Accepted to AAAI 2024 AICS workshop

**SubmitDate**: 2024-02-19    [abs](http://arxiv.org/abs/2402.12426v1) [paper-pdf](http://arxiv.org/pdf/2402.12426v1)

**Authors**: Ying Xu, Michael Lanier, Anindya Sarkar, Yevgeniy Vorobeychik

**Abstract**: Graphs are commonly used to model complex networks prevalent in modern social media and literacy applications. Our research investigates the vulnerability of these graphs through the application of feature based adversarial attacks, focusing on both decision-time attacks and poisoning attacks. In contrast to state-of-the-art models like Net Attack and Meta Attack, which target node attributes and graph structure, our study specifically targets node attributes. For our analysis, we utilized the text dataset Hellaswag and graph datasets Cora and CiteSeer, providing a diverse basis for evaluation. Our findings indicate that decision-time attacks using Projected Gradient Descent (PGD) are more potent compared to poisoning attacks that employ Mean Node Embeddings and Graph Contrastive Learning strategies. This provides insights for graph data security, pinpointing where graph-based models are most vulnerable and thereby informing the development of stronger defense mechanisms against such attacks.



## **16. Can AI-Generated Text be Reliably Detected?**

cs.CL

**SubmitDate**: 2024-02-19    [abs](http://arxiv.org/abs/2303.11156v3) [paper-pdf](http://arxiv.org/pdf/2303.11156v3)

**Authors**: Vinu Sankar Sadasivan, Aounon Kumar, Sriram Balasubramanian, Wenxiao Wang, Soheil Feizi

**Abstract**: The unregulated use of LLMs can potentially lead to malicious consequences such as plagiarism, generating fake news, spamming, etc. Therefore, reliable detection of AI-generated text can be critical to ensure the responsible use of LLMs. Recent works attempt to tackle this problem either using certain model signatures present in the generated text outputs or by applying watermarking techniques that imprint specific patterns onto them. In this paper, we show that these detectors are not reliable in practical scenarios. In particular, we develop a recursive paraphrasing attack to apply on AI text, which can break a whole range of detectors, including the ones using the watermarking schemes as well as neural network-based detectors, zero-shot classifiers, and retrieval-based detectors. Our experiments include passages around 300 tokens in length, showing the sensitivity of the detectors even in the case of relatively long passages. We also observe that our recursive paraphrasing only degrades text quality slightly, measured via human studies, and metrics such as perplexity scores and accuracy on text benchmarks. Additionally, we show that even LLMs protected by watermarking schemes can be vulnerable against spoofing attacks aimed to mislead detectors to classify human-written text as AI-generated, potentially causing reputational damages to the developers. In particular, we show that an adversary can infer hidden AI text signatures of the LLM outputs without having white-box access to the detection method. Finally, we provide a theoretical connection between the AUROC of the best possible detector and the Total Variation distance between human and AI text distributions that can be used to study the fundamental hardness of the reliable detection problem for advanced language models. Our code is publicly available at https://github.com/vinusankars/Reliability-of-AI-text-detectors.



## **17. On the Byzantine-Resilience of Distillation-Based Federated Learning**

cs.LG

**SubmitDate**: 2024-02-19    [abs](http://arxiv.org/abs/2402.12265v1) [paper-pdf](http://arxiv.org/pdf/2402.12265v1)

**Authors**: Christophe Roux, Max Zimmer, Sebastian Pokutta

**Abstract**: Federated Learning (FL) algorithms using Knowledge Distillation (KD) have received increasing attention due to their favorable properties with respect to privacy, non-i.i.d. data and communication cost. These methods depart from transmitting model parameters and, instead, communicate information about a learning task by sharing predictions on a public dataset. In this work, we study the performance of such approaches in the byzantine setting, where a subset of the clients act in an adversarial manner aiming to disrupt the learning process. We show that KD-based FL algorithms are remarkably resilient and analyze how byzantine clients can influence the learning process compared to Federated Averaging. Based on these insights, we introduce two new byzantine attacks and demonstrate that they are effective against prior byzantine-resilient methods. Additionally, we propose FilterExp, a novel method designed to enhance the byzantine resilience of KD-based FL algorithms and demonstrate its efficacy. Finally, we provide a general method to make attacks harder to detect, improving their effectiveness.



## **18. Amplifying Training Data Exposure through Fine-Tuning with Pseudo-Labeled Memberships**

cs.CL

20 pages, 6 figures, 15 tables

**SubmitDate**: 2024-02-19    [abs](http://arxiv.org/abs/2402.12189v1) [paper-pdf](http://arxiv.org/pdf/2402.12189v1)

**Authors**: Myung Gyo Oh, Hong Eun Ahn, Leo Hyun Park, Taekyoung Kwon

**Abstract**: Neural language models (LMs) are vulnerable to training data extraction attacks due to data memorization. This paper introduces a novel attack scenario wherein an attacker adversarially fine-tunes pre-trained LMs to amplify the exposure of the original training data. This strategy differs from prior studies by aiming to intensify the LM's retention of its pre-training dataset. To achieve this, the attacker needs to collect generated texts that are closely aligned with the pre-training data. However, without knowledge of the actual dataset, quantifying the amount of pre-training data within generated texts is challenging. To address this, we propose the use of pseudo-labels for these generated texts, leveraging membership approximations indicated by machine-generated probabilities from the target LM. We subsequently fine-tune the LM to favor generations with higher likelihoods of originating from the pre-training data, based on their membership probabilities. Our empirical findings indicate a remarkable outcome: LMs with over 1B parameters exhibit a four to eight-fold increase in training data exposure. We discuss potential mitigations and suggest future research directions.



## **19. Adversarial Feature Alignment: Balancing Robustness and Accuracy in Deep Learning via Adversarial Training**

cs.CV

19 pages, 5 figures, 16 tables, 2 algorithms

**SubmitDate**: 2024-02-19    [abs](http://arxiv.org/abs/2402.12187v1) [paper-pdf](http://arxiv.org/pdf/2402.12187v1)

**Authors**: Leo Hyun Park, Jaeuk Kim, Myung Gyo Oh, Jaewoo Park, Taekyoung Kwon

**Abstract**: Deep learning models continue to advance in accuracy, yet they remain vulnerable to adversarial attacks, which often lead to the misclassification of adversarial examples. Adversarial training is used to mitigate this problem by increasing robustness against these attacks. However, this approach typically reduces a model's standard accuracy on clean, non-adversarial samples. The necessity for deep learning models to balance both robustness and accuracy for security is obvious, but achieving this balance remains challenging, and the underlying reasons are yet to be clarified. This paper proposes a novel adversarial training method called Adversarial Feature Alignment (AFA), to address these problems. Our research unveils an intriguing insight: misalignment within the feature space often leads to misclassification, regardless of whether the samples are benign or adversarial. AFA mitigates this risk by employing a novel optimization algorithm based on contrastive learning to alleviate potential feature misalignment. Through our evaluations, we demonstrate the superior performance of AFA. The baseline AFA delivers higher robust accuracy than previous adversarial contrastive learning methods while minimizing the drop in clean accuracy to 1.86% and 8.91% on CIFAR10 and CIFAR100, respectively, in comparison to cross-entropy. We also show that joint optimization of AFA and TRADES, accompanied by data augmentation using a recent diffusion model, achieves state-of-the-art accuracy and robustness.



## **20. Dynamic Graph Information Bottleneck**

cs.LG

Accepted by the research tracks of The Web Conference 2024 (WWW 2024)

**SubmitDate**: 2024-02-19    [abs](http://arxiv.org/abs/2402.06716v2) [paper-pdf](http://arxiv.org/pdf/2402.06716v2)

**Authors**: Haonan Yuan, Qingyun Sun, Xingcheng Fu, Cheng Ji, Jianxin Li

**Abstract**: Dynamic Graphs widely exist in the real world, which carry complicated spatial and temporal feature patterns, challenging their representation learning. Dynamic Graph Neural Networks (DGNNs) have shown impressive predictive abilities by exploiting the intrinsic dynamics. However, DGNNs exhibit limited robustness, prone to adversarial attacks. This paper presents the novel Dynamic Graph Information Bottleneck (DGIB) framework to learn robust and discriminative representations. Leveraged by the Information Bottleneck (IB) principle, we first propose the expected optimal representations should satisfy the Minimal-Sufficient-Consensual (MSC) Condition. To compress redundant as well as conserve meritorious information into latent representation, DGIB iteratively directs and refines the structural and feature information flow passing through graph snapshots. To meet the MSC Condition, we decompose the overall IB objectives into DGIB$_{MS}$ and DGIB$_C$, in which the DGIB$_{MS}$ channel aims to learn the minimal and sufficient representations, with the DGIB$_{MS}$ channel guarantees the predictive consensus. Extensive experiments on real-world and synthetic dynamic graph datasets demonstrate the superior robustness of DGIB against adversarial attacks compared with state-of-the-art baselines in the link prediction task. To the best of our knowledge, DGIB is the first work to learn robust representations of dynamic graphs grounded in the information-theoretic IB principle.



## **21. Self-Guided Robust Graph Structure Refinement**

cs.LG

This paper has been accepted by TheWebConf 2024

**SubmitDate**: 2024-02-19    [abs](http://arxiv.org/abs/2402.11837v1) [paper-pdf](http://arxiv.org/pdf/2402.11837v1)

**Authors**: Yeonjun In, Kanghoon Yoon, Kibum Kim, Kijung Shin, Chanyoung Park

**Abstract**: Recent studies have revealed that GNNs are vulnerable to adversarial attacks. To defend against such attacks, robust graph structure refinement (GSR) methods aim at minimizing the effect of adversarial edges based on node features, graph structure, or external information. However, we have discovered that existing GSR methods are limited by narrowassumptions, such as assuming clean node features, moderate structural attacks, and the availability of external clean graphs, resulting in the restricted applicability in real-world scenarios. In this paper, we propose a self-guided GSR framework (SG-GSR), which utilizes a clean sub-graph found within the given attacked graph itself. Furthermore, we propose a novel graph augmentation and a group-training strategy to handle the two technical challenges in the clean sub-graph extraction: 1) loss of structural information, and 2) imbalanced node degree distribution. Extensive experiments demonstrate the effectiveness of SG-GSR under various scenarios including non-targeted attacks, targeted attacks, feature attacks, e-commerce fraud, and noisy node labels. Our code is available at https://github.com/yeonjun-in/torch-SG-GSR.



## **22. On the Safety Concerns of Deploying LLMs/VLMs in Robotics: Highlighting the Risks and Vulnerabilities**

cs.RO

**SubmitDate**: 2024-02-19    [abs](http://arxiv.org/abs/2402.10340v2) [paper-pdf](http://arxiv.org/pdf/2402.10340v2)

**Authors**: Xiyang Wu, Ruiqi Xian, Tianrui Guan, Jing Liang, Souradip Chakraborty, Fuxiao Liu, Brian Sadler, Dinesh Manocha, Amrit Singh Bedi

**Abstract**: In this paper, we highlight the critical issues of robustness and safety associated with integrating large language models (LLMs) and vision-language models (VLMs) into robotics applications. Recent works have focused on using LLMs and VLMs to improve the performance of robotics tasks, such as manipulation, navigation, etc. However, such integration can introduce significant vulnerabilities, in terms of their susceptibility to adversarial attacks due to the language models, potentially leading to catastrophic consequences. By examining recent works at the interface of LLMs/VLMs and robotics, we show that it is easy to manipulate or misguide the robot's actions, leading to safety hazards. We define and provide examples of several plausible adversarial attacks, and conduct experiments on three prominent robot frameworks integrated with a language model, including KnowNo VIMA, and Instruct2Act, to assess their susceptibility to these attacks. Our empirical findings reveal a striking vulnerability of LLM/VLM-robot integrated systems: simple adversarial attacks can significantly undermine the effectiveness of LLM/VLM-robot integrated systems. Specifically, our data demonstrate an average performance deterioration of 21.2% under prompt attacks and a more alarming 30.2% under perception attacks. These results underscore the critical need for robust countermeasures to ensure the safe and reliable deployment of the advanced LLM/VLM-based robotic systems.



## **23. SAGMAN: Stability Analysis of Graph Neural Networks on the Manifolds**

cs.LG

**SubmitDate**: 2024-02-19    [abs](http://arxiv.org/abs/2402.08653v2) [paper-pdf](http://arxiv.org/pdf/2402.08653v2)

**Authors**: Wuxinlin Cheng, Chenhui Deng, Ali Aghdaei, Zhiru Zhang, Zhuo Feng

**Abstract**: Modern graph neural networks (GNNs) can be sensitive to changes in the input graph structure and node features, potentially resulting in unpredictable behavior and degraded performance. In this work, we introduce a spectral framework known as SAGMAN for examining the stability of GNNs. This framework assesses the distance distortions that arise from the nonlinear mappings of GNNs between the input and output manifolds: when two nearby nodes on the input manifold are mapped (through a GNN model) to two distant ones on the output manifold, it implies a large distance distortion and thus a poor GNN stability. We propose a distance-preserving graph dimension reduction (GDR) approach that utilizes spectral graph embedding and probabilistic graphical models (PGMs) to create low-dimensional input/output graph-based manifolds for meaningful stability analysis. Our empirical evaluations show that SAGMAN effectively assesses the stability of each node when subjected to various edge or feature perturbations, offering a scalable approach for evaluating the stability of GNNs, extending to applications within recommendation systems. Furthermore, we illustrate its utility in downstream tasks, notably in enhancing GNN stability and facilitating adversarial targeted attacks.



## **24. The Effectiveness of Random Forgetting for Robust Generalization**

cs.LG

Published as a conference paper at ICLR 2024

**SubmitDate**: 2024-02-18    [abs](http://arxiv.org/abs/2402.11733v1) [paper-pdf](http://arxiv.org/pdf/2402.11733v1)

**Authors**: Vijaya Raghavan T Ramkumar, Bahram Zonooz, Elahe Arani

**Abstract**: Deep neural networks are susceptible to adversarial attacks, which can compromise their performance and accuracy. Adversarial Training (AT) has emerged as a popular approach for protecting neural networks against such attacks. However, a key challenge of AT is robust overfitting, where the network's robust performance on test data deteriorates with further training, thus hindering generalization. Motivated by the concept of active forgetting in the brain, we introduce a novel learning paradigm called "Forget to Mitigate Overfitting (FOMO)". FOMO alternates between the forgetting phase, which randomly forgets a subset of weights and regulates the model's information through weight reinitialization, and the relearning phase, which emphasizes learning generalizable features. Our experiments on benchmark datasets and adversarial attacks show that FOMO alleviates robust overfitting by significantly reducing the gap between the best and last robust test accuracy while improving the state-of-the-art robustness. Furthermore, FOMO provides a better trade-off between standard and robust accuracy, outperforming baseline adversarial methods. Finally, our framework is robust to AutoAttacks and increases generalization in many real-world scenarios.



## **25. OUTFOX: LLM-Generated Essay Detection Through In-Context Learning with Adversarially Generated Examples**

cs.CL

AAAI 2024 camera ready. Code and dataset available at  https://github.com/ryuryukke/OUTFOX

**SubmitDate**: 2024-02-18    [abs](http://arxiv.org/abs/2307.11729v3) [paper-pdf](http://arxiv.org/pdf/2307.11729v3)

**Authors**: Ryuto Koike, Masahiro Kaneko, Naoaki Okazaki

**Abstract**: Large Language Models (LLMs) have achieved human-level fluency in text generation, making it difficult to distinguish between human-written and LLM-generated texts. This poses a growing risk of misuse of LLMs and demands the development of detectors to identify LLM-generated texts. However, existing detectors lack robustness against attacks: they degrade detection accuracy by simply paraphrasing LLM-generated texts. Furthermore, a malicious user might attempt to deliberately evade the detectors based on detection results, but this has not been assumed in previous studies. In this paper, we propose OUTFOX, a framework that improves the robustness of LLM-generated-text detectors by allowing both the detector and the attacker to consider each other's output. In this framework, the attacker uses the detector's prediction labels as examples for in-context learning and adversarially generates essays that are harder to detect, while the detector uses the adversarially generated essays as examples for in-context learning to learn to detect essays from a strong attacker. Experiments in the domain of student essays show that the proposed detector improves the detection performance on the attacker-generated texts by up to +41.3 points F1-score. Furthermore, the proposed detector shows a state-of-the-art detection performance: up to 96.9 points F1-score, beating existing detectors on non-attacked texts. Finally, the proposed attacker drastically degrades the performance of detectors by up to -57.0 points F1-score, massively outperforming the baseline paraphrasing method for evading detection.



## **26. Evaluating Adversarial Robustness of Low dose CT Recovery**

eess.IV

MIDL 2023

**SubmitDate**: 2024-02-18    [abs](http://arxiv.org/abs/2402.11557v1) [paper-pdf](http://arxiv.org/pdf/2402.11557v1)

**Authors**: Kanchana Vaishnavi Gandikota, Paramanand Chandramouli, Hannah Droege, Michael Moeller

**Abstract**: Low dose computed tomography (CT) acquisition using reduced radiation or sparse angle measurements is recommended to decrease the harmful effects of X-ray radiation. Recent works successfully apply deep networks to the problem of low dose CT recovery on bench-mark datasets. However, their robustness needs a thorough evaluation before use in clinical settings. In this work, we evaluate the robustness of different deep learning approaches and classical methods for CT recovery. We show that deep networks, including model-based networks encouraging data consistency, are more susceptible to untargeted attacks. Surprisingly, we observe that data consistency is not heavily affected even for these poor quality reconstructions, motivating the need for better regularization for the networks. We demonstrate the feasibility of universal attacks and study attack transferability across different methods. We analyze robustness to attacks causing localized changes in clinically relevant regions. Both classical approaches and deep networks are affected by such attacks leading to changes in the visual appearance of localized lesions, for extremely small perturbations. As the resulting reconstructions have high data consistency with the original measurements, these localized attacks can be used to explore the solution space of the CT recovery problem.



## **27. Measuring Privacy Loss in Distributed Spatio-Temporal Data**

cs.CR

Chrome PDF viewer might not display Figures 3 and 4 properly

**SubmitDate**: 2024-02-18    [abs](http://arxiv.org/abs/2402.11526v1) [paper-pdf](http://arxiv.org/pdf/2402.11526v1)

**Authors**: Tatsuki Koga, Casey Meehan, Kamalika Chaudhuri

**Abstract**: Statistics about traffic flow and people's movement gathered from multiple geographical locations in a distributed manner are the driving force powering many applications, such as traffic prediction, demand prediction, and restaurant occupancy reports. However, these statistics are often based on sensitive location data of people, and hence privacy has to be preserved while releasing them. The standard way to do this is via differential privacy, which guarantees a form of rigorous, worst-case, person-level privacy. In this work, motivated by several counter-intuitive features of differential privacy in distributed location applications, we propose an alternative privacy loss against location reconstruction attacks by an informed adversary. Our experiments on real and synthetic data demonstrate that our privacy loss better reflects our intuitions on individual privacy violation in the distributed spatio-temporal setting.



## **28. A Wolf in Sheep's Clothing: Generalized Nested Jailbreak Prompts can Fool Large Language Models Easily**

cs.CL

Pre-print, code is available at https://github.com/NJUNLP/ReNeLLM

**SubmitDate**: 2024-02-18    [abs](http://arxiv.org/abs/2311.08268v2) [paper-pdf](http://arxiv.org/pdf/2311.08268v2)

**Authors**: Peng Ding, Jun Kuang, Dan Ma, Xuezhi Cao, Yunsen Xian, Jiajun Chen, Shujian Huang

**Abstract**: Large Language Models (LLMs), such as ChatGPT and GPT-4, are designed to provide useful and safe responses. However, adversarial prompts known as 'jailbreaks' can circumvent safeguards, leading LLMs to generate potentially harmful content. Exploring jailbreak prompts can help to better reveal the weaknesses of LLMs and further steer us to secure them. Unfortunately, existing jailbreak methods either suffer from intricate manual design or require optimization on other white-box models, compromising generalization or efficiency. In this paper, we generalize jailbreak prompt attacks into two aspects: (1) Prompt Rewriting and (2) Scenario Nesting. Based on this, we propose ReNeLLM, an automatic framework that leverages LLMs themselves to generate effective jailbreak prompts. Extensive experiments demonstrate that ReNeLLM significantly improves the attack success rate while greatly reducing the time cost compared to existing baselines. Our study also reveals the inadequacy of current defense methods in safeguarding LLMs. Finally, we analyze the failure of LLMs defense from the perspective of prompt execution priority, and propose corresponding defense strategies. We hope that our research can catalyze both the academic community and LLMs developers towards the provision of safer and more regulated LLMs. The code is available at https://github.com/NJUNLP/ReNeLLM.



## **29. Token-Level Adversarial Prompt Detection Based on Perplexity Measures and Contextual Information**

cs.CL

**SubmitDate**: 2024-02-18    [abs](http://arxiv.org/abs/2311.11509v3) [paper-pdf](http://arxiv.org/pdf/2311.11509v3)

**Authors**: Zhengmian Hu, Gang Wu, Saayan Mitra, Ruiyi Zhang, Tong Sun, Heng Huang, Viswanathan Swaminathan

**Abstract**: In recent years, Large Language Models (LLM) have emerged as pivotal tools in various applications. However, these models are susceptible to adversarial prompt attacks, where attackers can carefully curate input strings that mislead LLMs into generating incorrect or undesired outputs. Previous work has revealed that with relatively simple yet effective attacks based on discrete optimization, it is possible to generate adversarial prompts that bypass moderation and alignment of the models. This vulnerability to adversarial prompts underscores a significant concern regarding the robustness and reliability of LLMs. Our work aims to address this concern by introducing a novel approach to detecting adversarial prompts at a token level, leveraging the LLM's capability to predict the next token's probability. We measure the degree of the model's perplexity, where tokens predicted with high probability are considered normal, and those exhibiting high perplexity are flagged as adversarial. Additionaly, our method also integrates context understanding by incorporating neighboring token information to encourage the detection of contiguous adversarial prompt sequences. To this end, we design two algorithms for adversarial prompt detection: one based on optimization techniques and another on Probabilistic Graphical Models (PGM). Both methods are equipped with efficient solving methods, ensuring efficient adversarial prompt detection. Our token-level detection result can be visualized as heatmap overlays on the text sequence, allowing for a clearer and more intuitive representation of which part of the text may contain adversarial prompts.



## **30. A Curious Case of Searching for the Correlation between Training Data and Adversarial Robustness of Transformer Textual Models**

cs.LG

**SubmitDate**: 2024-02-18    [abs](http://arxiv.org/abs/2402.11469v1) [paper-pdf](http://arxiv.org/pdf/2402.11469v1)

**Authors**: Cuong Dang, Dung D. Le, Thai Le

**Abstract**: Existing works have shown that fine-tuned textual transformer models achieve state-of-the-art prediction performances but are also vulnerable to adversarial text perturbations. Traditional adversarial evaluation is often done \textit{only after} fine-tuning the models and ignoring the training data. In this paper, we want to prove that there is also a strong correlation between training data and model robustness. To this end, we extract 13 different features representing a wide range of input fine-tuning corpora properties and use them to predict the adversarial robustness of the fine-tuned models. Focusing mostly on encoder-only transformer models BERT and RoBERTa with additional results for BART, ELECTRA and GPT2, we provide diverse evidence to support our argument. First, empirical analyses show that (a) extracted features can be used with a lightweight classifier such as Random Forest to effectively predict the attack success rate and (b) features with the most influence on the model robustness have a clear correlation with the robustness. Second, our framework can be used as a fast and effective additional tool for robustness evaluation since it (a) saves 30x-193x runtime compared to the traditional technique, (b) is transferable across models, (c) can be used under adversarial training, and (d) robust to statistical randomness. Our code will be publicly available.



## **31. VoltSchemer: Use Voltage Noise to Manipulate Your Wireless Charger**

cs.CR

This paper has been accepted by the 33rd USENIX Security Symposium

**SubmitDate**: 2024-02-18    [abs](http://arxiv.org/abs/2402.11423v1) [paper-pdf](http://arxiv.org/pdf/2402.11423v1)

**Authors**: Zihao Zhan, Yirui Yang, Haoqi Shan, Hanqiu Wang, Yier Jin, Shuo Wang

**Abstract**: Wireless charging is becoming an increasingly popular charging solution in portable electronic products for a more convenient and safer charging experience than conventional wired charging. However, our research identified new vulnerabilities in wireless charging systems, making them susceptible to intentional electromagnetic interference. These vulnerabilities facilitate a set of novel attack vectors, enabling adversaries to manipulate the charger and perform a series of attacks.   In this paper, we propose VoltSchemer, a set of innovative attacks that grant attackers control over commercial-off-the-shelf wireless chargers merely by modulating the voltage from the power supply. These attacks represent the first of its kind, exploiting voltage noises from the power supply to manipulate wireless chargers without necessitating any malicious modifications to the chargers themselves. The significant threats imposed by VoltSchemer are substantiated by three practical attacks, where a charger can be manipulated to: control voice assistants via inaudible voice commands, damage devices being charged through overcharging or overheating, and bypass Qi-standard specified foreign-object-detection mechanism to damage valuable items exposed to intense magnetic fields.   We demonstrate the effectiveness and practicality of the VoltSchemer attacks with successful attacks on 9 top-selling COTS wireless chargers. Furthermore, we discuss the security implications of our findings and suggest possible countermeasures to mitigate potential threats.



## **32. Effective Prompt Extraction from Language Models**

cs.CL

**SubmitDate**: 2024-02-17    [abs](http://arxiv.org/abs/2307.06865v2) [paper-pdf](http://arxiv.org/pdf/2307.06865v2)

**Authors**: Yiming Zhang, Nicholas Carlini, Daphne Ippolito

**Abstract**: The text generated by large language models is commonly controlled by prompting, where a prompt prepended to a user's query guides the model's output. The prompts used by companies to guide their models are often treated as secrets, to be hidden from the user making the query. They have even been treated as commodities to be bought and sold. However, anecdotal reports have shown adversarial users employing prompt extraction attacks to recover these prompts. In this paper, we present a framework for systematically measuring the effectiveness of these attacks. In experiments with 3 different sources of prompts and 11 underlying large language models, we find that simple text-based attacks can in fact reveal prompts with high probability. Our framework determines with high precision whether an extracted prompt is the actual secret prompt, rather than a model hallucination. Prompt extraction experiments on real systems such as Bing Chat and ChatGPT suggest that system prompts can be revealed by an adversary despite existing defenses in place.



## **33. DALA: A Distribution-Aware LoRA-Based Adversarial Attack against Language Models**

cs.CL

First two authors contribute equally

**SubmitDate**: 2024-02-17    [abs](http://arxiv.org/abs/2311.08598v2) [paper-pdf](http://arxiv.org/pdf/2311.08598v2)

**Authors**: Yibo Wang, Xiangjue Dong, James Caverlee, Philip S. Yu

**Abstract**: Language models (LMs) can be manipulated by adversarial attacks, which introduce subtle perturbations to input data. While recent attack methods can achieve a relatively high attack success rate (ASR), we've observed that the generated adversarial examples have a different data distribution compared with the original examples. Specifically, these adversarial examples exhibit reduced confidence levels and greater divergence from the training data distribution. Consequently, they are easy to detect using straightforward detection methods, diminishing the efficacy of such attacks. To address this issue, we propose a Distribution-Aware LoRA-based Adversarial Attack (DALA) method. DALA considers distribution shifts of adversarial examples to improve the attack's effectiveness under detection methods. We further design a novel evaluation metric, the Non-detectable Attack Success Rate (NASR), which integrates both ASR and detectability for the attack task. We conduct experiments on four widely used datasets to validate the attack effectiveness and transferability of adversarial examples generated by DALA against both the white-box BERT-base model and the black-box LLaMA2-7b model. Our codes are available at https://anonymous.4open.science/r/DALA-A16D/.



## **34. On the Evaluation of User Privacy in Deep Neural Networks using Timing Side Channel**

cs.CR

15 pages, 20 figures

**SubmitDate**: 2024-02-17    [abs](http://arxiv.org/abs/2208.01113v3) [paper-pdf](http://arxiv.org/pdf/2208.01113v3)

**Authors**: Shubhi Shukla, Manaar Alam, Sarani Bhattacharya, Debdeep Mukhopadhyay, Pabitra Mitra

**Abstract**: Recent Deep Learning (DL) advancements in solving complex real-world tasks have led to its widespread adoption in practical applications. However, this opportunity comes with significant underlying risks, as many of these models rely on privacy-sensitive data for training in a variety of applications, making them an overly-exposed threat surface for privacy violations. Furthermore, the widespread use of cloud-based Machine-Learning-as-a-Service (MLaaS) for its robust infrastructure support has broadened the threat surface to include a variety of remote side-channel attacks. In this paper, we first identify and report a novel data-dependent timing side-channel leakage (termed Class Leakage) in DL implementations originating from non-constant time branching operation in a widely used DL framework PyTorch. We further demonstrate a practical inference-time attack where an adversary with user privilege and hard-label black-box access to an MLaaS can exploit Class Leakage to compromise the privacy of MLaaS users. DL models are vulnerable to Membership Inference Attack (MIA), where an adversary's objective is to deduce whether any particular data has been used while training the model. In this paper, as a separate case study, we demonstrate that a DL model secured with differential privacy (a popular countermeasure against MIA) is still vulnerable to MIA against an adversary exploiting Class Leakage. We develop an easy-to-implement countermeasure by making a constant-time branching operation that alleviates the Class Leakage and also aids in mitigating MIA. We have chosen two standard benchmarking image classification datasets, CIFAR-10 and CIFAR-100 to train five state-of-the-art pre-trained DL models, over two different computing environments having Intel Xeon and Intel i7 processors to validate our approach.



## **35. Maintaining Adversarial Robustness in Continuous Learning**

cs.LG

**SubmitDate**: 2024-02-17    [abs](http://arxiv.org/abs/2402.11196v1) [paper-pdf](http://arxiv.org/pdf/2402.11196v1)

**Authors**: Xiaolei Ru, Xiaowei Cao, Zijia Liu, Jack Murdoch Moore, Xin-Ya Zhang, Xia Zhu, Wenjia Wei, Gang Yan

**Abstract**: Adversarial robustness is essential for security and reliability of machine learning systems. However, the adversarial robustness gained by sophisticated defense algorithms is easily erased as the neural network evolves to learn new tasks. This vulnerability can be addressed by fostering a novel capability for neural networks, termed continual robust learning, which focuses on both the (classification) performance and adversarial robustness on previous tasks during continuous learning. To achieve continuous robust learning, we propose an approach called Double Gradient Projection that projects the gradients for weight updates orthogonally onto two crucial subspaces -- one for stabilizing the smoothed sample gradients and another for stabilizing the final outputs of the neural network. The experimental results on four benchmarks demonstrate that the proposed approach effectively maintains continuous robustness against strong adversarial attacks, outperforming the baselines formed by combining the existing defense strategies and continual learning methods.



## **36. Quantization Aware Attack: Enhancing Transferable Adversarial Attacks by Model Quantization**

cs.CR

Accepted by IEEE Transactions on Information Forensics and Security  in 2024

**SubmitDate**: 2024-02-17    [abs](http://arxiv.org/abs/2305.05875v3) [paper-pdf](http://arxiv.org/pdf/2305.05875v3)

**Authors**: Yulong Yang, Chenhao Lin, Qian Li, Zhengyu Zhao, Haoran Fan, Dawei Zhou, Nannan Wang, Tongliang Liu, Chao Shen

**Abstract**: Quantized neural networks (QNNs) have received increasing attention in resource-constrained scenarios due to their exceptional generalizability. However, their robustness against realistic black-box adversarial attacks has not been extensively studied. In this scenario, adversarial transferability is pursued across QNNs with different quantization bitwidths, which particularly involve unknown architectures and defense methods. Previous studies claim that transferability is difficult to achieve across QNNs with different bitwidths on the condition that they share the same architecture. However, we discover that under different architectures, transferability can be largely improved by using a QNN quantized with an extremely low bitwidth as the substitute model. We further improve the attack transferability by proposing \textit{quantization aware attack} (QAA), which fine-tunes a QNN substitute model with a multiple-bitwidth training objective. In particular, we demonstrate that QAA addresses the two issues that are commonly known to hinder transferability: 1) quantization shifts and 2) gradient misalignments. Extensive experimental results validate the high transferability of the QAA to diverse target models. For instance, when adopting the ResNet-34 substitute model on ImageNet, QAA outperforms the current best attack in attacking standardly trained DNNs, adversarially trained DNNs, and QNNs with varied bitwidths by 4.3\% $\sim$ 20.9\%, 8.7\% $\sim$ 15.5\%, and 2.6\% $\sim$ 31.1\% (absolute), respectively. In addition, QAA is efficient since it only takes one epoch for fine-tuning. In the end, we empirically explain the effectiveness of QAA from the view of the loss landscape. Our code is available at https://github.com/yyl-github-1896/QAA/



## **37. Adversarial Illusions in Multi-Modal Embeddings**

cs.CR

**SubmitDate**: 2024-02-17    [abs](http://arxiv.org/abs/2308.11804v3) [paper-pdf](http://arxiv.org/pdf/2308.11804v3)

**Authors**: Tingwei Zhang, Rishi Jha, Eugene Bagdasaryan, Vitaly Shmatikov

**Abstract**: Multi-modal embeddings encode texts, images, sounds, videos, etc., into a single embedding space, aligning representations across different modalities (e.g., associate an image of a dog with a barking sound). In this paper, we show that multi-modal embeddings can be vulnerable to an attack we call "adversarial illusions." Given an image or a sound, an adversary can perturb it to make its embedding close to an arbitrary, adversary-chosen input in another modality.   These attacks are cross-modal and targeted: the adversary is free to align any image and any sound with any target of his choice. Adversarial illusions exploit proximity in the embedding space and are thus agnostic to downstream tasks and modalities, enabling a wholesale compromise of current and future downstream tasks and modalities not available to the adversary. Using ImageBind and AudioCLIP embeddings, we demonstrate how adversarially aligned inputs, generated without knowledge of specific downstream tasks, mislead image generation, text generation, zero-shot classification, and audio retrieval.   We investigate transferability of illusions across different embeddings and develop a black-box version of our method that we use to demonstrate the first adversarial alignment attack on Amazon's commercial, proprietary Titan embedding. Finally, we analyze countermeasures and evasion attacks.



## **38. Token-Ensemble Text Generation: On Attacking the Automatic AI-Generated Text Detection**

cs.CL

Submitted to ACL 2024

**SubmitDate**: 2024-02-17    [abs](http://arxiv.org/abs/2402.11167v1) [paper-pdf](http://arxiv.org/pdf/2402.11167v1)

**Authors**: Fan Huang, Haewoon Kwak, Jisun An

**Abstract**: The robustness of AI-content detection models against cultivated attacks (e.g., paraphrasing or word switching) remains a significant concern. This study proposes a novel token-ensemble generation strategy to challenge the robustness of current AI-content detection approaches. We explore the ensemble attack strategy by completing the prompt with the next token generated from random candidate LLMs. We find the token-ensemble approach significantly drops the performance of AI-content detection models (The code and test sets will be released). Our findings reveal that token-ensemble generation poses a vital challenge to current detection models and underlines the need for advancing detection technologies to counter sophisticated adversarial strategies.



## **39. DART: A Principled Approach to Adversarially Robust Unsupervised Domain Adaptation**

cs.LG

**SubmitDate**: 2024-02-16    [abs](http://arxiv.org/abs/2402.11120v1) [paper-pdf](http://arxiv.org/pdf/2402.11120v1)

**Authors**: Yunjuan Wang, Hussein Hazimeh, Natalia Ponomareva, Alexey Kurakin, Ibrahim Hammoud, Raman Arora

**Abstract**: Distribution shifts and adversarial examples are two major challenges for deploying machine learning models. While these challenges have been studied individually, their combination is an important topic that remains relatively under-explored. In this work, we study the problem of adversarial robustness under a common setting of distribution shift - unsupervised domain adaptation (UDA). Specifically, given a labeled source domain $D_S$ and an unlabeled target domain $D_T$ with related but different distributions, the goal is to obtain an adversarially robust model for $D_T$. The absence of target domain labels poses a unique challenge, as conventional adversarial robustness defenses cannot be directly applied to $D_T$. To address this challenge, we first establish a generalization bound for the adversarial target loss, which consists of (i) terms related to the loss on the data, and (ii) a measure of worst-case domain divergence. Motivated by this bound, we develop a novel unified defense framework called Divergence Aware adveRsarial Training (DART), which can be used in conjunction with a variety of standard UDA methods; e.g., DANN [Ganin and Lempitsky, 2015]. DART is applicable to general threat models, including the popular $\ell_p$-norm model, and does not require heuristic regularizers or architectural changes. We also release DomainRobust: a testbed for evaluating robustness of UDA models to adversarial attacks. DomainRobust consists of 4 multi-domain benchmark datasets (with 46 source-target pairs) and 7 meta-algorithms with a total of 11 variants. Our large-scale experiments demonstrate that on average, DART significantly enhances model robustness on all benchmarks compared to the state of the art, while maintaining competitive standard accuracy. The relative improvement in robustness from DART reaches up to 29.2% on the source-target domain pairs considered.



## **40. VQAttack: Transferable Adversarial Attacks on Visual Question Answering via Pre-trained Models**

cs.CV

AAAI 2024, 11 pages

**SubmitDate**: 2024-02-16    [abs](http://arxiv.org/abs/2402.11083v1) [paper-pdf](http://arxiv.org/pdf/2402.11083v1)

**Authors**: Ziyi Yin, Muchao Ye, Tianrong Zhang, Jiaqi Wang, Han Liu, Jinghui Chen, Ting Wang, Fenglong Ma

**Abstract**: Visual Question Answering (VQA) is a fundamental task in computer vision and natural language process fields. Although the ``pre-training & finetuning'' learning paradigm significantly improves the VQA performance, the adversarial robustness of such a learning paradigm has not been explored. In this paper, we delve into a new problem: using a pre-trained multimodal source model to create adversarial image-text pairs and then transferring them to attack the target VQA models. Correspondingly, we propose a novel VQAttack model, which can iteratively generate both image and text perturbations with the designed modules: the large language model (LLM)-enhanced image attack and the cross-modal joint attack module. At each iteration, the LLM-enhanced image attack module first optimizes the latent representation-based loss to generate feature-level image perturbations. Then it incorporates an LLM to further enhance the image perturbations by optimizing the designed masked answer anti-recovery loss. The cross-modal joint attack module will be triggered at a specific iteration, which updates the image and text perturbations sequentially. Notably, the text perturbation updates are based on both the learned gradients in the word embedding space and word synonym-based substitution. Experimental results on two VQA datasets with five validated models demonstrate the effectiveness of the proposed VQAttack in the transferable attack setting, compared with state-of-the-art baselines. This work reveals a significant blind spot in the ``pre-training & fine-tuning'' paradigm on VQA tasks. Source codes will be released.



## **41. The AI Security Pyramid of Pain**

cs.CR

SPIE DCS 2024

**SubmitDate**: 2024-02-16    [abs](http://arxiv.org/abs/2402.11082v1) [paper-pdf](http://arxiv.org/pdf/2402.11082v1)

**Authors**: Chris M. Ward, Josh Harguess, Julia Tao, Daniel Christman, Paul Spicer, Mike Tan

**Abstract**: We introduce the AI Security Pyramid of Pain, a framework that adapts the cybersecurity Pyramid of Pain to categorize and prioritize AI-specific threats. This framework provides a structured approach to understanding and addressing various levels of AI threats. Starting at the base, the pyramid emphasizes Data Integrity, which is essential for the accuracy and reliability of datasets and AI models, including their weights and parameters. Ensuring data integrity is crucial, as it underpins the effectiveness of all AI-driven decisions and operations. The next level, AI System Performance, focuses on MLOps-driven metrics such as model drift, accuracy, and false positive rates. These metrics are crucial for detecting potential security breaches, allowing for early intervention and maintenance of AI system integrity. Advancing further, the pyramid addresses the threat posed by Adversarial Tools, identifying and neutralizing tools used by adversaries to target AI systems. This layer is key to staying ahead of evolving attack methodologies. At the Adversarial Input layer, the framework addresses the detection and mitigation of inputs designed to deceive or exploit AI models. This includes techniques like adversarial patterns and prompt injection attacks, which are increasingly used in sophisticated attacks on AI systems. Data Provenance is the next critical layer, ensuring the authenticity and lineage of data and models. This layer is pivotal in preventing the use of compromised or biased data in AI systems. At the apex is the tactics, techniques, and procedures (TTPs) layer, dealing with the most complex and challenging aspects of AI security. This involves a deep understanding and strategic approach to counter advanced AI-targeted attacks, requiring comprehensive knowledge and planning.



## **42. QDoor: Exploiting Approximate Synthesis for Backdoor Attacks in Quantum Neural Networks**

quant-ph

**SubmitDate**: 2024-02-16    [abs](http://arxiv.org/abs/2307.09529v2) [paper-pdf](http://arxiv.org/pdf/2307.09529v2)

**Authors**: Cheng Chu, Fan Chen, Philip Richerme, Lei Jiang

**Abstract**: Quantum neural networks (QNNs) succeed in object recognition, natural language processing, and financial analysis. To maximize the accuracy of a QNN on a Noisy Intermediate Scale Quantum (NISQ) computer, approximate synthesis modifies the QNN circuit by reducing error-prone 2-qubit quantum gates. The success of QNNs motivates adversaries to attack QNNs via backdoors. However, na\"ively transplanting backdoors designed for classical neural networks to QNNs yields only low attack success rate, due to the noises and approximate synthesis on NISQ computers. Prior quantum circuit-based backdoors cannot selectively attack some inputs or work with all types of encoding layers of a QNN circuit. Moreover, it is easy to detect both transplanted and circuit-based backdoors in a QNN.   In this paper, we propose a novel and stealthy backdoor attack, QDoor, to achieve high attack success rate in approximately-synthesized QNN circuits by weaponizing unitary differences between uncompiled QNNs and their synthesized counterparts. QDoor trains a QNN behaving normally for all inputs with and without a trigger. However, after approximate synthesis, the QNN circuit always predicts any inputs with a trigger to a predefined class while still acts normally for benign inputs. Compared to prior backdoor attacks, QDoor improves the attack success rate by $13\times$ and the clean data accuracy by $65\%$ on average. Furthermore, prior backdoor detection techniques cannot find QDoor attacks in uncompiled QNN circuits.



## **43. Decorrelative Network Architecture for Robust Electrocardiogram Classification**

cs.LG

24 pages, 7 figures

**SubmitDate**: 2024-02-16    [abs](http://arxiv.org/abs/2207.09031v4) [paper-pdf](http://arxiv.org/pdf/2207.09031v4)

**Authors**: Christopher Wiedeman, Ge Wang

**Abstract**: Artificial intelligence has made great progress in medical data analysis, but the lack of robustness and trustworthiness has kept these methods from being widely deployed. As it is not possible to train networks that are accurate in all scenarios, models must recognize situations where they cannot operate confidently. Bayesian deep learning methods sample the model parameter space to estimate uncertainty, but these parameters are often subject to the same vulnerabilities, which can be exploited by adversarial attacks. We propose a novel ensemble approach based on feature decorrelation and Fourier partitioning for teaching networks diverse complementary features, reducing the chance of perturbation-based fooling. We test our approach on single and multi-channel electrocardiogram classification, and adapt adversarial training and DVERGE into the Bayesian ensemble framework for comparison. Our results indicate that the combination of decorrelation and Fourier partitioning generally maintains performance on unperturbed data while demonstrating superior robustness and uncertainty estimation on projected gradient descent and smooth adversarial attacks of various magnitudes. Furthermore, our approach does not require expensive optimization with adversarial samples, adding much less compute to the training process than adversarial training or DVERGE. These methods can be applied to other tasks for more robust and trustworthy models.



## **44. TernaryVote: Differentially Private, Communication Efficient, and Byzantine Resilient Distributed Optimization on Heterogeneous Data**

cs.LG

**SubmitDate**: 2024-02-16    [abs](http://arxiv.org/abs/2402.10816v1) [paper-pdf](http://arxiv.org/pdf/2402.10816v1)

**Authors**: Richeng Jin, Yujie Gu, Kai Yue, Xiaofan He, Zhaoyang Zhang, Huaiyu Dai

**Abstract**: Distributed training of deep neural networks faces three critical challenges: privacy preservation, communication efficiency, and robustness to fault and adversarial behaviors. Although significant research efforts have been devoted to addressing these challenges independently, their synthesis remains less explored. In this paper, we propose TernaryVote, which combines a ternary compressor and the majority vote mechanism to realize differential privacy, gradient compression, and Byzantine resilience simultaneously. We theoretically quantify the privacy guarantee through the lens of the emerging f-differential privacy (DP) and the Byzantine resilience of the proposed algorithm. Particularly, in terms of privacy guarantees, compared to the existing sign-based approach StoSign, the proposed method improves the dimension dependence on the gradient size and enjoys privacy amplification by mini-batch sampling while ensuring a comparable convergence rate. We also prove that TernaryVote is robust when less than 50% of workers are blind attackers, which matches that of SIGNSGD with majority vote. Extensive experimental results validate the effectiveness of the proposed algorithm.



## **45. Uncertainty, Calibration, and Membership Inference Attacks: An Information-Theoretic Perspective**

cs.IT

27 pages, 13 figures

**SubmitDate**: 2024-02-16    [abs](http://arxiv.org/abs/2402.10686v1) [paper-pdf](http://arxiv.org/pdf/2402.10686v1)

**Authors**: Meiyi Zhu, Caili Guo, Chunyan Feng, Osvaldo Simeone

**Abstract**: In a membership inference attack (MIA), an attacker exploits the overconfidence exhibited by typical machine learning models to determine whether a specific data point was used to train a target model. In this paper, we analyze the performance of the state-of-the-art likelihood ratio attack (LiRA) within an information-theoretical framework that allows the investigation of the impact of the aleatoric uncertainty in the true data generation process, of the epistemic uncertainty caused by a limited training data set, and of the calibration level of the target model. We compare three different settings, in which the attacker receives decreasingly informative feedback from the target model: confidence vector (CV) disclosure, in which the output probability vector is released; true label confidence (TLC) disclosure, in which only the probability assigned to the true label is made available by the model; and decision set (DS) disclosure, in which an adaptive prediction set is produced as in conformal prediction. We derive bounds on the advantage of an MIA adversary with the aim of offering insights into the impact of uncertainty and calibration on the effectiveness of MIAs. Simulation results demonstrate that the derived analytical bounds predict well the effectiveness of MIAs.



## **46. Zero-shot sampling of adversarial entities in biomedical question answering**

cs.CL

20 pages incl. appendix, under review

**SubmitDate**: 2024-02-16    [abs](http://arxiv.org/abs/2402.10527v1) [paper-pdf](http://arxiv.org/pdf/2402.10527v1)

**Authors**: R. Patrick Xian, Alex J. Lee, Vincent Wang, Qiming Cui, Russell Ro, Reza Abbasi-Asl

**Abstract**: The increasing depth of parametric domain knowledge in large language models (LLMs) is fueling their rapid deployment in real-world applications. In high-stakes and knowledge-intensive tasks, understanding model vulnerabilities is essential for quantifying the trustworthiness of model predictions and regulating their use. The recent discovery of named entities as adversarial examples in natural language processing tasks raises questions about their potential guises in other settings. Here, we propose a powerscaled distance-weighted sampling scheme in embedding space to discover diverse adversarial entities as distractors. We demonstrate its advantage over random sampling in adversarial question answering on biomedical topics. Our approach enables the exploration of different regions on the attack surface, which reveals two regimes of adversarial entities that markedly differ in their characteristics. Moreover, we show that the attacks successfully manipulate token-wise Shapley value explanations, which become deceptive in the adversarial setting. Our investigations illustrate the brittleness of domain knowledge in LLMs and reveal a shortcoming of standard evaluations for high-capacity models.



## **47. Benchmarking Transferable Adversarial Attacks**

cs.CV

Accepted by NDSS 2024 Workshop

**SubmitDate**: 2024-02-16    [abs](http://arxiv.org/abs/2402.00418v3) [paper-pdf](http://arxiv.org/pdf/2402.00418v3)

**Authors**: Zhibo Jin, Jiayu Zhang, Zhiyu Zhu, Huaming Chen

**Abstract**: The robustness of deep learning models against adversarial attacks remains a pivotal concern. This study presents, for the first time, an exhaustive review of the transferability aspect of adversarial attacks. It systematically categorizes and critically evaluates various methodologies developed to augment the transferability of adversarial attacks. This study encompasses a spectrum of techniques, including Generative Structure, Semantic Similarity, Gradient Editing, Target Modification, and Ensemble Approach. Concurrently, this paper introduces a benchmark framework \textit{TAA-Bench}, integrating ten leading methodologies for adversarial attack transferability, thereby providing a standardized and systematic platform for comparative analysis across diverse model architectures. Through comprehensive scrutiny, we delineate the efficacy and constraints of each method, shedding light on their underlying operational principles and practical utility. This review endeavors to be a quintessential resource for both scholars and practitioners in the field, charting the complex terrain of adversarial transferability and setting a foundation for future explorations in this vital sector. The associated codebase is accessible at: https://github.com/KxPlaug/TAA-Bench



## **48. Rethinking Adversarial Policies: A Generalized Attack Formulation and Provable Defense in RL**

cs.LG

International Conference on Learning Representations (ICLR) 2024

**SubmitDate**: 2024-02-20    [abs](http://arxiv.org/abs/2305.17342v3) [paper-pdf](http://arxiv.org/pdf/2305.17342v3)

**Authors**: Xiangyu Liu, Souradip Chakraborty, Yanchao Sun, Furong Huang

**Abstract**: Most existing works focus on direct perturbations to the victim's state/action or the underlying transition dynamics to demonstrate the vulnerability of reinforcement learning agents to adversarial attacks. However, such direct manipulations may not be always realizable. In this paper, we consider a multi-agent setting where a well-trained victim agent $\nu$ is exploited by an attacker controlling another agent $\alpha$ with an \textit{adversarial policy}. Previous models do not account for the possibility that the attacker may only have partial control over $\alpha$ or that the attack may produce easily detectable "abnormal" behaviors. Furthermore, there is a lack of provably efficient defenses against these adversarial policies. To address these limitations, we introduce a generalized attack framework that has the flexibility to model to what extent the adversary is able to control the agent, and allows the attacker to regulate the state distribution shift and produce stealthier adversarial policies. Moreover, we offer a provably efficient defense with polynomial convergence to the most robust victim policy through adversarial training with timescale separation. This stands in sharp contrast to supervised learning, where adversarial training typically provides only \textit{empirical} defenses. Using the Robosumo competition experiments, we show that our generalized attack formulation results in much stealthier adversarial policies when maintaining the same winning rate as baselines. Additionally, our adversarial training approach yields stable learning dynamics and less exploitable victim policies.



## **49. PPR: Enhancing Dodging Attacks while Maintaining Impersonation Attacks on Face Recognition Systems**

cs.CV

**SubmitDate**: 2024-02-16    [abs](http://arxiv.org/abs/2401.08903v2) [paper-pdf](http://arxiv.org/pdf/2401.08903v2)

**Authors**: Fengfan Zhou, Heifei Ling, Bangjie Yin, Hui Zheng

**Abstract**: Adversarial Attacks on Face Recognition (FR) encompass two types: impersonation attacks and evasion attacks. We observe that achieving a successful impersonation attack on FR does not necessarily ensure a successful dodging attack on FR in the black-box setting. Introducing a novel attack method named Pre-training Pruning Restoration Attack (PPR), we aim to enhance the performance of dodging attacks whilst avoiding the degradation of impersonation attacks. Our method employs adversarial example pruning, enabling a portion of adversarial perturbations to be set to zero, while tending to maintain the attack performance. By utilizing adversarial example pruning, we can prune the pre-trained adversarial examples and selectively free up certain adversarial perturbations. Thereafter, we embed adversarial perturbations in the pruned area, which enhances the dodging performance of the adversarial face examples. The effectiveness of our proposed attack method is demonstrated through our experimental results, showcasing its superior performance.



## **50. Quantum-Inspired Analysis of Neural Network Vulnerabilities: The Role of Conjugate Variables in System Attacks**

cs.LG

13 pages, 3 figures

**SubmitDate**: 2024-02-16    [abs](http://arxiv.org/abs/2402.10983v1) [paper-pdf](http://arxiv.org/pdf/2402.10983v1)

**Authors**: Jun-Jie Zhang, Deyu Meng

**Abstract**: Neural networks demonstrate inherent vulnerability to small, non-random perturbations, emerging as adversarial attacks. Such attacks, born from the gradient of the loss function relative to the input, are discerned as input conjugates, revealing a systemic fragility within the network structure. Intriguingly, a mathematical congruence manifests between this mechanism and the quantum physics' uncertainty principle, casting light on a hitherto unanticipated interdisciplinarity. This inherent susceptibility within neural network systems is generally intrinsic, highlighting not only the innate vulnerability of these networks but also suggesting potential advancements in the interdisciplinary area for understanding these black-box networks.



