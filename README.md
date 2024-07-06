# Latest Adversarial Attack Papers
**update at 2024-07-06 15:45:12**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. Correlated Privacy Mechanisms for Differentially Private Distributed Mean Estimation**

cs.IT

**SubmitDate**: 2024-07-03    [abs](http://arxiv.org/abs/2407.03289v1) [paper-pdf](http://arxiv.org/pdf/2407.03289v1)

**Authors**: Sajani Vithana, Viveck R. Cadambe, Flavio P. Calmon, Haewon Jeong

**Abstract**: Differentially private distributed mean estimation (DP-DME) is a fundamental building block in privacy-preserving federated learning, where a central server estimates the mean of $d$-dimensional vectors held by $n$ users while ensuring $(\epsilon,\delta)$-DP. Local differential privacy (LDP) and distributed DP with secure aggregation (SecAgg) are the most common notions of DP used in DP-DME settings with an untrusted server. LDP provides strong resilience to dropouts, colluding users, and malicious server attacks, but suffers from poor utility. In contrast, SecAgg-based DP-DME achieves an $O(n)$ utility gain over LDP in DME, but requires increased communication and computation overheads and complex multi-round protocols to handle dropouts and malicious attacks. In this work, we propose CorDP-DME, a novel DP-DME mechanism that spans the gap between DME with LDP and distributed DP, offering a favorable balance between utility and resilience to dropout and collusion. CorDP-DME is based on correlated Gaussian noise, ensuring DP without the perfect conditional privacy guarantees of SecAgg-based approaches. We provide an information-theoretic analysis of CorDP-DME, and derive theoretical guarantees for utility under any given privacy parameters and dropout/colluding user thresholds. Our results demonstrate that (anti) correlated Gaussian DP mechanisms can significantly improve utility in mean estimation tasks compared to LDP -- even in adversarial settings -- while maintaining better resilience to dropouts and attacks compared to distributed DP.



## **2. Self-Evaluation as a Defense Against Adversarial Attacks on LLMs**

cs.LG

8 pages, 7 figures

**SubmitDate**: 2024-07-03    [abs](http://arxiv.org/abs/2407.03234v1) [paper-pdf](http://arxiv.org/pdf/2407.03234v1)

**Authors**: Hannah Brown, Leon Lin, Kenji Kawaguchi, Michael Shieh

**Abstract**: When LLMs are deployed in sensitive, human-facing settings, it is crucial that they do not output unsafe, biased, or privacy-violating outputs. For this reason, models are both trained and instructed to refuse to answer unsafe prompts such as "Tell me how to build a bomb." We find that, despite these safeguards, it is possible to break model defenses simply by appending a space to the end of a model's input. In a study of eight open-source models, we demonstrate that this acts as a strong enough attack to cause the majority of models to generate harmful outputs with very high success rates. We examine the causes of this behavior, finding that the contexts in which single spaces occur in tokenized training data encourage models to generate lists when prompted, overriding training signals to refuse to answer unsafe requests. Our findings underscore the fragile state of current model alignment and promote the importance of developing more robust alignment methods. Code and data will be made available at https://github.com/Linlt-leon/Adversarial-Alignments.



## **3. Venomancer: Towards Imperceptible and Target-on-Demand Backdoor Attacks in Federated Learning**

cs.CV

**SubmitDate**: 2024-07-03    [abs](http://arxiv.org/abs/2407.03144v1) [paper-pdf](http://arxiv.org/pdf/2407.03144v1)

**Authors**: Son Nguyen, Thinh Nguyen, Khoa Doan, Kok-Seng Wong

**Abstract**: Federated Learning (FL) is a distributed machine learning approach that maintains data privacy by training on decentralized data sources. Similar to centralized machine learning, FL is also susceptible to backdoor attacks. Most backdoor attacks in FL assume a predefined target class and require control over a large number of clients or knowledge of benign clients' information. Furthermore, they are not imperceptible and are easily detected by human inspection due to clear artifacts left on the poison data. To overcome these challenges, we propose Venomancer, an effective backdoor attack that is imperceptible and allows target-on-demand. Specifically, imperceptibility is achieved by using a visual loss function to make the poison data visually indistinguishable from the original data. Target-on-demand property allows the attacker to choose arbitrary target classes via conditional adversarial training. Additionally, experiments showed that the method is robust against state-of-the-art defenses such as Norm Clipping, Weak DP, Krum, and Multi-Krum. The source code is available at https://anonymous.4open.science/r/Venomancer-3426.



## **4. $L_p$-norm Distortion-Efficient Adversarial Attack**

cs.CV

**SubmitDate**: 2024-07-03    [abs](http://arxiv.org/abs/2407.03115v1) [paper-pdf](http://arxiv.org/pdf/2407.03115v1)

**Authors**: Chao Zhou, Yuan-Gen Wang, Zi-jia Wang, Xiangui Kang

**Abstract**: Adversarial examples have shown a powerful ability to make a well-trained model misclassified. Current mainstream adversarial attack methods only consider one of the distortions among $L_0$-norm, $L_2$-norm, and $L_\infty$-norm. $L_0$-norm based methods cause large modification on a single pixel, resulting in naked-eye visible detection, while $L_2$-norm and $L_\infty$-norm based methods suffer from weak robustness against adversarial defense since they always diffuse tiny perturbations to all pixels. A more realistic adversarial perturbation should be sparse and imperceptible. In this paper, we propose a novel $L_p$-norm distortion-efficient adversarial attack, which not only owns the least $L_2$-norm loss but also significantly reduces the $L_0$-norm distortion. To this aim, we design a new optimization scheme, which first optimizes an initial adversarial perturbation under $L_2$-norm constraint, and then constructs a dimension unimportance matrix for the initial perturbation. Such a dimension unimportance matrix can indicate the adversarial unimportance of each dimension of the initial perturbation. Furthermore, we introduce a new concept of adversarial threshold for the dimension unimportance matrix. The dimensions of the initial perturbation whose unimportance is higher than the threshold will be all set to zero, greatly decreasing the $L_0$-norm distortion. Experimental results on three benchmark datasets show that under the same query budget, the adversarial examples generated by our method have lower $L_0$-norm and $L_2$-norm distortion than the state-of-the-art. Especially for the MNIST dataset, our attack reduces 8.1$\%$ $L_2$-norm distortion meanwhile remaining 47$\%$ pixels unattacked. This demonstrates the superiority of the proposed method over its competitors in terms of adversarial robustness and visual imperceptibility.



## **5. JailbreakHunter: A Visual Analytics Approach for Jailbreak Prompts Discovery from Large-Scale Human-LLM Conversational Datasets**

cs.HC

18 pages, 9 figures

**SubmitDate**: 2024-07-03    [abs](http://arxiv.org/abs/2407.03045v1) [paper-pdf](http://arxiv.org/pdf/2407.03045v1)

**Authors**: Zhihua Jin, Shiyi Liu, Haotian Li, Xun Zhao, Huamin Qu

**Abstract**: Large Language Models (LLMs) have gained significant attention but also raised concerns due to the risk of misuse. Jailbreak prompts, a popular type of adversarial attack towards LLMs, have appeared and constantly evolved to breach the safety protocols of LLMs. To address this issue, LLMs are regularly updated with safety patches based on reported jailbreak prompts. However, malicious users often keep their successful jailbreak prompts private to exploit LLMs. To uncover these private jailbreak prompts, extensive analysis of large-scale conversational datasets is necessary to identify prompts that still manage to bypass the system's defenses. This task is highly challenging due to the immense volume of conversation data, diverse characteristics of jailbreak prompts, and their presence in complex multi-turn conversations. To tackle these challenges, we introduce JailbreakHunter, a visual analytics approach for identifying jailbreak prompts in large-scale human-LLM conversational datasets. We have designed a workflow with three analysis levels: group-level, conversation-level, and turn-level. Group-level analysis enables users to grasp the distribution of conversations and identify suspicious conversations using multiple criteria, such as similarity with reported jailbreak prompts in previous research and attack success rates. Conversation-level analysis facilitates the understanding of the progress of conversations and helps discover jailbreak prompts within their conversation contexts. Turn-level analysis allows users to explore the semantic similarity and token overlap between a singleturn prompt and the reported jailbreak prompts, aiding in the identification of new jailbreak strategies. The effectiveness and usability of the system were verified through multiple case studies and expert interviews.



## **6. Expressivity of Graph Neural Networks Through the Lens of Adversarial Robustness**

cs.LG

Published in ${2}^{nd}$ AdvML Frontiers workshop at ${40}^{th}$  International Conference on Machine Learning (ICML)

**SubmitDate**: 2024-07-03    [abs](http://arxiv.org/abs/2308.08173v2) [paper-pdf](http://arxiv.org/pdf/2308.08173v2)

**Authors**: Francesco Campi, Lukas Gosch, Tom Wollschläger, Yan Scholten, Stephan Günnemann

**Abstract**: We perform the first adversarial robustness study into Graph Neural Networks (GNNs) that are provably more powerful than traditional Message Passing Neural Networks (MPNNs). In particular, we use adversarial robustness as a tool to uncover a significant gap between their theoretically possible and empirically achieved expressive power. To do so, we focus on the ability of GNNs to count specific subgraph patterns, which is an established measure of expressivity, and extend the concept of adversarial robustness to this task. Based on this, we develop efficient adversarial attacks for subgraph counting and show that more powerful GNNs fail to generalize even to small perturbations to the graph's structure. Expanding on this, we show that such architectures also fail to count substructures on out-of-distribution graphs.



## **7. A Wolf in Sheep's Clothing: Practical Black-box Adversarial Attacks for Evading Learning-based Windows Malware Detection in the Wild**

cs.CR

This paper has been accepted by 33rd USENIX Security Symposium 2024

**SubmitDate**: 2024-07-03    [abs](http://arxiv.org/abs/2407.02886v1) [paper-pdf](http://arxiv.org/pdf/2407.02886v1)

**Authors**: Xiang Ling, Zhiyu Wu, Bin Wang, Wei Deng, Jingzheng Wu, Shouling Ji, Tianyue Luo, Yanjun Wu

**Abstract**: Given the remarkable achievements of existing learning-based malware detection in both academia and industry, this paper presents MalGuise, a practical black-box adversarial attack framework that evaluates the security risks of existing learning-based Windows malware detection systems under the black-box setting. MalGuise first employs a novel semantics-preserving transformation of call-based redividing to concurrently manipulate both nodes and edges of malware's control-flow graph, making it less noticeable. By employing a Monte-Carlo-tree-search-based optimization, MalGuise then searches for an optimized sequence of call-based redividing transformations to apply to the input Windows malware for evasions. Finally, it reconstructs the adversarial malware file based on the optimized transformation sequence while adhering to Windows executable format constraints, thereby maintaining the same semantics as the original. MalGuise is systematically evaluated against three state-of-the-art learning-based Windows malware detection systems under the black-box setting. Evaluation results demonstrate that MalGuise achieves a remarkably high attack success rate, mostly exceeding 95%, with over 91% of the generated adversarial malware files maintaining the same semantics. Furthermore, MalGuise achieves up to a 74.97% attack success rate against five anti-virus products, highlighting potential tangible security concerns to real-world users.



## **8. Steering cooperation: Adversarial attacks on prisoner's dilemma in complex networks**

physics.soc-ph

14 pages, 4 figures

**SubmitDate**: 2024-07-03    [abs](http://arxiv.org/abs/2406.19692v2) [paper-pdf](http://arxiv.org/pdf/2406.19692v2)

**Authors**: Kazuhiro Takemoto

**Abstract**: This study examines the application of adversarial attack concepts to control the evolution of cooperation in the prisoner's dilemma game in complex networks. Specifically, it proposes a simple adversarial attack method that drives players' strategies towards a target state by adding small perturbations to social networks. The proposed method is evaluated on both model and real-world networks. Numerical simulations demonstrate that the proposed method can effectively promote cooperation with significantly smaller perturbations compared to other techniques. Additionally, this study shows that adversarial attacks can also be useful in inhibiting cooperation (promoting defection). The findings reveal that adversarial attacks on social networks can be potent tools for both promoting and inhibiting cooperation, opening new possibilities for controlling cooperative behavior in social systems while also highlighting potential risks.



## **9. Light-weight Fine-tuning Method for Defending Adversarial Noise in Pre-trained Medical Vision-Language Models**

cs.CV

**SubmitDate**: 2024-07-02    [abs](http://arxiv.org/abs/2407.02716v1) [paper-pdf](http://arxiv.org/pdf/2407.02716v1)

**Authors**: Xu Han, Linghao Jin, Xuezhe Ma, Xiaofeng Liu

**Abstract**: Fine-tuning pre-trained Vision-Language Models (VLMs) has shown remarkable capabilities in medical image and textual depiction synergy. Nevertheless, many pre-training datasets are restricted by patient privacy concerns, potentially containing noise that can adversely affect downstream performance. Moreover, the growing reliance on multi-modal generation exacerbates this issue because of its susceptibility to adversarial attacks. To investigate how VLMs trained on adversarial noisy data perform on downstream medical tasks, we first craft noisy upstream datasets using multi-modal adversarial attacks. Through our comprehensive analysis, we unveil that moderate noise enhances model robustness and transferability, but increasing noise levels negatively impact downstream task performance. To mitigate this issue, we propose rectify adversarial noise (RAN) framework, a recipe designed to effectively defend adversarial attacks and rectify the influence of upstream noise during fine-tuning.



## **10. Adversarial Magnification to Deceive Deepfake Detection through Super Resolution**

cs.CV

**SubmitDate**: 2024-07-02    [abs](http://arxiv.org/abs/2407.02670v1) [paper-pdf](http://arxiv.org/pdf/2407.02670v1)

**Authors**: Davide Alessandro Coccomini, Roberto Caldelli, Giuseppe Amato, Fabrizio Falchi, Claudio Gennaro

**Abstract**: Deepfake technology is rapidly advancing, posing significant challenges to the detection of manipulated media content. Parallel to that, some adversarial attack techniques have been developed to fool the deepfake detectors and make deepfakes even more difficult to be detected. This paper explores the application of super resolution techniques as a possible adversarial attack in deepfake detection. Through our experiments, we demonstrate that minimal changes made by these methods in the visual appearance of images can have a profound impact on the performance of deepfake detection systems. We propose a novel attack using super resolution as a quick, black-box and effective method to camouflage fake images and/or generate false alarms on pristine images. Our results indicate that the usage of super resolution can significantly impair the accuracy of deepfake detectors, thereby highlighting the vulnerability of such systems to adversarial attacks. The code to reproduce our experiments is available at: https://github.com/davide-coccomini/Adversarial-Magnification-to-Deceive-Deepfake-Detection-through-Super-Resolution



## **11. Towards More Realistic Extraction Attacks: An Adversarial Perspective**

cs.CR

To be presented at PrivateNLP@ACL2024

**SubmitDate**: 2024-07-02    [abs](http://arxiv.org/abs/2407.02596v1) [paper-pdf](http://arxiv.org/pdf/2407.02596v1)

**Authors**: Yash More, Prakhar Ganesh, Golnoosh Farnadi

**Abstract**: Language models are prone to memorizing large parts of their training data, making them vulnerable to extraction attacks. Existing research on these attacks remains limited in scope, often studying isolated trends rather than the real-world interactions with these models. In this paper, we revisit extraction attacks from an adversarial perspective, exploiting the brittleness of language models. We find significant churn in extraction attack trends, i.e., even minor, unintuitive changes to the prompt, or targeting smaller models and older checkpoints, can exacerbate the risks of extraction by up to $2-4 \times$. Moreover, relying solely on the widely accepted verbatim match underestimates the extent of extracted information, and we provide various alternatives to more accurately capture the true risks of extraction. We conclude our discussion with data deduplication, a commonly suggested mitigation strategy, and find that while it addresses some memorization concerns, it remains vulnerable to the same escalation of extraction risks against a real-world adversary. Our findings highlight the necessity of acknowledging an adversary's true capabilities to avoid underestimating extraction risks.



## **12. A False Sense of Safety: Unsafe Information Leakage in 'Safe' AI Responses**

cs.CR

**SubmitDate**: 2024-07-02    [abs](http://arxiv.org/abs/2407.02551v1) [paper-pdf](http://arxiv.org/pdf/2407.02551v1)

**Authors**: David Glukhov, Ziwen Han, Ilia Shumailov, Vardan Papyan, Nicolas Papernot

**Abstract**: Large Language Models (LLMs) are vulnerable to jailbreaks$\unicode{x2013}$methods to elicit harmful or generally impermissible outputs. Safety measures are developed and assessed on their effectiveness at defending against jailbreak attacks, indicating a belief that safety is equivalent to robustness. We assert that current defense mechanisms, such as output filters and alignment fine-tuning, are, and will remain, fundamentally insufficient for ensuring model safety. These defenses fail to address risks arising from dual-intent queries and the ability to composite innocuous outputs to achieve harmful goals. To address this critical gap, we introduce an information-theoretic threat model called inferential adversaries who exploit impermissible information leakage from model outputs to achieve malicious goals. We distinguish these from commonly studied security adversaries who only seek to force victim models to generate specific impermissible outputs. We demonstrate the feasibility of automating inferential adversaries through question decomposition and response aggregation. To provide safety guarantees, we define an information censorship criterion for censorship mechanisms, bounding the leakage of impermissible information. We propose a defense mechanism which ensures this bound and reveal an intrinsic safety-utility trade-off. Our work provides the first theoretically grounded understanding of the requirements for releasing safe LLMs and the utility costs involved.



## **13. Greedy-DiM: Greedy Algorithms for Unreasonably Effective Face Morphs**

cs.CV

Accepted as a conference paper at IJCB 2024

**SubmitDate**: 2024-07-02    [abs](http://arxiv.org/abs/2404.06025v2) [paper-pdf](http://arxiv.org/pdf/2404.06025v2)

**Authors**: Zander W. Blasingame, Chen Liu

**Abstract**: Morphing attacks are an emerging threat to state-of-the-art Face Recognition (FR) systems, which aim to create a single image that contains the biometric information of multiple identities. Diffusion Morphs (DiM) are a recently proposed morphing attack that has achieved state-of-the-art performance for representation-based morphing attacks. However, none of the existing research on DiMs have leveraged the iterative nature of DiMs and left the DiM model as a black box, treating it no differently than one would a Generative Adversarial Network (GAN) or Varational AutoEncoder (VAE). We propose a greedy strategy on the iterative sampling process of DiM models which searches for an optimal step guided by an identity-based heuristic function. We compare our proposed algorithm against ten other state-of-the-art morphing algorithms using the open-source SYN-MAD 2022 competition dataset. We find that our proposed algorithm is unreasonably effective, fooling all of the tested FR systems with an MMPMR of 100%, outperforming all other morphing algorithms compared.



## **14. Steerable Pyramid Transform Enables Robust Left Ventricle Quantification**

eess.IV

Code is available at https://github.com/yangyangyang127/RobustLV

**SubmitDate**: 2024-07-02    [abs](http://arxiv.org/abs/2201.08388v2) [paper-pdf](http://arxiv.org/pdf/2201.08388v2)

**Authors**: Xiangyang Zhu, Kede Ma, Wufeng Xue

**Abstract**: Predicting cardiac indices has long been a focal point in the medical imaging community. While various deep learning models have demonstrated success in quantifying cardiac indices, they remain susceptible to mild input perturbations, e.g., spatial transformations, image distortions, and adversarial attacks. This vulnerability undermines confidence in using learning-based automated systems for diagnosing cardiovascular diseases. In this work, we describe a simple yet effective method to learn robust models for left ventricle (LV) quantification, encompassing cavity and myocardium areas, directional dimensions, and regional wall thicknesses. Our success hinges on employing the biologically inspired steerable pyramid transform (SPT) for fixed front-end processing, which offers three main benefits. First, the basis functions of SPT align with the anatomical structure of LV and the geometric features of the measured indices. Second, SPT facilitates weight sharing across different orientations as a form of parameter regularization and naturally captures the scale variations of LV. Third, the residual highpass subband can be conveniently discarded, promoting robust feature learning. Extensive experiments on the Cardiac-Dig benchmark show that our SPT-augmented model not only achieves reasonable prediction accuracy compared to state-of-the-art methods, but also exhibits significantly improved robustness against input perturbations.



## **15. EvolBA: Evolutionary Boundary Attack under Hard-label Black Box condition**

cs.CV

**SubmitDate**: 2024-07-02    [abs](http://arxiv.org/abs/2407.02248v1) [paper-pdf](http://arxiv.org/pdf/2407.02248v1)

**Authors**: Ayane Tajima, Satoshi Ono

**Abstract**: Research has shown that deep neural networks (DNNs) have vulnerabilities that can lead to the misrecognition of Adversarial Examples (AEs) with specifically designed perturbations. Various adversarial attack methods have been proposed to detect vulnerabilities under hard-label black box (HL-BB) conditions in the absence of loss gradients and confidence scores.However, these methods fall into local solutions because they search only local regions of the search space. Therefore, this study proposes an adversarial attack method named EvolBA to generate AEs using Covariance Matrix Adaptation Evolution Strategy (CMA-ES) under the HL-BB condition, where only a class label predicted by the target DNN model is available. Inspired by formula-driven supervised learning, the proposed method introduces domain-independent operators for the initialization process and a jump that enhances search exploration. Experimental results confirmed that the proposed method could determine AEs with smaller perturbations than previous methods in images where the previous methods have difficulty.



## **16. MALT Powers Up Adversarial Attacks**

cs.LG

**SubmitDate**: 2024-07-02    [abs](http://arxiv.org/abs/2407.02240v1) [paper-pdf](http://arxiv.org/pdf/2407.02240v1)

**Authors**: Odelia Melamed, Gilad Yehudai, Adi Shamir

**Abstract**: Current adversarial attacks for multi-class classifiers choose the target class for a given input naively, based on the classifier's confidence levels for various target classes. We present a novel adversarial targeting method, \textit{MALT - Mesoscopic Almost Linearity Targeting}, based on medium-scale almost linearity assumptions. Our attack wins over the current state of the art AutoAttack on the standard benchmark datasets CIFAR-100 and ImageNet and for a variety of robust models. In particular, our attack is \emph{five times faster} than AutoAttack, while successfully matching all of AutoAttack's successes and attacking additional samples that were previously out of reach. We then prove formally and demonstrate empirically that our targeting method, although inspired by linear predictors, also applies to standard non-linear models.



## **17. Secure Semantic Communication via Paired Adversarial Residual Networks**

cs.IT

**SubmitDate**: 2024-07-02    [abs](http://arxiv.org/abs/2407.02053v1) [paper-pdf](http://arxiv.org/pdf/2407.02053v1)

**Authors**: Boxiang He, Fanggang Wang, Tony Q. S. Quek

**Abstract**: This letter explores the positive side of the adversarial attack for the security-aware semantic communication system. Specifically, a pair of matching pluggable modules is installed: one after the semantic transmitter and the other before the semantic receiver. The module at transmitter uses a trainable adversarial residual network (ARN) to generate adversarial examples, while the module at receiver employs another trainable ARN to remove the adversarial attacks and the channel noise. To mitigate the threat of semantic eavesdropping, the trainable ARNs are jointly optimized to minimize the weighted sum of the power of adversarial attack, the mean squared error of semantic communication, and the confidence of eavesdropper correctly retrieving private information. Numerical results show that the proposed scheme is capable of fooling the eavesdropper while maintaining the high-quality semantic communication.



## **18. TTSlow: Slow Down Text-to-Speech with Efficiency Robustness Evaluations**

eess.AS

This work has been submitted to the IEEE for possible publication.  Copyright may be transferred without notice, after which this version may no  longer be accessible

**SubmitDate**: 2024-07-02    [abs](http://arxiv.org/abs/2407.01927v1) [paper-pdf](http://arxiv.org/pdf/2407.01927v1)

**Authors**: Xiaoxue Gao, Yiming Chen, Xianghu Yue, Yu Tsao, Nancy F. Chen

**Abstract**: Text-to-speech (TTS) has been extensively studied for generating high-quality speech with textual inputs, playing a crucial role in various real-time applications. For real-world deployment, ensuring stable and timely generation in TTS models against minor input perturbations is of paramount importance. Therefore, evaluating the robustness of TTS models against such perturbations, commonly known as adversarial attacks, is highly desirable. In this paper, we propose TTSlow, a novel adversarial approach specifically tailored to slow down the speech generation process in TTS systems. To induce long TTS waiting time, we design novel efficiency-oriented adversarial loss to encourage endless generation process. TTSlow encompasses two attack strategies targeting both text inputs and speaker embedding. Specifically, we propose TTSlow-text, which utilizes a combination of homoglyphs-based and swap-based perturbations, along with TTSlow-spk, which employs a gradient optimization attack approach for speaker embedding. TTSlow serves as the first attack approach targeting a wide range of TTS models, including autoregressive and non-autoregressive TTS ones, thereby advancing exploration in audio security. Extensive experiments are conducted to evaluate the inference efficiency of TTS models, and in-depth analysis of generated speech intelligibility is performed using Gemini. The results demonstrate that TTSlow can effectively slow down two TTS models across three publicly available datasets. We are committed to releasing the source code upon acceptance, facilitating further research and benchmarking in this domain.



## **19. Looking From the Future: Multi-order Iterations Can Enhance Adversarial Attack Transferability**

cs.CV

**SubmitDate**: 2024-07-02    [abs](http://arxiv.org/abs/2407.01925v1) [paper-pdf](http://arxiv.org/pdf/2407.01925v1)

**Authors**: Zijian Ying, Qianmu Li, Tao Wang, Zhichao Lian, Shunmei Meng, Xuyun Zhang

**Abstract**: Various methods try to enhance adversarial transferability by improving the generalization from different perspectives. In this paper, we rethink the optimization process and propose a novel sequence optimization concept, which is named Looking From the Future (LFF). LFF makes use of the original optimization process to refine the very first local optimization choice. Adapting the LFF concept to the adversarial attack task, we further propose an LFF attack as well as an MLFF attack with better generalization ability. Furthermore, guiding with the LFF concept, we propose an $LLF^{\mathcal{N}}$ attack which entends the LFF attack to a multi-order attack, further enhancing the transfer attack ability. All our proposed methods can be directly applied to the iteration-based attack methods. We evaluate our proposed method on the ImageNet1k dataset by applying several SOTA adversarial attack methods under four kinds of tasks. Experimental results show that our proposed method can greatly enhance the attack transferability. Ablation experiments are also applied to verify the effectiveness of each component. The source code will be released after this paper is accepted.



## **20. A Method to Facilitate Membership Inference Attacks in Deep Learning Models**

cs.CR

NDSS'25 (a shorter version of this paper will appear in the  conference proceeding)

**SubmitDate**: 2024-07-02    [abs](http://arxiv.org/abs/2407.01919v1) [paper-pdf](http://arxiv.org/pdf/2407.01919v1)

**Authors**: Zitao Chen, Karthik Pattabiraman

**Abstract**: Modern machine learning (ML) ecosystems offer a surging number of ML frameworks and code repositories that can greatly facilitate the development of ML models. Today, even ordinary data holders who are not ML experts can apply off-the-shelf codebase to build high-performance ML models on their data, many of which are sensitive in nature (e.g., clinical records).   In this work, we consider a malicious ML provider who supplies model-training code to the data holders, does not have access to the training process, and has only black-box query access to the resulting model. In this setting, we demonstrate a new form of membership inference attack that is strictly more powerful than prior art. Our attack empowers the adversary to reliably de-identify all the training samples (average >99% attack TPR@0.1% FPR), and the compromised models still maintain competitive performance as their uncorrupted counterparts (average <1% accuracy drop). Moreover, we show that the poisoned models can effectively disguise the amplified membership leakage under common membership privacy auditing, which can only be revealed by a set of secret samples known by the adversary.   Overall, our study not only points to the worst-case membership privacy leakage, but also unveils a common pitfall underlying existing privacy auditing methods, which calls for future efforts to rethink the current practice of auditing membership privacy in machine learning models.



## **21. Sequential Manipulation Against Rank Aggregation: Theory and Algorithm**

cs.AI

Accepted by IEEE TPAMI URL:  https://ieeexplore.ieee.org/document/10564181

**SubmitDate**: 2024-07-02    [abs](http://arxiv.org/abs/2407.01916v1) [paper-pdf](http://arxiv.org/pdf/2407.01916v1)

**Authors**: Ke Ma, Qianqian Xu, Jinshan Zeng, Wei Liu, Xiaochun Cao, Yingfei Sun, Qingming Huang

**Abstract**: Rank aggregation with pairwise comparisons is widely encountered in sociology, politics, economics, psychology, sports, etc . Given the enormous social impact and the consequent incentives, the potential adversary has a strong motivation to manipulate the ranking list. However, the ideal attack opportunity and the excessive adversarial capability cause the existing methods to be impractical. To fully explore the potential risks, we leverage an online attack on the vulnerable data collection process. Since it is independent of rank aggregation and lacks effective protection mechanisms, we disrupt the data collection process by fabricating pairwise comparisons without knowledge of the future data or the true distribution. From the game-theoretic perspective, the confrontation scenario between the online manipulator and the ranker who takes control of the original data source is formulated as a distributionally robust game that deals with the uncertainty of knowledge. Then we demonstrate that the equilibrium in the above game is potentially favorable to the adversary by analyzing the vulnerability of the sampling algorithms such as Bernoulli and reservoir methods. According to the above theoretical analysis, different sequential manipulation policies are proposed under a Bayesian decision framework and a large class of parametric pairwise comparison models. For attackers with complete knowledge, we establish the asymptotic optimality of the proposed policies. To increase the success rate of the sequential manipulation with incomplete knowledge, a distributionally robust estimator, which replaces the maximum likelihood estimation in a saddle point problem, provides a conservative data generation solution. Finally, the corroborating empirical evidence shows that the proposed method manipulates the results of rank aggregation methods in a sequential manner.



## **22. A Curious Case of Searching for the Correlation between Training Data and Adversarial Robustness of Transformer Textual Models**

cs.LG

Accepted to ACL Findings 2024

**SubmitDate**: 2024-07-02    [abs](http://arxiv.org/abs/2402.11469v2) [paper-pdf](http://arxiv.org/pdf/2402.11469v2)

**Authors**: Cuong Dang, Dung D. Le, Thai Le

**Abstract**: Existing works have shown that fine-tuned textual transformer models achieve state-of-the-art prediction performances but are also vulnerable to adversarial text perturbations. Traditional adversarial evaluation is often done \textit{only after} fine-tuning the models and ignoring the training data. In this paper, we want to prove that there is also a strong correlation between training data and model robustness. To this end, we extract 13 different features representing a wide range of input fine-tuning corpora properties and use them to predict the adversarial robustness of the fine-tuned models. Focusing mostly on encoder-only transformer models BERT and RoBERTa with additional results for BART, ELECTRA, and GPT2, we provide diverse evidence to support our argument. First, empirical analyses show that (a) extracted features can be used with a lightweight classifier such as Random Forest to predict the attack success rate effectively, and (b) features with the most influence on the model robustness have a clear correlation with the robustness. Second, our framework can be used as a fast and effective additional tool for robustness evaluation since it (a) saves 30x-193x runtime compared to the traditional technique, (b) is transferable across models, (c) can be used under adversarial training, and (d) robust to statistical randomness. Our code is publicly available at \url{https://github.com/CaptainCuong/RobustText_ACL2024}.



## **23. Purple-teaming LLMs with Adversarial Defender Training**

cs.CL

**SubmitDate**: 2024-07-01    [abs](http://arxiv.org/abs/2407.01850v1) [paper-pdf](http://arxiv.org/pdf/2407.01850v1)

**Authors**: Jingyan Zhou, Kun Li, Junan Li, Jiawen Kang, Minda Hu, Xixin Wu, Helen Meng

**Abstract**: Existing efforts in safeguarding LLMs are limited in actively exposing the vulnerabilities of the target LLM and readily adapting to newly emerging safety risks. To address this, we present Purple-teaming LLMs with Adversarial Defender training (PAD), a pipeline designed to safeguard LLMs by novelly incorporating the red-teaming (attack) and blue-teaming (safety training) techniques. In PAD, we automatically collect conversational data that cover the vulnerabilities of an LLM around specific safety risks in a self-play manner, where the attacker aims to elicit unsafe responses and the defender generates safe responses to these attacks. We then update both modules in a generative adversarial network style by training the attacker to elicit more unsafe responses and updating the defender to identify them and explain the unsafe reason. Experimental results demonstrate that PAD significantly outperforms existing baselines in both finding effective attacks and establishing a robust safe guardrail. Furthermore, our findings indicate that PAD excels in striking a balance between safety and overall model quality. We also reveal key challenges in safeguarding LLMs, including defending multi-turn attacks and the need for more delicate strategies to identify specific risks.



## **24. Adversarial Attacks on Reinforcement Learning Agents for Command and Control**

cs.CR

Accepted to appear in the Journal Of Defense Modeling and Simulation  (JDMS)

**SubmitDate**: 2024-07-01    [abs](http://arxiv.org/abs/2405.01693v2) [paper-pdf](http://arxiv.org/pdf/2405.01693v2)

**Authors**: Ahaan Dabholkar, James Z. Hare, Mark Mittrick, John Richardson, Nicholas Waytowich, Priya Narayanan, Saurabh Bagchi

**Abstract**: Given the recent impact of Deep Reinforcement Learning in training agents to win complex games like StarCraft and DoTA(Defense Of The Ancients) - there has been a surge in research for exploiting learning based techniques for professional wargaming, battlefield simulation and modeling. Real time strategy games and simulators have become a valuable resource for operational planning and military research. However, recent work has shown that such learning based approaches are highly susceptible to adversarial perturbations. In this paper, we investigate the robustness of an agent trained for a Command and Control task in an environment that is controlled by an active adversary. The C2 agent is trained on custom StarCraft II maps using the state of the art RL algorithms - A3C and PPO. We empirically show that an agent trained using these algorithms is highly susceptible to noise injected by the adversary and investigate the effects these perturbations have on the performance of the trained agent. Our work highlights the urgent need to develop more robust training algorithms especially for critical arenas like the battlefield.



## **25. On the Abuse and Detection of Polyglot Files**

cs.CR

18 pages, 11 figures

**SubmitDate**: 2024-07-01    [abs](http://arxiv.org/abs/2407.01529v1) [paper-pdf](http://arxiv.org/pdf/2407.01529v1)

**Authors**: Luke Koch, Sean Oesch, Amul Chaulagain, Jared Dixon, Matthew Dixon, Mike Huettal, Amir Sadovnik, Cory Watson, Brian Weber, Jacob Hartman, Richard Patulski

**Abstract**: A polyglot is a file that is valid in two or more formats. Polyglot files pose a problem for malware detection systems that route files to format-specific detectors/signatures, as well as file upload and sanitization tools. In this work we found that existing file-format and embedded-file detection tools, even those developed specifically for polyglot files, fail to reliably detect polyglot files used in the wild, leaving organizations vulnerable to attack. To address this issue, we studied the use of polyglot files by malicious actors in the wild, finding $30$ polyglot samples and $15$ attack chains that leveraged polyglot files. In this report, we highlight two well-known APTs whose cyber attack chains relied on polyglot files to bypass detection mechanisms. Using knowledge from our survey of polyglot usage in the wild -- the first of its kind -- we created a novel data set based on adversary techniques. We then trained a machine learning detection solution, PolyConv, using this data set. PolyConv achieves a precision-recall area-under-curve score of $0.999$ with an F1 score of $99.20$% for polyglot detection and $99.47$% for file-format identification, significantly outperforming all other tools tested. We developed a content disarmament and reconstruction tool, ImSan, that successfully sanitized $100$% of the tested image-based polyglots, which were the most common type found via the survey. Our work provides concrete tools and suggestions to enable defenders to better defend themselves against polyglot files, as well as directions for future work to create more robust file specifications and methods of disarmament.



## **26. Image-to-Text Logic Jailbreak: Your Imagination can Help You Do Anything**

cs.CR

**SubmitDate**: 2024-07-01    [abs](http://arxiv.org/abs/2407.02534v1) [paper-pdf](http://arxiv.org/pdf/2407.02534v1)

**Authors**: Xiaotian Zou, Yongkang Chen

**Abstract**: Large Visual Language Models (VLMs) such as GPT-4 have achieved remarkable success in generating comprehensive and nuanced responses, surpassing the capabilities of large language models. However, with the integration of visual inputs, new security concerns emerge, as malicious attackers can exploit multiple modalities to achieve their objectives. This has led to increasing attention on the vulnerabilities of VLMs to jailbreak. Most existing research focuses on generating adversarial images or nonsensical image collections to compromise these models. However, the challenge of leveraging meaningful images to produce targeted textual content using the VLMs' logical comprehension of images remains unexplored. In this paper, we explore the problem of logical jailbreak from meaningful images to text. To investigate this issue, we introduce a novel dataset designed to evaluate flowchart image jailbreak. Furthermore, we develop a framework for text-to-text jailbreak using VLMs. Finally, we conduct an extensive evaluation of the framework on GPT-4o and GPT-4-vision-preview, with jailbreak rates of 92.8% and 70.0%, respectively. Our research reveals significant vulnerabilities in current VLMs concerning image-to-text jailbreak. These findings underscore the need for a deeper examination of the security flaws in VLMs before their practical deployment.



## **27. Enhancing the Capability and Robustness of Large Language Models through Reinforcement Learning-Driven Query Refinement**

cs.CL

**SubmitDate**: 2024-07-01    [abs](http://arxiv.org/abs/2407.01461v1) [paper-pdf](http://arxiv.org/pdf/2407.01461v1)

**Authors**: Zisu Huang, Xiaohua Wang, Feiran Zhang, Zhibo Xu, Cenyuan Zhang, Xiaoqing Zheng, Xuanjing Huang

**Abstract**: The capacity of large language models (LLMs) to generate honest, harmless, and helpful responses heavily relies on the quality of user prompts. However, these prompts often tend to be brief and vague, thereby significantly limiting the full potential of LLMs. Moreover, harmful prompts can be meticulously crafted and manipulated by adversaries to jailbreak LLMs, inducing them to produce potentially toxic content. To enhance the capabilities of LLMs while maintaining strong robustness against harmful jailbreak inputs, this study proposes a transferable and pluggable framework that refines user prompts before they are input into LLMs. This strategy improves the quality of the queries, empowering LLMs to generate more truthful, benign and useful responses. Specifically, a lightweight query refinement model is introduced and trained using a specially designed reinforcement learning approach that incorporates multiple objectives to enhance particular capabilities of LLMs. Extensive experiments demonstrate that the refinement model not only improves the quality of responses but also strengthens their robustness against jailbreak attacks. Code is available at: https://github.com/Huangzisu/query-refinement .



## **28. Cutting through buggy adversarial example defenses: fixing 1 line of code breaks Sabre**

cs.CR

**SubmitDate**: 2024-07-01    [abs](http://arxiv.org/abs/2405.03672v3) [paper-pdf](http://arxiv.org/pdf/2405.03672v3)

**Authors**: Nicholas Carlini

**Abstract**: Sabre is a defense to adversarial examples that was accepted at IEEE S&P 2024. We first reveal significant flaws in the evaluation that point to clear signs of gradient masking. We then show the cause of this gradient masking: a bug in the original evaluation code. By fixing a single line of code in the original repository, we reduce Sabre's robust accuracy to 0%. In response to this, the authors modify the defense and introduce a new defense component not described in the original paper. But this fix contains a second bug; modifying one more line of code reduces robust accuracy to below baseline levels. After we released the first version of our paper online, the authors introduced another change to the defense; by commenting out one line of code during attack we reduce the robust accuracy to 0% again.



## **29. Jailbreak Vision Language Models via Bi-Modal Adversarial Prompt**

cs.CV

**SubmitDate**: 2024-07-01    [abs](http://arxiv.org/abs/2406.04031v2) [paper-pdf](http://arxiv.org/pdf/2406.04031v2)

**Authors**: Zonghao Ying, Aishan Liu, Tianyuan Zhang, Zhengmin Yu, Siyuan Liang, Xianglong Liu, Dacheng Tao

**Abstract**: In the realm of large vision language models (LVLMs), jailbreak attacks serve as a red-teaming approach to bypass guardrails and uncover safety implications. Existing jailbreaks predominantly focus on the visual modality, perturbing solely visual inputs in the prompt for attacks. However, they fall short when confronted with aligned models that fuse visual and textual features simultaneously for generation. To address this limitation, this paper introduces the Bi-Modal Adversarial Prompt Attack (BAP), which executes jailbreaks by optimizing textual and visual prompts cohesively. Initially, we adversarially embed universally harmful perturbations in an image, guided by a few-shot query-agnostic corpus (e.g., affirmative prefixes and negative inhibitions). This process ensures that image prompt LVLMs to respond positively to any harmful queries. Subsequently, leveraging the adversarial image, we optimize textual prompts with specific harmful intent. In particular, we utilize a large language model to analyze jailbreak failures and employ chain-of-thought reasoning to refine textual prompts through a feedback-iteration manner. To validate the efficacy of our approach, we conducted extensive evaluations on various datasets and LVLMs, demonstrating that our method significantly outperforms other methods by large margins (+29.03% in attack success rate on average). Additionally, we showcase the potential of our attacks on black-box commercial LVLMs, such as Gemini and ChatGLM.



## **30. Formal Verification of Object Detection**

cs.CV

**SubmitDate**: 2024-07-01    [abs](http://arxiv.org/abs/2407.01295v1) [paper-pdf](http://arxiv.org/pdf/2407.01295v1)

**Authors**: Avraham Raviv, Yizhak Y. Elboher, Michelle Aluf-Medina, Yael Leibovich Weiss, Omer Cohen, Roy Assa, Guy Katz, Hillel Kugler

**Abstract**: Deep Neural Networks (DNNs) are ubiquitous in real-world applications, yet they remain vulnerable to errors and adversarial attacks. This work tackles the challenge of applying formal verification to ensure the safety of computer vision models, extending verification beyond image classification to object detection. We propose a general formulation for certifying the robustness of object detection models using formal verification and outline implementation strategies compatible with state-of-the-art verification tools. Our approach enables the application of these tools, originally designed for verifying classification models, to object detection. We define various attacks for object detection, illustrating the diverse ways adversarial inputs can compromise neural network outputs. Our experiments, conducted on several common datasets and networks, reveal potential errors in object detection models, highlighting system vulnerabilities and emphasizing the need for expanding formal verification to these new domains. This work paves the way for further research in integrating formal verification across a broader range of computer vision applications.



## **31. DeepiSign-G: Generic Watermark to Stamp Hidden DNN Parameters for Self-contained Tracking**

cs.CR

13 pages

**SubmitDate**: 2024-07-01    [abs](http://arxiv.org/abs/2407.01260v1) [paper-pdf](http://arxiv.org/pdf/2407.01260v1)

**Authors**: Alsharif Abuadbba, Nicholas Rhodes, Kristen Moore, Bushra Sabir, Shuo Wang, Yansong Gao

**Abstract**: Deep learning solutions in critical domains like autonomous vehicles, facial recognition, and sentiment analysis require caution due to the severe consequences of errors. Research shows these models are vulnerable to adversarial attacks, such as data poisoning and neural trojaning, which can covertly manipulate model behavior, compromising reliability and safety. Current defense strategies like watermarking have limitations: they fail to detect all model modifications and primarily focus on attacks on CNNs in the image domain, neglecting other critical architectures like RNNs.   To address these gaps, we introduce DeepiSign-G, a versatile watermarking approach designed for comprehensive verification of leading DNN architectures, including CNNs and RNNs. DeepiSign-G enhances model security by embedding an invisible watermark within the Walsh-Hadamard transform coefficients of the model's parameters. This watermark is highly sensitive and fragile, ensuring prompt detection of any modifications. Unlike traditional hashing techniques, DeepiSign-G allows substantial metadata incorporation directly within the model, enabling detailed, self-contained tracking and verification.   We demonstrate DeepiSign-G's applicability across various architectures, including CNN models (VGG, ResNets, DenseNet) and RNNs (Text sentiment classifier). We experiment with four popular datasets: VGG Face, CIFAR10, GTSRB Traffic Sign, and Large Movie Review. We also evaluate DeepiSign-G under five potential attacks. Our comprehensive evaluation confirms that DeepiSign-G effectively detects these attacks without compromising CNN and RNN model performance, highlighting its efficacy as a robust security measure for deep learning applications. Detection of integrity breaches is nearly perfect, while hiding only a bit in approximately 1% of the Walsh-Hadamard coefficients.



## **32. QUEEN: Query Unlearning against Model Extraction**

cs.CR

**SubmitDate**: 2024-07-01    [abs](http://arxiv.org/abs/2407.01251v1) [paper-pdf](http://arxiv.org/pdf/2407.01251v1)

**Authors**: Huajie Chen, Tianqing Zhu, Lefeng Zhang, Bo Liu, Derui Wang, Wanlei Zhou, Minhui Xue

**Abstract**: Model extraction attacks currently pose a non-negligible threat to the security and privacy of deep learning models. By querying the model with a small dataset and usingthe query results as the ground-truth labels, an adversary can steal a piracy model with performance comparable to the original model. Two key issues that cause the threat are, on the one hand, accurate and unlimited queries can be obtained by the adversary; on the other hand, the adversary can aggregate the query results to train the model step by step. The existing defenses usually employ model watermarking or fingerprinting to protect the ownership. However, these methods cannot proactively prevent the violation from happening. To mitigate the threat, we propose QUEEN (QUEry unlEarNing) that proactively launches counterattacks on potential model extraction attacks from the very beginning. To limit the potential threat, QUEEN has sensitivity measurement and outputs perturbation that prevents the adversary from training a piracy model with high performance. In sensitivity measurement, QUEEN measures the single query sensitivity by its distance from the center of its cluster in the feature space. To reduce the learning accuracy of attacks, for the highly sensitive query batch, QUEEN applies query unlearning, which is implemented by gradient reverse to perturb the softmax output such that the piracy model will generate reverse gradients to worsen its performance unconsciously. Experiments show that QUEEN outperforms the state-of-the-art defenses against various model extraction attacks with a relatively low cost to the model accuracy. The artifact is publicly available at https://anonymous.4open.science/r/queen implementation-5408/.



## **33. Multi-View Black-Box Physical Attacks on Infrared Pedestrian Detectors Using Adversarial Infrared Grid**

cs.CV

**SubmitDate**: 2024-07-01    [abs](http://arxiv.org/abs/2407.01168v1) [paper-pdf](http://arxiv.org/pdf/2407.01168v1)

**Authors**: Kalibinuer Tiliwalidi, Chengyin Hu, Weiwen Shi

**Abstract**: While extensive research exists on physical adversarial attacks within the visible spectrum, studies on such techniques in the infrared spectrum are limited. Infrared object detectors are vital in modern technological applications but are susceptible to adversarial attacks, posing significant security threats. Previous studies using physical perturbations like light bulb arrays and aerogels for white-box attacks, or hot and cold patches for black-box attacks, have proven impractical or limited in multi-view support. To address these issues, we propose the Adversarial Infrared Grid (AdvGrid), which models perturbations in a grid format and uses a genetic algorithm for black-box optimization. These perturbations are cyclically applied to various parts of a pedestrian's clothing to facilitate multi-view black-box physical attacks on infrared pedestrian detectors. Extensive experiments validate AdvGrid's effectiveness, stealthiness, and robustness. The method achieves attack success rates of 80.00\% in digital environments and 91.86\% in physical environments, outperforming baseline methods. Additionally, the average attack success rate exceeds 50\% against mainstream detectors, demonstrating AdvGrid's robustness. Our analyses include ablation studies, transfer attacks, and adversarial defenses, confirming the method's superiority.



## **34. Unaligning Everything: Or Aligning Any Text to Any Image in Multimodal Models**

cs.CV

14 pages, 14 figures. arXiv admin note: substantial text overlap with  arXiv:2401.15568, arXiv:2402.08473

**SubmitDate**: 2024-07-01    [abs](http://arxiv.org/abs/2407.01157v1) [paper-pdf](http://arxiv.org/pdf/2407.01157v1)

**Authors**: Shaeke Salman, Md Montasir Bin Shams, Xiuwen Liu

**Abstract**: Utilizing a shared embedding space, emerging multimodal models exhibit unprecedented zero-shot capabilities. However, the shared embedding space could lead to new vulnerabilities if different modalities can be misaligned. In this paper, we extend and utilize a recently developed effective gradient-based procedure that allows us to match the embedding of a given text by minimally modifying an image. Using the procedure, we show that we can align the embeddings of distinguishable texts to any image through unnoticeable adversarial attacks in joint image-text models, revealing that semantically unrelated images can have embeddings of identical texts and at the same time visually indistinguishable images can be matched to the embeddings of very different texts. Our technique achieves 100\% success rate when it is applied to text datasets and images from multiple sources. Without overcoming the vulnerability, multimodal models cannot robustly align inputs from different modalities in a semantically meaningful way. \textbf{Warning: the text data used in this paper are toxic in nature and may be offensive to some readers.}



## **35. SecGenAI: Enhancing Security of Cloud-based Generative AI Applications within Australian Critical Technologies of National Interest**

cs.CR

10 pages, 4 figures, 9 tables, submitted to the 2024 11th  International Conference on Soft Computing & Machine Intelligence (ISCMI  2024)

**SubmitDate**: 2024-07-01    [abs](http://arxiv.org/abs/2407.01110v1) [paper-pdf](http://arxiv.org/pdf/2407.01110v1)

**Authors**: Christoforus Yoga Haryanto, Minh Hieu Vu, Trung Duc Nguyen, Emily Lomempow, Yulia Nurliana, Sona Taheri

**Abstract**: The rapid advancement of Generative AI (GenAI) technologies offers transformative opportunities within Australia's critical technologies of national interest while introducing unique security challenges. This paper presents SecGenAI, a comprehensive security framework for cloud-based GenAI applications, with a focus on Retrieval-Augmented Generation (RAG) systems. SecGenAI addresses functional, infrastructure, and governance requirements, integrating end-to-end security analysis to generate specifications emphasizing data privacy, secure deployment, and shared responsibility models. Aligned with Australian Privacy Principles, AI Ethics Principles, and guidelines from the Australian Cyber Security Centre and Digital Transformation Agency, SecGenAI mitigates threats such as data leakage, adversarial attacks, and model inversion. The framework's novel approach combines advanced machine learning techniques with robust security measures, ensuring compliance with Australian regulations while enhancing the reliability and trustworthiness of GenAI systems. This research contributes to the field of intelligent systems by providing actionable strategies for secure GenAI implementation in industry, fostering innovation in AI applications, and safeguarding national interests.



## **36. DifAttack++: Query-Efficient Black-Box Adversarial Attack via Hierarchical Disentangled Feature Space in Cross-Domain**

cs.CV

arXiv admin note: substantial text overlap with arXiv:2309.14585 An  extension of the AAAI24 paper "DifAttack: Query-Efficient Black-Box Attack  via Disentangled Feature Space."

**SubmitDate**: 2024-07-01    [abs](http://arxiv.org/abs/2406.03017v3) [paper-pdf](http://arxiv.org/pdf/2406.03017v3)

**Authors**: Jun Liu, Jiantao Zhou, Jiandian Zeng, Jinyu Tian, Zheng Li

**Abstract**: This work investigates efficient score-based black-box adversarial attacks with a high Attack Success Rate (\textbf{ASR}) and good generalizability. We design a novel attack method based on a hierarchical DIsentangled Feature space, called \textbf{DifAttack++}, which differs significantly from the existing ones operating over the entire feature space. Specifically, DifAttack++ firstly disentangles an image's latent feature into an Adversarial Feature (\textbf{AF}) and a Visual Feature (\textbf{VF}) via an autoencoder equipped with our specially designed Hierarchical Decouple-Fusion (\textbf{HDF}) module, where the AF dominates the adversarial capability of an image, while the VF largely determines its visual appearance. We train such two autoencoders for the clean and adversarial image domains (i.e., cross-domain) respectively to achieve image reconstructions and feature disentanglement, by using pairs of clean images and their Adversarial Examples (\textbf{AE}s) generated from available surrogate models via white-box attack methods. Eventually, in the black-box attack stage, DifAttack++ iteratively optimizes the AF according to the query feedback from the victim model until a successful AE is generated, while keeping the VF unaltered. Extensive experimental results demonstrate that our DifAttack++ leads to superior ASR and query efficiency than state-of-the-art methods, meanwhile exhibiting much better visual quality of AEs. The code is available at https://github.com/csjunjun/DifAttack.git.



## **37. Time-Frequency Jointed Imperceptible Adversarial Attack to Brainprint Recognition with Deep Learning Models**

cs.CR

This work is accepted by ICME 2024

**SubmitDate**: 2024-07-01    [abs](http://arxiv.org/abs/2403.10021v3) [paper-pdf](http://arxiv.org/pdf/2403.10021v3)

**Authors**: Hangjie Yi, Yuhang Ming, Dongjun Liu, Wanzeng Kong

**Abstract**: EEG-based brainprint recognition with deep learning models has garnered much attention in biometric identification. Yet, studies have indicated vulnerability to adversarial attacks in deep learning models with EEG inputs. In this paper, we introduce a novel adversarial attack method that jointly attacks time-domain and frequency-domain EEG signals by employing wavelet transform. Different from most existing methods which only target time-domain EEG signals, our method not only takes advantage of the time-domain attack's potent adversarial strength but also benefits from the imperceptibility inherent in frequency-domain attack, achieving a better balance between attack performance and imperceptibility. Extensive experiments are conducted in both white- and grey-box scenarios and the results demonstrate that our attack method achieves state-of-the-art attack performance on three datasets and three deep-learning models. In the meanwhile, the perturbations in the signals attacked by our method are barely perceptible to the human visual system.



## **38. Learning Robust 3D Representation from CLIP via Dual Denoising**

cs.CV

**SubmitDate**: 2024-07-01    [abs](http://arxiv.org/abs/2407.00905v1) [paper-pdf](http://arxiv.org/pdf/2407.00905v1)

**Authors**: Shuqing Luo, Bowen Qu, Wei Gao

**Abstract**: In this paper, we explore a critical yet under-investigated issue: how to learn robust and well-generalized 3D representation from pre-trained vision language models such as CLIP. Previous works have demonstrated that cross-modal distillation can provide rich and useful knowledge for 3D data. However, like most deep learning models, the resultant 3D learning network is still vulnerable to adversarial attacks especially the iterative attack. In this work, we propose Dual Denoising, a novel framework for learning robust and well-generalized 3D representations from CLIP. It combines a denoising-based proxy task with a novel feature denoising network for 3D pre-training. Additionally, we propose utilizing parallel noise inference to enhance the generalization of point cloud features under cross domain settings. Experiments show that our model can effectively improve the representation learning performance and adversarial robustness of the 3D learning network under zero-shot settings without adversarial training. Our code is available at https://github.com/luoshuqing2001/Dual_Denoising.



## **39. GRACE: Graph-Regularized Attentive Convolutional Entanglement with Laplacian Smoothing for Robust DeepFake Video Detection**

cs.CV

Submitted to TPAMI 2024

**SubmitDate**: 2024-07-01    [abs](http://arxiv.org/abs/2406.19941v2) [paper-pdf](http://arxiv.org/pdf/2406.19941v2)

**Authors**: Chih-Chung Hsu, Shao-Ning Chen, Mei-Hsuan Wu, Yi-Fang Wang, Chia-Ming Lee, Yi-Shiuan Chou

**Abstract**: As DeepFake video manipulation techniques escalate, posing profound threats, the urgent need to develop efficient detection strategies is underscored. However, one particular issue lies with facial images being mis-detected, often originating from degraded videos or adversarial attacks, leading to unexpected temporal artifacts that can undermine the efficacy of DeepFake video detection techniques. This paper introduces a novel method for robust DeepFake video detection, harnessing the power of the proposed Graph-Regularized Attentive Convolutional Entanglement (GRACE) based on the graph convolutional network with graph Laplacian to address the aforementioned challenges. First, conventional Convolution Neural Networks are deployed to perform spatiotemporal features for the entire video. Then, the spatial and temporal features are mutually entangled by constructing a graph with sparse constraint, enforcing essential features of valid face images in the noisy face sequences remaining, thus augmenting stability and performance for DeepFake video detection. Furthermore, the Graph Laplacian prior is proposed in the graph convolutional network to remove the noise pattern in the feature space to further improve the performance. Comprehensive experiments are conducted to illustrate that our proposed method delivers state-of-the-art performance in DeepFake video detection under noisy face sequences. The source code is available at https://github.com/ming053l/GRACE.



## **40. A Two-Layer Blockchain Sharding Protocol Leveraging Safety and Liveness for Enhanced Performance**

cs.CR

The paper has been accepted to Network and Distributed System  Security (NDSS) Symposium 2024

**SubmitDate**: 2024-07-01    [abs](http://arxiv.org/abs/2310.11373v4) [paper-pdf](http://arxiv.org/pdf/2310.11373v4)

**Authors**: Yibin Xu, Jingyi Zheng, Boris Düdder, Tijs Slaats, Yongluan Zhou

**Abstract**: Sharding is essential for improving blockchain scalability. Existing protocols overlook diverse adversarial attacks, limiting transaction throughput. This paper presents Reticulum, a groundbreaking sharding protocol addressing this issue, boosting blockchain scalability.   Reticulum employs a two-phase approach, adapting transaction throughput based on runtime adversarial attacks. It comprises "control" and "process" shards in two layers. Process shards contain at least one trustworthy node, while control shards have a majority of trusted nodes. In the first phase, transactions are written to blocks and voted on by nodes in process shards. Unanimously accepted blocks are confirmed. In the second phase, blocks without unanimous acceptance are voted on by control shards. Blocks are accepted if the majority votes in favor, eliminating first-phase opponents and silent voters. Reticulum uses unanimous voting in the first phase, involving fewer nodes, enabling more parallel process shards. Control shards finalize decisions and resolve disputes.   Experiments confirm Reticulum's innovative design, providing high transaction throughput and robustness against various network attacks, outperforming existing sharding protocols for blockchain networks.



## **41. Fortify the Guardian, Not the Treasure: Resilient Adversarial Detectors**

cs.CV

**SubmitDate**: 2024-06-30    [abs](http://arxiv.org/abs/2404.12120v2) [paper-pdf](http://arxiv.org/pdf/2404.12120v2)

**Authors**: Raz Lapid, Almog Dubin, Moshe Sipper

**Abstract**: This paper presents RADAR-Robust Adversarial Detection via Adversarial Retraining-an approach designed to enhance the robustness of adversarial detectors against adaptive attacks, while maintaining classifier performance. An adaptive attack is one where the attacker is aware of the defenses and adapts their strategy accordingly. Our proposed method leverages adversarial training to reinforce the ability to detect attacks, without compromising clean accuracy. During the training phase, we integrate into the dataset adversarial examples, which were optimized to fool both the classifier and the adversarial detector, enabling the adversarial detector to learn and adapt to potential attack scenarios. Experimental evaluations on the CIFAR-10 and SVHN datasets demonstrate that our proposed algorithm significantly improves a detector's ability to accurately identify adaptive adversarial attacks -- without sacrificing clean accuracy.



## **42. Query-Efficient Hard-Label Black-Box Attack against Vision Transformers**

cs.CV

**SubmitDate**: 2024-06-29    [abs](http://arxiv.org/abs/2407.00389v1) [paper-pdf](http://arxiv.org/pdf/2407.00389v1)

**Authors**: Chao Zhou, Xiaowen Shi, Yuan-Gen Wang

**Abstract**: Recent studies have revealed that vision transformers (ViTs) face similar security risks from adversarial attacks as deep convolutional neural networks (CNNs). However, directly applying attack methodology on CNNs to ViTs has been demonstrated to be ineffective since the ViTs typically work on patch-wise encoding. This article explores the vulnerability of ViTs against adversarial attacks under a black-box scenario, and proposes a novel query-efficient hard-label adversarial attack method called AdvViT. Specifically, considering that ViTs are highly sensitive to patch modification, we propose to optimize the adversarial perturbation on the individual patches. To reduce the dimension of perturbation search space, we modify only a handful of low-frequency components of each patch. Moreover, we design a weight mask matrix for all patches to further optimize the perturbation on different regions of a whole image. We test six mainstream ViT backbones on the ImageNet-1k dataset. Experimental results show that compared with the state-of-the-art attacks on CNNs, our AdvViT achieves much lower $L_2$-norm distortion under the same query budget, sufficiently validating the vulnerability of ViTs against adversarial attacks.



## **43. DiffuseDef: Improved Robustness to Adversarial Attacks**

cs.CL

**SubmitDate**: 2024-06-28    [abs](http://arxiv.org/abs/2407.00248v1) [paper-pdf](http://arxiv.org/pdf/2407.00248v1)

**Authors**: Zhenhao Li, Marek Rei, Lucia Specia

**Abstract**: Pretrained language models have significantly advanced performance across various natural language processing tasks. However, adversarial attacks continue to pose a critical challenge to system built using these models, as they can be exploited with carefully crafted adversarial texts. Inspired by the ability of diffusion models to predict and reduce noise in computer vision, we propose a novel and flexible adversarial defense method for language classification tasks, DiffuseDef, which incorporates a diffusion layer as a denoiser between the encoder and the classifier. During inference, the adversarial hidden state is first combined with sampled noise, then denoised iteratively and finally ensembled to produce a robust text representation. By integrating adversarial training, denoising, and ensembling techniques, we show that DiffuseDef improves over different existing adversarial defense methods and achieves state-of-the-art performance against common adversarial attacks.



## **44. Deciphering the Definition of Adversarial Robustness for post-hoc OOD Detectors**

cs.CR

**SubmitDate**: 2024-06-28    [abs](http://arxiv.org/abs/2406.15104v3) [paper-pdf](http://arxiv.org/pdf/2406.15104v3)

**Authors**: Peter Lorenz, Mario Fernandez, Jens Müller, Ullrich Köthe

**Abstract**: Detecting out-of-distribution (OOD) inputs is critical for safely deploying deep learning models in real-world scenarios. In recent years, many OOD detectors have been developed, and even the benchmarking has been standardized, i.e. OpenOOD. The number of post-hoc detectors is growing fast and showing an option to protect a pre-trained classifier against natural distribution shifts, claiming to be ready for real-world scenarios. However, its efficacy in handling adversarial examples has been neglected in the majority of studies. This paper investigates the adversarial robustness of the 16 post-hoc detectors on several evasion attacks and discuss a roadmap towards adversarial defense in OOD detectors.



## **45. Stackelberg Games with $k$-Submodular Function under Distributional Risk-Receptiveness and Robustness**

math.OC

**SubmitDate**: 2024-06-28    [abs](http://arxiv.org/abs/2406.13023v3) [paper-pdf](http://arxiv.org/pdf/2406.13023v3)

**Authors**: Seonghun Park, Manish Bansal

**Abstract**: We study submodular optimization in adversarial context, applicable to machine learning problems such as feature selection using data susceptible to uncertainties and attacks. We focus on Stackelberg games between an attacker (or interdictor) and a defender where the attacker aims to minimize the defender's objective of maximizing a $k$-submodular function. We allow uncertainties arising from the success of attacks and inherent data noise, and address challenges due to incomplete knowledge of the probability distribution of random parameters. Specifically, we introduce Distributionally Risk-Averse $k$-Submodular Interdiction Problem (DRA $k$-SIP) and Distributionally Risk-Receptive $k$-Submodular Interdiction Problem (DRR $k$-SIP) along with finitely convergent exact algorithms for solving them. The DRA $k$-SIP solution allows risk-averse interdictor to develop robust strategies for real-world uncertainties. Conversely, DRR $k$-SIP solution suggests aggressive tactics for attackers, willing to embrace (distributional) risk to inflict maximum damage, identifying critical vulnerable components, which can be used for the defender's defensive strategies. The optimal values derived from both DRA $k$-SIP and DRR $k$-SIP offer a confidence interval-like range for the expected value of the defender's objective function, capturing distributional ambiguity. We conduct computational experiments using instances of feature selection and sensor placement problems, and Wisconsin breast cancer data and synthetic data, respectively.



## **46. Emotion Loss Attacking: Adversarial Attack Perception for Skeleton based on Multi-dimensional Features**

cs.CV

**SubmitDate**: 2024-06-28    [abs](http://arxiv.org/abs/2406.19815v1) [paper-pdf](http://arxiv.org/pdf/2406.19815v1)

**Authors**: Feng Liu, Qing Xu, Qijian Zheng

**Abstract**: Adversarial attack on skeletal motion is a hot topic. However, existing researches only consider part of dynamic features when measuring distance between skeleton graph sequences, which results in poor imperceptibility. To this end, we propose a novel adversarial attack method to attack action recognizers for skeletal motions. Firstly, our method systematically proposes a dynamic distance function to measure the difference between skeletal motions. Meanwhile, we innovatively introduce emotional features for complementary information. In addition, we use Alternating Direction Method of Multipliers(ADMM) to solve the constrained optimization problem, which generates adversarial samples with better imperceptibility to deceive the classifiers. Experiments show that our method is effective on multiple action classifiers and datasets. When the perturbation magnitude measured by l norms is the same, the dynamic perturbations generated by our method are much lower than that of other methods. What's more, we are the first to prove the effectiveness of emotional features, and provide a new idea for measuring the distance between skeletal motions.



## **47. Deceptive Diffusion: Generating Synthetic Adversarial Examples**

cs.LG

**SubmitDate**: 2024-06-28    [abs](http://arxiv.org/abs/2406.19807v1) [paper-pdf](http://arxiv.org/pdf/2406.19807v1)

**Authors**: Lucas Beerens, Catherine F. Higham, Desmond J. Higham

**Abstract**: We introduce the concept of deceptive diffusion -- training a generative AI model to produce adversarial images. Whereas a traditional adversarial attack algorithm aims to perturb an existing image to induce a misclassificaton, the deceptive diffusion model can create an arbitrary number of new, misclassified images that are not directly associated with training or test images. Deceptive diffusion offers the possibility of strengthening defence algorithms by providing adversarial training data at scale, including types of misclassification that are otherwise difficult to find. In our experiments, we also investigate the effect of training on a partially attacked data set. This highlights a new type of vulnerability for generative diffusion models: if an attacker is able to stealthily poison a portion of the training data, then the resulting diffusion model will generate a similar proportion of misleading outputs.



## **48. Backdoor Attack in Prompt-Based Continual Learning**

cs.LG

**SubmitDate**: 2024-06-28    [abs](http://arxiv.org/abs/2406.19753v1) [paper-pdf](http://arxiv.org/pdf/2406.19753v1)

**Authors**: Trang Nguyen, Anh Tran, Nhat Ho

**Abstract**: Prompt-based approaches offer a cutting-edge solution to data privacy issues in continual learning, particularly in scenarios involving multiple data suppliers where long-term storage of private user data is prohibited. Despite delivering state-of-the-art performance, its impressive remembering capability can become a double-edged sword, raising security concerns as it might inadvertently retain poisoned knowledge injected during learning from private user data. Following this insight, in this paper, we expose continual learning to a potential threat: backdoor attack, which drives the model to follow a desired adversarial target whenever a specific trigger is present while still performing normally on clean samples. We highlight three critical challenges in executing backdoor attacks on incremental learners and propose corresponding solutions: (1) \emph{Transferability}: We employ a surrogate dataset and manipulate prompt selection to transfer backdoor knowledge to data from other suppliers; (2) \emph{Resiliency}: We simulate static and dynamic states of the victim to ensure the backdoor trigger remains robust during intense incremental learning processes; and (3) \emph{Authenticity}: We apply binary cross-entropy loss as an anti-cheating factor to prevent the backdoor trigger from devolving into adversarial noise. Extensive experiments across various benchmark datasets and continual learners validate our continual backdoor framework, achieving up to $100\%$ attack success rate, with further ablation studies confirming our contributions' effectiveness.



## **49. IDT: Dual-Task Adversarial Attacks for Privacy Protection**

cs.CL

28 pages, 1 figure

**SubmitDate**: 2024-06-28    [abs](http://arxiv.org/abs/2406.19642v1) [paper-pdf](http://arxiv.org/pdf/2406.19642v1)

**Authors**: Pedro Faustini, Shakila Mahjabin Tonni, Annabelle McIver, Qiongkai Xu, Mark Dras

**Abstract**: Natural language processing (NLP) models may leak private information in different ways, including membership inference, reconstruction or attribute inference attacks. Sensitive information may not be explicit in the text, but hidden in underlying writing characteristics. Methods to protect privacy can involve using representations inside models that are demonstrated not to detect sensitive attributes or -- for instance, in cases where users might not trust a model, the sort of scenario of interest here -- changing the raw text before models can have access to it. The goal is to rewrite text to prevent someone from inferring a sensitive attribute (e.g. the gender of the author, or their location by the writing style) whilst keeping the text useful for its original intention (e.g. the sentiment of a product review). The few works tackling this have focused on generative techniques. However, these often create extensively different texts from the original ones or face problems such as mode collapse. This paper explores a novel adaptation of adversarial attack techniques to manipulate a text to deceive a classifier w.r.t one task (privacy) whilst keeping the predictions of another classifier trained for another task (utility) unchanged. We propose IDT, a method that analyses predictions made by auxiliary and interpretable models to identify which tokens are important to change for the privacy task, and which ones should be kept for the utility task. We evaluate different datasets for NLP suitable for different tasks. Automatic and human evaluations show that IDT retains the utility of text, while also outperforming existing methods when deceiving a classifier w.r.t privacy task.



## **50. Data-Driven Lipschitz Continuity: A Cost-Effective Approach to Improve Adversarial Robustness**

cs.LG

**SubmitDate**: 2024-06-28    [abs](http://arxiv.org/abs/2406.19622v1) [paper-pdf](http://arxiv.org/pdf/2406.19622v1)

**Authors**: Erh-Chung Chen, Pin-Yu Chen, I-Hsin Chung, Che-Rung Lee

**Abstract**: The security and robustness of deep neural networks (DNNs) have become increasingly concerning. This paper aims to provide both a theoretical foundation and a practical solution to ensure the reliability of DNNs. We explore the concept of Lipschitz continuity to certify the robustness of DNNs against adversarial attacks, which aim to mislead the network with adding imperceptible perturbations into inputs. We propose a novel algorithm that remaps the input domain into a constrained range, reducing the Lipschitz constant and potentially enhancing robustness. Unlike existing adversarially trained models, where robustness is enhanced by introducing additional examples from other datasets or generative models, our method is almost cost-free as it can be integrated with existing models without requiring re-training. Experimental results demonstrate the generalizability of our method, as it can be combined with various models and achieve enhancements in robustness. Furthermore, our method achieves the best robust accuracy for CIFAR10, CIFAR100, and ImageNet datasets on the RobustBench leaderboard.



