# Latest Adversarial Attack Papers
**update at 2024-11-06 17:29:05**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. Introducing Perturb-ability Score (PS) to Enhance Robustness Against Evasion Adversarial Attacks on ML-NIDS**

cs.CR

**SubmitDate**: 2024-11-05    [abs](http://arxiv.org/abs/2409.07448v2) [paper-pdf](http://arxiv.org/pdf/2409.07448v2)

**Authors**: Mohamed elShehaby, Ashraf Matrawy

**Abstract**: As network security threats continue to evolve, safeguarding Machine Learning (ML)-based Network Intrusion Detection Systems (NIDS) from adversarial attacks is crucial. This paper introduces the notion of feature perturb-ability and presents a novel Perturb-ability Score (PS) metric that identifies NIDS features susceptible to manipulation in the problem-space by an attacker. By quantifying a feature's susceptibility to perturbations within the problem-space, the PS facilitates the selection of features that are inherently more robust against evasion adversarial attacks on ML-NIDS during the feature selection phase. These features exhibit natural resilience to perturbations, as they are heavily constrained by the problem-space limitations and correlations of the NIDS domain. Furthermore, manipulating these features may either disrupt the malicious function of evasion adversarial attacks on NIDS or render the network traffic invalid for processing (or both). This proposed novel approach employs a fresh angle by leveraging network domain constraints as a defense mechanism against problem-space evasion adversarial attacks targeting ML-NIDS. We demonstrate the effectiveness of our PS-guided feature selection defense in enhancing NIDS robustness. Experimental results across various ML-based NIDS models and public datasets show that selecting only robust features (low-PS features) can maintain solid detection performance while significantly reducing vulnerability to evasion adversarial attacks. Additionally, our findings verify that the PS effectively identifies NIDS features highly vulnerable to problem-space perturbations.



## **2. Oblivious Defense in ML Models: Backdoor Removal without Detection**

cs.LG

**SubmitDate**: 2024-11-05    [abs](http://arxiv.org/abs/2411.03279v1) [paper-pdf](http://arxiv.org/pdf/2411.03279v1)

**Authors**: Shafi Goldwasser, Jonathan Shafer, Neekon Vafa, Vinod Vaikuntanathan

**Abstract**: As society grows more reliant on machine learning, ensuring the security of machine learning systems against sophisticated attacks becomes a pressing concern. A recent result of Goldwasser, Kim, Vaikuntanathan, and Zamir (2022) shows that an adversary can plant undetectable backdoors in machine learning models, allowing the adversary to covertly control the model's behavior. Backdoors can be planted in such a way that the backdoored machine learning model is computationally indistinguishable from an honest model without backdoors.   In this paper, we present strategies for defending against backdoors in ML models, even if they are undetectable. The key observation is that it is sometimes possible to provably mitigate or even remove backdoors without needing to detect them, using techniques inspired by the notion of random self-reducibility. This depends on properties of the ground-truth labels (chosen by nature), and not of the proposed ML model (which may be chosen by an attacker).   We give formal definitions for secure backdoor mitigation, and proceed to show two types of results. First, we show a "global mitigation" technique, which removes all backdoors from a machine learning model under the assumption that the ground-truth labels are close to a Fourier-heavy function. Second, we consider distributions where the ground-truth labels are close to a linear or polynomial function in $\mathbb{R}^n$. Here, we show "local mitigation" techniques, which remove backdoors with high probability for every inputs of interest, and are computationally cheaper than global mitigation. All of our constructions are black-box, so our techniques work without needing access to the model's representation (i.e., its code or parameters). Along the way we prove a simple result for robust mean estimation.



## **3. Formal Logic-guided Robust Federated Learning against Poisoning Attacks**

cs.CR

arXiv admin note: text overlap with arXiv:2305.00328 by other authors

**SubmitDate**: 2024-11-05    [abs](http://arxiv.org/abs/2411.03231v1) [paper-pdf](http://arxiv.org/pdf/2411.03231v1)

**Authors**: Dung Thuy Nguyen, Ziyan An, Taylor T. Johnson, Meiyi Ma, Kevin Leach

**Abstract**: Federated Learning (FL) offers a promising solution to the privacy concerns associated with centralized Machine Learning (ML) by enabling decentralized, collaborative learning. However, FL is vulnerable to various security threats, including poisoning attacks, where adversarial clients manipulate the training data or model updates to degrade overall model performance. Recognizing this threat, researchers have focused on developing defense mechanisms to counteract poisoning attacks in FL systems. However, existing robust FL methods predominantly focus on computer vision tasks, leaving a gap in addressing the unique challenges of FL with time series data. In this paper, we present FLORAL, a defense mechanism designed to mitigate poisoning attacks in federated learning for time-series tasks, even in scenarios with heterogeneous client data and a large number of adversarial participants. Unlike traditional model-centric defenses, FLORAL leverages logical reasoning to evaluate client trustworthiness by aligning their predictions with global time-series patterns, rather than relying solely on the similarity of client updates. Our approach extracts logical reasoning properties from clients, then hierarchically infers global properties, and uses these to verify client updates. Through formal logic verification, we assess the robustness of each client contribution, identifying deviations indicative of adversarial behavior. Experimental results on two datasets demonstrate the superior performance of our approach compared to existing baseline methods, highlighting its potential to enhance the robustness of FL to time series applications. Notably, FLORAL reduced the prediction error by 93.27\% in the best-case scenario compared to the second-best baseline. Our code is available at \url{https://anonymous.4open.science/r/FLORAL-Robust-FTS}.



## **4. Gradient-Guided Conditional Diffusion Models for Private Image Reconstruction: Analyzing Adversarial Impacts of Differential Privacy and Denoising**

cs.CV

**SubmitDate**: 2024-11-05    [abs](http://arxiv.org/abs/2411.03053v1) [paper-pdf](http://arxiv.org/pdf/2411.03053v1)

**Authors**: Tao Huang, Jiayang Meng, Hong Chen, Guolong Zheng, Xu Yang, Xun Yi, Hua Wang

**Abstract**: We investigate the construction of gradient-guided conditional diffusion models for reconstructing private images, focusing on the adversarial interplay between differential privacy noise and the denoising capabilities of diffusion models. While current gradient-based reconstruction methods struggle with high-resolution images due to computational complexity and prior knowledge requirements, we propose two novel methods that require minimal modifications to the diffusion model's generation process and eliminate the need for prior knowledge. Our approach leverages the strong image generation capabilities of diffusion models to reconstruct private images starting from randomly generated noise, even when a small amount of differentially private noise has been added to the gradients. We also conduct a comprehensive theoretical analysis of the impact of differential privacy noise on the quality of reconstructed images, revealing the relationship among noise magnitude, the architecture of attacked models, and the attacker's reconstruction capability. Additionally, extensive experiments validate the effectiveness of our proposed methods and the accuracy of our theoretical findings, suggesting new directions for privacy risk auditing using conditional diffusion models.



## **5. Adversarial Markov Games: On Adaptive Decision-Based Attacks and Defenses**

cs.AI

**SubmitDate**: 2024-11-05    [abs](http://arxiv.org/abs/2312.13435v2) [paper-pdf](http://arxiv.org/pdf/2312.13435v2)

**Authors**: Ilias Tsingenopoulos, Vera Rimmer, Davy Preuveneers, Fabio Pierazzi, Lorenzo Cavallaro, Wouter Joosen

**Abstract**: Despite considerable efforts on making them robust, real-world ML-based systems remain vulnerable to decision based attacks, as definitive proofs of their operational robustness have so far proven intractable. The canonical approach in robustness evaluation calls for adaptive attacks, that is with complete knowledge of the defense and tailored to bypass it. In this study, we introduce a more expansive notion of being adaptive and show how attacks but also defenses can benefit by it and by learning from each other through interaction. We propose and evaluate a framework for adaptively optimizing black-box attacks and defenses against each other through the competitive game they form. To reliably measure robustness, it is important to evaluate against realistic and worst-case attacks. We thus augment both attacks and the evasive arsenal at their disposal through adaptive control, and observe that the same can be done for defenses, before we evaluate them first apart and then jointly under a multi-agent perspective. We demonstrate that active defenses, which control how the system responds, are a necessary complement to model hardening when facing decision-based attacks; then how these defenses can be circumvented by adaptive attacks, only to finally elicit active and adaptive defenses. We validate our observations through a wide theoretical and empirical investigation to confirm that AI-enabled adversaries pose a considerable threat to black-box ML-based systems, rekindling the proverbial arms race where defenses have to be AI-enabled too. Succinctly, we address the challenges posed by adaptive adversaries and develop adaptive defenses, thereby laying out effective strategies in ensuring the robustness of ML-based systems deployed in the real-world.



## **6. Flashy Backdoor: Real-world Environment Backdoor Attack on SNNs with DVS Cameras**

cs.CR

**SubmitDate**: 2024-11-05    [abs](http://arxiv.org/abs/2411.03022v1) [paper-pdf](http://arxiv.org/pdf/2411.03022v1)

**Authors**: Roberto Riaño, Gorka Abad, Stjepan Picek, Aitor Urbieta

**Abstract**: While security vulnerabilities in traditional Deep Neural Networks (DNNs) have been extensively studied, the susceptibility of Spiking Neural Networks (SNNs) to adversarial attacks remains mostly underexplored. Until now, the mechanisms to inject backdoors into SNN models have been limited to digital scenarios; thus, we present the first evaluation of backdoor attacks in real-world environments.   We begin by assessing the applicability of existing digital backdoor attacks and identifying their limitations for deployment in physical environments. To address each of the found limitations, we present three novel backdoor attack methods on SNNs, i.e., Framed, Strobing, and Flashy Backdoor. We also assess the effectiveness of traditional backdoor procedures and defenses adapted for SNNs, such as pruning, fine-tuning, and fine-pruning. The results show that while these procedures and defenses can mitigate some attacks, they often fail against stronger methods like Flashy Backdoor or sacrifice too much clean accuracy, rendering the models unusable.   Overall, all our methods can achieve up to a 100% Attack Success Rate while maintaining high clean accuracy in every tested dataset. Additionally, we evaluate the stealthiness of the triggers with commonly used metrics, finding them highly stealthy. Thus, we propose new alternatives more suited for identifying poisoned samples in these scenarios. Our results show that further research is needed to ensure the security of SNN-based systems against backdoor attacks and their safe application in real-world scenarios. The code, experiments, and results are available in our repository.



## **7. Region-Guided Attack on the Segment Anything Model (SAM)**

cs.CV

**SubmitDate**: 2024-11-05    [abs](http://arxiv.org/abs/2411.02974v1) [paper-pdf](http://arxiv.org/pdf/2411.02974v1)

**Authors**: Xiaoliang Liu, Furao Shen, Jian Zhao

**Abstract**: The Segment Anything Model (SAM) is a cornerstone of image segmentation, demonstrating exceptional performance across various applications, particularly in autonomous driving and medical imaging, where precise segmentation is crucial. However, SAM is vulnerable to adversarial attacks that can significantly impair its functionality through minor input perturbations. Traditional techniques, such as FGSM and PGD, are often ineffective in segmentation tasks due to their reliance on global perturbations that overlook spatial nuances. Recent methods like Attack-SAM-K and UAD have begun to address these challenges, but they frequently depend on external cues and do not fully leverage the structural interdependencies within segmentation processes. This limitation underscores the need for a novel adversarial strategy that exploits the unique characteristics of segmentation tasks. In response, we introduce the Region-Guided Attack (RGA), designed specifically for SAM. RGA utilizes a Region-Guided Map (RGM) to manipulate segmented regions, enabling targeted perturbations that fragment large segments and expand smaller ones, resulting in erroneous outputs from SAM. Our experiments demonstrate that RGA achieves high success rates in both white-box and black-box scenarios, emphasizing the need for robust defenses against such sophisticated attacks. RGA not only reveals SAM's vulnerabilities but also lays the groundwork for developing more resilient defenses against adversarial threats in image segmentation.



## **8. Bias in the Mirror: Are LLMs opinions robust to their own adversarial attacks ?**

cs.CL

**SubmitDate**: 2024-11-05    [abs](http://arxiv.org/abs/2410.13517v2) [paper-pdf](http://arxiv.org/pdf/2410.13517v2)

**Authors**: Virgile Rennard, Christos Xypolopoulos, Michalis Vazirgiannis

**Abstract**: Large language models (LLMs) inherit biases from their training data and alignment processes, influencing their responses in subtle ways. While many studies have examined these biases, little work has explored their robustness during interactions. In this paper, we introduce a novel approach where two instances of an LLM engage in self-debate, arguing opposing viewpoints to persuade a neutral version of the model. Through this, we evaluate how firmly biases hold and whether models are susceptible to reinforcing misinformation or shifting to harmful viewpoints. Our experiments span multiple LLMs of varying sizes, origins, and languages, providing deeper insights into bias persistence and flexibility across linguistic and cultural contexts.



## **9. Towards Efficient Transferable Preemptive Adversarial Defense**

cs.CR

**SubmitDate**: 2024-11-05    [abs](http://arxiv.org/abs/2407.15524v3) [paper-pdf](http://arxiv.org/pdf/2407.15524v3)

**Authors**: Hanrui Wang, Ching-Chun Chang, Chun-Shien Lu, Isao Echizen

**Abstract**: Deep learning technology has brought convenience and advanced developments but has become untrustworthy because of its sensitivity to inconspicuous perturbations (i.e., adversarial attacks). Attackers may utilize this sensitivity to manipulate predictions. To defend against such attacks, we have devised a proactive strategy for "attacking" the medias before it is attacked by the third party, so that when the protected medias are further attacked, the adversarial perturbations are automatically neutralized. This strategy, dubbed Fast Preemption, provides an efficient transferable preemptive defense by using different models for labeling inputs and learning crucial features. A forward-backward cascade learning algorithm is used to compute protective perturbations, starting with forward propagation optimization to achieve rapid convergence, followed by iterative backward propagation learning to alleviate overfitting. This strategy offers state-of-the-art transferability and protection across various systems. With the running of only three steps, our Fast Preemption framework outperforms benchmark training-time, test-time, and preemptive adversarial defenses. We have also devised the first to our knowledge effective white-box adaptive reversion attack and demonstrate that the protection added by our defense strategy is irreversible unless the backbone model, algorithm, and settings are fully compromised. This work provides a new direction to developing proactive defenses against adversarial attacks. The proposed methodology will be made available on GitHub.



## **10. Query-Efficient Adversarial Attack Against Vertical Federated Graph Learning**

cs.LG

**SubmitDate**: 2024-11-05    [abs](http://arxiv.org/abs/2411.02809v1) [paper-pdf](http://arxiv.org/pdf/2411.02809v1)

**Authors**: Jinyin Chen, Wenbo Mu, Luxin Zhang, Guohan Huang, Haibin Zheng, Yao Cheng

**Abstract**: Graph neural network (GNN) has captured wide attention due to its capability of graph representation learning for graph-structured data. However, the distributed data silos limit the performance of GNN. Vertical federated learning (VFL), an emerging technique to process distributed data, successfully makes GNN possible to handle the distributed graph-structured data. Despite the prosperous development of vertical federated graph learning (VFGL), the robustness of VFGL against the adversarial attack has not been explored yet. Although numerous adversarial attacks against centralized GNNs are proposed, their attack performance is challenged in the VFGL scenario. To the best of our knowledge, this is the first work to explore the adversarial attack against VFGL. A query-efficient hybrid adversarial attack framework is proposed to significantly improve the centralized adversarial attacks against VFGL, denoted as NA2, short for Neuron-based Adversarial Attack. Specifically, a malicious client manipulates its local training data to improve its contribution in a stealthy fashion. Then a shadow model is established based on the manipulated data to simulate the behavior of the server model in VFGL. As a result, the shadow model can improve the attack success rate of various centralized attacks with a few queries. Extensive experiments on five real-world benchmarks demonstrate that NA2 improves the performance of the centralized adversarial attacks against VFGL, achieving state-of-the-art performance even under potential adaptive defense where the defender knows the attack method. Additionally, we provide interpretable experiments of the effectiveness of NA2 via sensitive neurons identification and visualization of t-SNE.



## **11. TRANSPOSE: Transitional Approaches for Spatially-Aware LFI Resilient FSM Encoding**

cs.CR

14 pages, 11 figures

**SubmitDate**: 2024-11-05    [abs](http://arxiv.org/abs/2411.02798v1) [paper-pdf](http://arxiv.org/pdf/2411.02798v1)

**Authors**: Muhtadi Choudhury, Minyan Gao, Avinash Varna, Elad Peer, Domenic Forte

**Abstract**: Finite state machines (FSMs) regulate sequential circuits, including access to sensitive information and privileged CPU states. Courtesy of contemporary research on laser attacks, laser-based fault injection (LFI) is becoming even more precise where an adversary can thwart chip security by altering individual flip-flop (FF) values. Different laser models, e.g., bit flip, bit set, and bit reset, have been developed to appreciate LFI on practical targets. As traditional approaches may incorporate substantial overhead, state-based SPARSE and transition-based TAMED countermeasures were proposed in our prior work to improve FSM resiliency efficiently. TAMED overcame SPARSE's limitation of being too conservative, and generating multiple LFI resilient encodings for contemporary LFI models on demand. SPARSE, however, incorporated design layout information into its vulnerability estimation which makes its vulnerability estimation metric more accurate. In this paper, we extend TAMED by proposing a transition-based encoding CAD framework (TRANSPOSE), that incorporates spatial transitional vulnerability metrics to quantify design susceptibility of FSMs based on both the bit flip model and the set-reset models. TRANSPOSE also incorporates floorplan optimization into its framework to accommodate secure spatial inter-distance of FF-sensitive regions. All TRANSPOSE approaches are demonstrated on 5 multifarious benchmarks and outperform existing FSM encoding schemes/frameworks in terms of security and overhead.



## **12. Stochastic Monkeys at Play: Random Augmentations Cheaply Break LLM Safety Alignment**

cs.LG

Under peer review

**SubmitDate**: 2024-11-05    [abs](http://arxiv.org/abs/2411.02785v1) [paper-pdf](http://arxiv.org/pdf/2411.02785v1)

**Authors**: Jason Vega, Junsheng Huang, Gaokai Zhang, Hangoo Kang, Minjia Zhang, Gagandeep Singh

**Abstract**: Safety alignment of Large Language Models (LLMs) has recently become a critical objective of model developers. In response, a growing body of work has been investigating how safety alignment can be bypassed through various jailbreaking methods, such as adversarial attacks. However, these jailbreak methods can be rather costly or involve a non-trivial amount of creativity and effort, introducing the assumption that malicious users are high-resource or sophisticated. In this paper, we study how simple random augmentations to the input prompt affect safety alignment effectiveness in state-of-the-art LLMs, such as Llama 3 and Qwen 2. We perform an in-depth evaluation of 17 different models and investigate the intersection of safety under random augmentations with multiple dimensions: augmentation type, model size, quantization, fine-tuning-based defenses, and decoding strategies (e.g., sampling temperature). We show that low-resource and unsophisticated attackers, i.e. $\textit{stochastic monkeys}$, can significantly improve their chances of bypassing alignment with just 25 random augmentations per prompt.



## **13. Steering cooperation: Adversarial attacks on prisoner's dilemma in complex networks**

physics.soc-ph

19 pages, 4 figures

**SubmitDate**: 2024-11-05    [abs](http://arxiv.org/abs/2406.19692v4) [paper-pdf](http://arxiv.org/pdf/2406.19692v4)

**Authors**: Kazuhiro Takemoto

**Abstract**: This study examines the application of adversarial attack concepts to control the evolution of cooperation in the prisoner's dilemma game in complex networks. Specifically, it proposes a simple adversarial attack method that drives players' strategies towards a target state by adding small perturbations to social networks. The proposed method is evaluated on both model and real-world networks. Numerical simulations demonstrate that the proposed method can effectively promote cooperation with significantly smaller perturbations compared to other techniques. Additionally, this study shows that adversarial attacks can also be useful in inhibiting cooperation (promoting defection). The findings reveal that adversarial attacks on social networks can be potent tools for both promoting and inhibiting cooperation, opening new possibilities for controlling cooperative behavior in social systems while also highlighting potential risks.



## **14. Semantic-Aligned Adversarial Evolution Triangle for High-Transferability Vision-Language Attack**

cs.CV

**SubmitDate**: 2024-11-04    [abs](http://arxiv.org/abs/2411.02669v1) [paper-pdf](http://arxiv.org/pdf/2411.02669v1)

**Authors**: Xiaojun Jia, Sensen Gao, Qing Guo, Ke Ma, Yihao Huang, Simeng Qin, Yang Liu, Ivor Tsang Fellow, Xiaochun Cao

**Abstract**: Vision-language pre-training (VLP) models excel at interpreting both images and text but remain vulnerable to multimodal adversarial examples (AEs). Advancing the generation of transferable AEs, which succeed across unseen models, is key to developing more robust and practical VLP models. Previous approaches augment image-text pairs to enhance diversity within the adversarial example generation process, aiming to improve transferability by expanding the contrast space of image-text features. However, these methods focus solely on diversity around the current AEs, yielding limited gains in transferability. To address this issue, we propose to increase the diversity of AEs by leveraging the intersection regions along the adversarial trajectory during optimization. Specifically, we propose sampling from adversarial evolution triangles composed of clean, historical, and current adversarial examples to enhance adversarial diversity. We provide a theoretical analysis to demonstrate the effectiveness of the proposed adversarial evolution triangle. Moreover, we find that redundant inactive dimensions can dominate similarity calculations, distorting feature matching and making AEs model-dependent with reduced transferability. Hence, we propose to generate AEs in the semantic image-text feature contrast space, which can project the original feature space into a semantic corpus subspace. The proposed semantic-aligned subspace can reduce the image feature redundancy, thereby improving adversarial transferability. Extensive experiments across different datasets and models demonstrate that the proposed method can effectively improve adversarial transferability and outperform state-of-the-art adversarial attack methods. The code is released at https://github.com/jiaxiaojunQAQ/SA-AET.



## **15. Class-Conditioned Transformation for Enhanced Robust Image Classification**

cs.CV

**SubmitDate**: 2024-11-04    [abs](http://arxiv.org/abs/2303.15409v2) [paper-pdf](http://arxiv.org/pdf/2303.15409v2)

**Authors**: Tsachi Blau, Roy Ganz, Chaim Baskin, Michael Elad, Alex M. Bronstein

**Abstract**: Robust classification methods predominantly concentrate on algorithms that address a specific threat model, resulting in ineffective defenses against other threat models. Real-world applications are exposed to this vulnerability, as malicious attackers might exploit alternative threat models. In this work, we propose a novel test-time threat model agnostic algorithm that enhances Adversarial-Trained (AT) models. Our method operates through COnditional image transformation and DIstance-based Prediction (CODIP) and includes two main steps: First, we transform the input image into each dataset class, where the input image might be either clean or attacked. Next, we make a prediction based on the shortest transformed distance. The conditional transformation utilizes the perceptually aligned gradients property possessed by AT models and, as a result, eliminates the need for additional models or additional training. Moreover, it allows users to choose the desired balance between clean and robust accuracy without training. The proposed method achieves state-of-the-art results demonstrated through extensive experiments on various models, AT methods, datasets, and attack types. Notably, applying CODIP leads to substantial robust accuracy improvement of up to $+23\%$, $+20\%$, $+26\%$, and $+22\%$ on CIFAR10, CIFAR100, ImageNet and Flowers datasets, respectively.



## **16. Attacking Vision-Language Computer Agents via Pop-ups**

cs.CL

10 pages, preprint

**SubmitDate**: 2024-11-04    [abs](http://arxiv.org/abs/2411.02391v1) [paper-pdf](http://arxiv.org/pdf/2411.02391v1)

**Authors**: Yanzhe Zhang, Tao Yu, Diyi Yang

**Abstract**: Autonomous agents powered by large vision and language models (VLM) have demonstrated significant potential in completing daily computer tasks, such as browsing the web to book travel and operating desktop software, which requires agents to understand these interfaces. Despite such visual inputs becoming more integrated into agentic applications, what types of risks and attacks exist around them still remain unclear. In this work, we demonstrate that VLM agents can be easily attacked by a set of carefully designed adversarial pop-ups, which human users would typically recognize and ignore. This distraction leads agents to click these pop-ups instead of performing the tasks as usual. Integrating these pop-ups into existing agent testing environments like OSWorld and VisualWebArena leads to an attack success rate (the frequency of the agent clicking the pop-ups) of 86% on average and decreases the task success rate by 47%. Basic defense techniques such as asking the agent to ignore pop-ups or including an advertisement notice, are ineffective against the attack.



## **17. Swiper: a new paradigm for efficient weighted distributed protocols**

cs.DC

**SubmitDate**: 2024-11-04    [abs](http://arxiv.org/abs/2307.15561v2) [paper-pdf](http://arxiv.org/pdf/2307.15561v2)

**Authors**: Andrei Tonkikh, Luciano Freitas

**Abstract**: The majority of fault-tolerant distributed algorithms are designed assuming a nominal corruption model, in which at most a fraction $f_n$ of parties can be corrupted by the adversary. However, due to the infamous Sybil attack, nominal models are not sufficient to express the trust assumptions in open (i.e., permissionless) settings. Instead, permissionless systems typically operate in a weighted model, where each participant is associated with a weight and the adversary can corrupt a set of parties holding at most a fraction $f_w$ of the total weight.   In this paper, we suggest a simple way to transform a large class of protocols designed for the nominal model into the weighted model. To this end, we formalize and solve three novel optimization problems, which we collectively call the weight reduction problems, that allow us to map large real weights into small integer weights while preserving the properties necessary for the correctness of the protocols. In all cases, we manage to keep the sum of the integer weights to be at most linear in the number of parties, resulting in extremely efficient protocols for the weighted model. Moreover, we demonstrate that, on weight distributions that emerge in practice, the sum of the integer weights tends to be far from the theoretical worst case and, sometimes, even smaller than the number of participants.   While, for some protocols, our transformation requires an arbitrarily small reduction in resilience (i.e., $f_w = f_n - \epsilon$), surprisingly, for many important problems, we manage to obtain weighted solutions with the same resilience ($f_w = f_n$) as nominal ones. Notable examples include erasure-coded distributed storage and broadcast protocols, verifiable secret sharing, and asynchronous consensus.



## **18. Manipulation Facing Threats: Evaluating Physical Vulnerabilities in End-to-End Vision Language Action Models**

cs.CV

**SubmitDate**: 2024-11-04    [abs](http://arxiv.org/abs/2409.13174v2) [paper-pdf](http://arxiv.org/pdf/2409.13174v2)

**Authors**: Hao Cheng, Erjia Xiao, Chengyuan Yu, Zhao Yao, Jiahang Cao, Qiang Zhang, Jiaxu Wang, Mengshu Sun, Kaidi Xu, Jindong Gu, Renjing Xu

**Abstract**: Recently, driven by advancements in Multimodal Large Language Models (MLLMs), Vision Language Action Models (VLAMs) are being proposed to achieve better performance in open-vocabulary scenarios for robotic manipulation tasks. Since manipulation tasks involve direct interaction with the physical world, ensuring robustness and safety during the execution of this task is always a very critical issue. In this paper, by synthesizing current safety research on MLLMs and the specific application scenarios of the manipulation task in the physical world, we comprehensively evaluate VLAMs in the face of potential physical threats. Specifically, we propose the Physical Vulnerability Evaluating Pipeline (PVEP) that can incorporate as many visual modal physical threats as possible for evaluating the physical robustness of VLAMs. The physical threats in PVEP specifically include Out-of-Distribution, Typography-based Visual Prompts, and Adversarial Patch Attacks. By comparing the performance fluctuations of VLAMs before and after being attacked, we provide generalizable Analyses of how VLAMs respond to different physical security threats. Our project page is in this link: https://chaducheng.github.io/Manipulat-Facing-Threats/.



## **19. Alignment-Based Adversarial Training (ABAT) for Improving the Robustness and Accuracy of EEG-Based BCIs**

cs.HC

**SubmitDate**: 2024-11-04    [abs](http://arxiv.org/abs/2411.02094v1) [paper-pdf](http://arxiv.org/pdf/2411.02094v1)

**Authors**: Xiaoqing Chen, Ziwei Wang, Dongrui Wu

**Abstract**: Machine learning has achieved great success in electroencephalogram (EEG) based brain-computer interfaces (BCIs). Most existing BCI studies focused on improving the decoding accuracy, with only a few considering the adversarial security. Although many adversarial defense approaches have been proposed in other application domains such as computer vision, previous research showed that their direct extensions to BCIs degrade the classification accuracy on benign samples. This phenomenon greatly affects the applicability of adversarial defense approaches to EEG-based BCIs. To mitigate this problem, we propose alignment-based adversarial training (ABAT), which performs EEG data alignment before adversarial training. Data alignment aligns EEG trials from different domains to reduce their distribution discrepancies, and adversarial training further robustifies the classification boundary. The integration of data alignment and adversarial training can make the trained EEG classifiers simultaneously more accurate and more robust. Experiments on five EEG datasets from two different BCI paradigms (motor imagery classification, and event related potential recognition), three convolutional neural network classifiers (EEGNet, ShallowCNN and DeepCNN) and three different experimental settings (offline within-subject cross-block/-session classification, online cross-session classification, and pre-trained classifiers) demonstrated its effectiveness. It is very intriguing that adversarial attacks, which are usually used to damage BCI systems, can be used in ABAT to simultaneously improve the model accuracy and robustness.



## **20. Exploiting LLM Quantization**

cs.LG

**SubmitDate**: 2024-11-04    [abs](http://arxiv.org/abs/2405.18137v2) [paper-pdf](http://arxiv.org/pdf/2405.18137v2)

**Authors**: Kazuki Egashira, Mark Vero, Robin Staab, Jingxuan He, Martin Vechev

**Abstract**: Quantization leverages lower-precision weights to reduce the memory usage of large language models (LLMs) and is a key technique for enabling their deployment on commodity hardware. While LLM quantization's impact on utility has been extensively explored, this work for the first time studies its adverse effects from a security perspective. We reveal that widely used quantization methods can be exploited to produce a harmful quantized LLM, even though the full-precision counterpart appears benign, potentially tricking users into deploying the malicious quantized model. We demonstrate this threat using a three-staged attack framework: (i) first, we obtain a malicious LLM through fine-tuning on an adversarial task; (ii) next, we quantize the malicious model and calculate constraints that characterize all full-precision models that map to the same quantized model; (iii) finally, using projected gradient descent, we tune out the poisoned behavior from the full-precision model while ensuring that its weights satisfy the constraints computed in step (ii). This procedure results in an LLM that exhibits benign behavior in full precision but when quantized, it follows the adversarial behavior injected in step (i). We experimentally demonstrate the feasibility and severity of such an attack across three diverse scenarios: vulnerable code generation, content injection, and over-refusal attack. In practice, the adversary could host the resulting full-precision model on an LLM community hub such as Hugging Face, exposing millions of users to the threat of deploying its malicious quantized version on their devices.



## **21. DeSparsify: Adversarial Attack Against Token Sparsification Mechanisms in Vision Transformers**

cs.CV

18 pages, 6 figures

**SubmitDate**: 2024-11-04    [abs](http://arxiv.org/abs/2402.02554v2) [paper-pdf](http://arxiv.org/pdf/2402.02554v2)

**Authors**: Oryan Yehezkel, Alon Zolfi, Amit Baras, Yuval Elovici, Asaf Shabtai

**Abstract**: Vision transformers have contributed greatly to advancements in the computer vision domain, demonstrating state-of-the-art performance in diverse tasks (e.g., image classification, object detection). However, their high computational requirements grow quadratically with the number of tokens used. Token sparsification mechanisms have been proposed to address this issue. These mechanisms employ an input-dependent strategy, in which uninformative tokens are discarded from the computation pipeline, improving the model's efficiency. However, their dynamism and average-case assumption makes them vulnerable to a new threat vector - carefully crafted adversarial examples capable of fooling the sparsification mechanism, resulting in worst-case performance. In this paper, we present DeSparsify, an attack targeting the availability of vision transformers that use token sparsification mechanisms. The attack aims to exhaust the operating system's resources, while maintaining its stealthiness. Our evaluation demonstrates the attack's effectiveness on three token sparsification mechanisms and examines the attack's transferability between them and its effect on the GPU resources. To mitigate the impact of the attack, we propose various countermeasures.



## **22. LiDAttack: Robust Black-box Attack on LiDAR-based Object Detection**

cs.CV

**SubmitDate**: 2024-11-04    [abs](http://arxiv.org/abs/2411.01889v1) [paper-pdf](http://arxiv.org/pdf/2411.01889v1)

**Authors**: Jinyin Chen, Danxin Liao, Sheng Xiang, Haibin Zheng

**Abstract**: Since DNN is vulnerable to carefully crafted adversarial examples, adversarial attack on LiDAR sensors have been extensively studied. We introduce a robust black-box attack dubbed LiDAttack. It utilizes a genetic algorithm with a simulated annealing strategy to strictly limit the location and number of perturbation points, achieving a stealthy and effective attack. And it simulates scanning deviations, allowing it to adapt to dynamic changes in real world scenario variations. Extensive experiments are conducted on 3 datasets (i.e., KITTI, nuScenes, and self-constructed data) with 3 dominant object detection models (i.e., PointRCNN, PointPillar, and PV-RCNN++). The results reveal the efficiency of the LiDAttack when targeting a wide range of object detection models, with an attack success rate (ASR) up to 90%.



## **23. Learning predictable and robust neural representations by straightening image sequences**

cs.CV

Accepted at NeurIPS 2024

**SubmitDate**: 2024-11-04    [abs](http://arxiv.org/abs/2411.01777v1) [paper-pdf](http://arxiv.org/pdf/2411.01777v1)

**Authors**: Xueyan Niu, Cristina Savin, Eero P. Simoncelli

**Abstract**: Prediction is a fundamental capability of all living organisms, and has been proposed as an objective for learning sensory representations. Recent work demonstrates that in primate visual systems, prediction is facilitated by neural representations that follow straighter temporal trajectories than their initial photoreceptor encoding, which allows for prediction by linear extrapolation. Inspired by these experimental findings, we develop a self-supervised learning (SSL) objective that explicitly quantifies and promotes straightening. We demonstrate the power of this objective in training deep feedforward neural networks on smoothly-rendered synthetic image sequences that mimic commonly-occurring properties of natural videos. The learned model contains neural embeddings that are predictive, but also factorize the geometric, photometric, and semantic attributes of objects. The representations also prove more robust to noise and adversarial attacks compared to previous SSL methods that optimize for invariance to random augmentations. Moreover, these beneficial properties can be transferred to other training procedures by using the straightening objective as a regularizer, suggesting a broader utility for straightening as a principle for robust unsupervised learning.



## **24. Survey on Adversarial Attack and Defense for Medical Image Analysis: Methods and Challenges**

eess.IV

Accepted by ACM Computing Surveys (CSUR) (DOI:  https://doi.org/10.1145/3702638)

**SubmitDate**: 2024-11-04    [abs](http://arxiv.org/abs/2303.14133v2) [paper-pdf](http://arxiv.org/pdf/2303.14133v2)

**Authors**: Junhao Dong, Junxi Chen, Xiaohua Xie, Jianhuang Lai, Hao Chen

**Abstract**: Deep learning techniques have achieved superior performance in computer-aided medical image analysis, yet they are still vulnerable to imperceptible adversarial attacks, resulting in potential misdiagnosis in clinical practice. Oppositely, recent years have also witnessed remarkable progress in defense against these tailored adversarial examples in deep medical diagnosis systems. In this exposition, we present a comprehensive survey on recent advances in adversarial attacks and defenses for medical image analysis with a systematic taxonomy in terms of the application scenario. We also provide a unified framework for different types of adversarial attack and defense methods in the context of medical image analysis. For a fair comparison, we establish a new benchmark for adversarially robust medical diagnosis models obtained by adversarial training under various scenarios. To the best of our knowledge, this is the first survey paper that provides a thorough evaluation of adversarially robust medical diagnosis models. By analyzing qualitative and quantitative results, we conclude this survey with a detailed discussion of current challenges for adversarial attack and defense in medical image analysis systems to shed light on future research directions. Code is available on \href{https://github.com/tomvii/Adv_MIA}{\color{red}{GitHub}}.



## **25. A General Recipe for Contractive Graph Neural Networks -- Technical Report**

cs.LG

**SubmitDate**: 2024-11-04    [abs](http://arxiv.org/abs/2411.01717v1) [paper-pdf](http://arxiv.org/pdf/2411.01717v1)

**Authors**: Maya Bechler-Speicher, Moshe Eliasof

**Abstract**: Graph Neural Networks (GNNs) have gained significant popularity for learning representations of graph-structured data due to their expressive power and scalability. However, despite their success in domains such as social network analysis, recommendation systems, and bioinformatics, GNNs often face challenges related to stability, generalization, and robustness to noise and adversarial attacks. Regularization techniques have shown promise in addressing these challenges by controlling model complexity and improving robustness. Building on recent advancements in contractive GNN architectures, this paper presents a novel method for inducing contractive behavior in any GNN through SVD regularization. By deriving a sufficient condition for contractiveness in the update step and applying constraints on network parameters, we demonstrate the impact of SVD regularization on the Lipschitz constant of GNNs. Our findings highlight the role of SVD regularization in enhancing the stability and generalization of GNNs, contributing to the development of more robust graph-based learning algorithms dynamics.



## **26. UniGuard: Towards Universal Safety Guardrails for Jailbreak Attacks on Multimodal Large Language Models**

cs.CL

14 pages

**SubmitDate**: 2024-11-03    [abs](http://arxiv.org/abs/2411.01703v1) [paper-pdf](http://arxiv.org/pdf/2411.01703v1)

**Authors**: Sejoon Oh, Yiqiao Jin, Megha Sharma, Donghyun Kim, Eric Ma, Gaurav Verma, Srijan Kumar

**Abstract**: Multimodal large language models (MLLMs) have revolutionized vision-language understanding but are vulnerable to multimodal jailbreak attacks, where adversaries meticulously craft inputs to elicit harmful or inappropriate responses. We propose UniGuard, a novel multimodal safety guardrail that jointly considers the unimodal and cross-modal harmful signals. UniGuard is trained such that the likelihood of generating harmful responses in a toxic corpus is minimized, and can be seamlessly applied to any input prompt during inference with minimal computational costs. Extensive experiments demonstrate the generalizability of UniGuard across multiple modalities and attack strategies. It demonstrates impressive generalizability across multiple state-of-the-art MLLMs, including LLaVA, Gemini Pro, GPT-4, MiniGPT-4, and InstructBLIP, thereby broadening the scope of our solution.



## **27. Building the Self-Improvement Loop: Error Detection and Correction in Goal-Oriented Semantic Communications**

cs.NI

7 pages, 8 figures, this paper has been accepted for publication in  IEEE CSCN 2024

**SubmitDate**: 2024-11-03    [abs](http://arxiv.org/abs/2411.01544v1) [paper-pdf](http://arxiv.org/pdf/2411.01544v1)

**Authors**: Peizheng Li, Xinyi Lin, Adnan Aijaz

**Abstract**: Error detection and correction are essential for ensuring robust and reliable operation in modern communication systems, particularly in complex transmission environments. However, discussions on these topics have largely been overlooked in semantic communication (SemCom), which focuses on transmitting meaning rather than symbols, leading to significant improvements in communication efficiency. Despite these advantages, semantic errors -- stemming from discrepancies between transmitted and received meanings -- present a major challenge to system reliability. This paper addresses this gap by proposing a comprehensive framework for detecting and correcting semantic errors in SemCom systems. We formally define semantic error, detection, and correction mechanisms, and identify key sources of semantic errors. To address these challenges, we develop a Gaussian process (GP)-based method for latent space monitoring to detect errors, alongside a human-in-the-loop reinforcement learning (HITL-RL) approach to optimize semantic model configurations using user feedback. Experimental results validate the effectiveness of the proposed methods in mitigating semantic errors under various conditions, including adversarial attacks, input feature changes, physical channel variations, and user preference shifts. This work lays the foundation for more reliable and adaptive SemCom systems with robust semantic error management techniques.



## **28. Privacy-Preserving Customer Churn Prediction Model in the Context of Telecommunication Industry**

cs.LG

26 pages, 14 tables, 13 figures

**SubmitDate**: 2024-11-03    [abs](http://arxiv.org/abs/2411.01447v1) [paper-pdf](http://arxiv.org/pdf/2411.01447v1)

**Authors**: Joydeb Kumar Sana, M Sohel Rahman, M Saifur Rahman

**Abstract**: Data is the main fuel of a successful machine learning model. A dataset may contain sensitive individual records e.g. personal health records, financial data, industrial information, etc. Training a model using this sensitive data has become a new privacy concern when someone uses third-party cloud computing. Trained models also suffer privacy attacks which leads to the leaking of sensitive information of the training data. This study is conducted to preserve the privacy of training data in the context of customer churn prediction modeling for the telecommunications industry (TCI). In this work, we propose a framework for privacy-preserving customer churn prediction (PPCCP) model in the cloud environment. We have proposed a novel approach which is a combination of Generative Adversarial Networks (GANs) and adaptive Weight-of-Evidence (aWOE). Synthetic data is generated from GANs, and aWOE is applied on the synthetic training dataset before feeding the data to the classification algorithms. Our experiments were carried out using eight different machine learning (ML) classifiers on three openly accessible datasets from the telecommunication sector. We then evaluated the performance using six commonly employed evaluation metrics. In addition to presenting a data privacy analysis, we also performed a statistical significance test. The training and prediction processes achieve data privacy and the prediction classifiers achieve high prediction performance (87.1\% in terms of F-Measure for GANs-aWOE based Na\"{\i}ve Bayes model). In contrast to earlier studies, our suggested approach demonstrates a prediction enhancement of up to 28.9\% and 27.9\% in terms of accuracy and F-measure, respectively.



## **29. Enhancing Cyber-Resilience in Integrated Energy System Scheduling with Demand Response Using Deep Reinforcement Learning**

eess.SY

Accepted by Applied Energy, Manuscript ID: APEN-D-24-03080

**SubmitDate**: 2024-11-03    [abs](http://arxiv.org/abs/2311.17941v2) [paper-pdf](http://arxiv.org/pdf/2311.17941v2)

**Authors**: Yang Li, Wenjie Ma, Yuanzheng Li, Sen Li, Zhe Chen, Mohammad Shahidehpor

**Abstract**: Optimally scheduling multi-energy flow is an effective method to utilize renewable energy sources (RES) and improve the stability and economy of integrated energy systems (IES). However, the stable demand-supply of IES faces challenges from uncertainties that arise from RES and loads, as well as the increasing impact of cyber-attacks with advanced information and communication technologies adoption. To address these challenges, this paper proposes an innovative model-free resilience scheduling method based on state-adversarial deep reinforcement learning (DRL) for integrated demand response (IDR)-enabled IES. The proposed method designs an IDR program to explore the interaction ability of electricity-gas-heat flexible loads. Additionally, the state-adversarial Markov decision process (SA-MDP) model characterizes the energy scheduling problem of IES under cyber-attack, incorporating cyber-attacks as adversaries directly into the scheduling process. The state-adversarial soft actor-critic (SA-SAC) algorithm is proposed to mitigate the impact of cyber-attacks on the scheduling strategy, integrating adversarial training into the learning process to against cyber-attacks. Simulation results demonstrate that our method is capable of adequately addressing the uncertainties resulting from RES and loads, mitigating the impact of cyber-attacks on the scheduling strategy, and ensuring a stable demand supply for various energy sources. Moreover, the proposed method demonstrates resilience against cyber-attacks. Compared to the original soft actor-critic (SAC) algorithm, it achieves a 10% improvement in economic performance under cyber-attack scenarios.



## **30. High-Frequency Anti-DreamBooth: Robust Defense against Personalized Image Synthesis**

cs.CV

ECCV 2024 Workshop The Dark Side of Generative AIs and Beyond. Our  code is available at https://github.com/mti-lab/HF-ADB

**SubmitDate**: 2024-11-03    [abs](http://arxiv.org/abs/2409.08167v3) [paper-pdf](http://arxiv.org/pdf/2409.08167v3)

**Authors**: Takuto Onikubo, Yusuke Matsui

**Abstract**: Recently, text-to-image generative models have been misused to create unauthorized malicious images of individuals, posing a growing social problem. Previous solutions, such as Anti-DreamBooth, add adversarial noise to images to protect them from being used as training data for malicious generation. However, we found that the adversarial noise can be removed by adversarial purification methods such as DiffPure. Therefore, we propose a new adversarial attack method that adds strong perturbation on the high-frequency areas of images to make it more robust to adversarial purification. Our experiment showed that the adversarial images retained noise even after adversarial purification, hindering malicious image generation.



## **31. AED: An black-box NLP classifier model attacker**

cs.LG

**SubmitDate**: 2024-11-03    [abs](http://arxiv.org/abs/2112.11660v4) [paper-pdf](http://arxiv.org/pdf/2112.11660v4)

**Authors**: Yueyang Liu, Yan Huang, Zhipeng Cai

**Abstract**: Deep Neural Networks (DNNs) have been successful in solving real-world tasks in domains such as connected and automated vehicles, disease, and job hiring. However, their implications are far-reaching in critical application areas. Hence, there is a growing concern regarding the potential bias and robustness of these DNN models. A transparency and robust model is always demanded in high-stakes domains where reliability and safety are enforced, such as healthcare and finance. While most studies have focused on adversarial image attack scenarios, fewer studies have investigated the robustness of DNN models in natural language processing (NLP) due to their adversarial samples are difficult to generate. To address this gap, we propose a word-level NLP classifier attack model called "AED," which stands for Attention mechanism enabled post-model Explanation with Density peaks clustering algorithm for synonyms search and substitution. AED aims to test the robustness of NLP DNN models by interpretability their weaknesses and exploring alternative ways to optimize them. By identifying vulnerabilities and providing explanations, AED can help improve the reliability and safety of DNN models in critical application areas such as healthcare and automated transportation. Our experiment results demonstrate that compared with other existing models, AED can effectively generate adversarial examples that can fool the victim model while maintaining the original meaning of the input.



## **32. Understanding and Improving Adversarial Collaborative Filtering for Robust Recommendation**

cs.IR

To appear in NeurIPS 2024

**SubmitDate**: 2024-11-02    [abs](http://arxiv.org/abs/2410.22844v2) [paper-pdf](http://arxiv.org/pdf/2410.22844v2)

**Authors**: Kaike Zhang, Qi Cao, Yunfan Wu, Fei Sun, Huawei Shen, Xueqi Cheng

**Abstract**: Adversarial Collaborative Filtering (ACF), which typically applies adversarial perturbations at user and item embeddings through adversarial training, is widely recognized as an effective strategy for enhancing the robustness of Collaborative Filtering (CF) recommender systems against poisoning attacks. Besides, numerous studies have empirically shown that ACF can also improve recommendation performance compared to traditional CF. Despite these empirical successes, the theoretical understanding of ACF's effectiveness in terms of both performance and robustness remains unclear. To bridge this gap, in this paper, we first theoretically show that ACF can achieve a lower recommendation error compared to traditional CF with the same training epochs in both clean and poisoned data contexts. Furthermore, by establishing bounds for reductions in recommendation error during ACF's optimization process, we find that applying personalized magnitudes of perturbation for different users based on their embedding scales can further improve ACF's effectiveness. Building on these theoretical understandings, we propose Personalized Magnitude Adversarial Collaborative Filtering (PamaCF). Extensive experiments demonstrate that PamaCF effectively defends against various types of poisoning attacks while significantly enhancing recommendation performance.



## **33. $B^4$: A Black-Box Scrubbing Attack on LLM Watermarks**

cs.CL

**SubmitDate**: 2024-11-02    [abs](http://arxiv.org/abs/2411.01222v1) [paper-pdf](http://arxiv.org/pdf/2411.01222v1)

**Authors**: Baizhou Huang, Xiao Pu, Xiaojun Wan

**Abstract**: Watermarking has emerged as a prominent technique for LLM-generated content detection by embedding imperceptible patterns. Despite supreme performance, its robustness against adversarial attacks remains underexplored. Previous work typically considers a grey-box attack setting, where the specific type of watermark is already known. Some even necessitates knowledge about hyperparameters of the watermarking method. Such prerequisites are unattainable in real-world scenarios. Targeting at a more realistic black-box threat model with fewer assumptions, we here propose $\mathcal{B}^4$, a black-box scrubbing attack on watermarks. Specifically, we formulate the watermark scrubbing attack as a constrained optimization problem by capturing its objectives with two distributions, a Watermark Distribution and a Fidelity Distribution. This optimization problem can be approximately solved using two proxy distributions. Experimental results across 12 different settings demonstrate the superior performance of $\mathcal{B}^4$ compared with other baselines.



## **34. Plentiful Jailbreaks with String Compositions**

cs.CL

NeurIPS SoLaR Workshop 2024

**SubmitDate**: 2024-11-01    [abs](http://arxiv.org/abs/2411.01084v1) [paper-pdf](http://arxiv.org/pdf/2411.01084v1)

**Authors**: Brian R. Y. Huang

**Abstract**: Large language models (LLMs) remain vulnerable to a slew of adversarial attacks and jailbreaking methods. One common approach employed by white-hat attackers, or \textit{red-teamers}, is to process model inputs and outputs using string-level obfuscations, which can include leetspeak, rotary ciphers, Base64, ASCII, and more. Our work extends these encoding-based attacks by unifying them in a framework of invertible string transformations. With invertibility, we can devise arbitrary \textit{string compositions}, defined as sequences of transformations, that we can encode and decode end-to-end programmatically. We devise a automated best-of-n attack that samples from a combinatorially large number of string compositions. Our jailbreaks obtain competitive attack success rates on several leading frontier models when evaluated on HarmBench, highlighting that encoding-based attacks remain a persistent vulnerability even in advanced LLMs.



## **35. AdjointDEIS: Efficient Gradients for Diffusion Models**

cs.CV

Accepted in NeurIPS 2024

**SubmitDate**: 2024-11-01    [abs](http://arxiv.org/abs/2405.15020v2) [paper-pdf](http://arxiv.org/pdf/2405.15020v2)

**Authors**: Zander W. Blasingame, Chen Liu

**Abstract**: The optimization of the latents and parameters of diffusion models with respect to some differentiable metric defined on the output of the model is a challenging and complex problem. The sampling for diffusion models is done by solving either the probability flow ODE or diffusion SDE wherein a neural network approximates the score function allowing a numerical ODE/SDE solver to be used. However, naive backpropagation techniques are memory intensive, requiring the storage of all intermediate states, and face additional complexity in handling the injected noise from the diffusion term of the diffusion SDE. We propose a novel family of bespoke ODE solvers to the continuous adjoint equations for diffusion models, which we call AdjointDEIS. We exploit the unique construction of diffusion SDEs to further simplify the formulation of the continuous adjoint equations using exponential integrators. Moreover, we provide convergence order guarantees for our bespoke solvers. Significantly, we show that continuous adjoint equations for diffusion SDEs actually simplify to a simple ODE. Lastly, we demonstrate the effectiveness of AdjointDEIS for guided generation with an adversarial attack in the form of the face morphing problem. Our code will be released on our project page https://zblasingame.github.io/AdjointDEIS/



## **36. AdvI2I: Adversarial Image Attack on Image-to-Image Diffusion models**

cs.CV

**SubmitDate**: 2024-11-01    [abs](http://arxiv.org/abs/2410.21471v2) [paper-pdf](http://arxiv.org/pdf/2410.21471v2)

**Authors**: Yaopei Zeng, Yuanpu Cao, Bochuan Cao, Yurui Chang, Jinghui Chen, Lu Lin

**Abstract**: Recent advances in diffusion models have significantly enhanced the quality of image synthesis, yet they have also introduced serious safety concerns, particularly the generation of Not Safe for Work (NSFW) content. Previous research has demonstrated that adversarial prompts can be used to generate NSFW content. However, such adversarial text prompts are often easily detectable by text-based filters, limiting their efficacy. In this paper, we expose a previously overlooked vulnerability: adversarial image attacks targeting Image-to-Image (I2I) diffusion models. We propose AdvI2I, a novel framework that manipulates input images to induce diffusion models to generate NSFW content. By optimizing a generator to craft adversarial images, AdvI2I circumvents existing defense mechanisms, such as Safe Latent Diffusion (SLD), without altering the text prompts. Furthermore, we introduce AdvI2I-Adaptive, an enhanced version that adapts to potential countermeasures and minimizes the resemblance between adversarial images and NSFW concept embeddings, making the attack more resilient against defenses. Through extensive experiments, we demonstrate that both AdvI2I and AdvI2I-Adaptive can effectively bypass current safeguards, highlighting the urgent need for stronger security measures to address the misuse of I2I diffusion models.



## **37. Efficient Adversarial Training in LLMs with Continuous Attacks**

cs.LG

19 pages, 4 figures

**SubmitDate**: 2024-11-01    [abs](http://arxiv.org/abs/2405.15589v3) [paper-pdf](http://arxiv.org/pdf/2405.15589v3)

**Authors**: Sophie Xhonneux, Alessandro Sordoni, Stephan Günnemann, Gauthier Gidel, Leo Schwinn

**Abstract**: Large language models (LLMs) are vulnerable to adversarial attacks that can bypass their safety guardrails. In many domains, adversarial training has proven to be one of the most promising methods to reliably improve robustness against such attacks. Yet, in the context of LLMs, current methods for adversarial training are hindered by the high computational costs required to perform discrete adversarial attacks at each training iteration. We address this problem by instead calculating adversarial attacks in the continuous embedding space of the LLM, which is orders of magnitudes more efficient. We propose a fast adversarial training algorithm (C-AdvUL) composed of two losses: the first makes the model robust on continuous embedding attacks computed on an adversarial behaviour dataset; the second ensures the usefulness of the final model by fine-tuning on utility data. Moreover, we introduce C-AdvIPO, an adversarial variant of IPO that does not require utility data for adversarially robust alignment. Our empirical evaluation on five models from different families (Gemma, Phi3, Mistral, Zephyr, Llama2) and at different scales (2B, 3.8B, 7B) shows that both algorithms substantially enhance LLM robustness against discrete attacks (GCG, AutoDAN, PAIR), while maintaining utility. Our results demonstrate that robustness to continuous perturbations can extrapolate to discrete threat models. Thereby, we present a path toward scalable adversarial training algorithms for robustly aligning LLMs.



## **38. Prevailing against Adversarial Noncentral Disturbances: Exact Recovery of Linear Systems with the $l_1$-norm Estimator**

math.OC

Theorem 1 turned out to be incorrect

**SubmitDate**: 2024-11-01    [abs](http://arxiv.org/abs/2410.03218v2) [paper-pdf](http://arxiv.org/pdf/2410.03218v2)

**Authors**: Jihun Kim, Javad Lavaei

**Abstract**: This paper studies the linear system identification problem in the general case where the disturbance is sub-Gaussian, correlated, and possibly adversarial. First, we consider the case with noncentral (nonzero-mean) disturbances for which the ordinary least-squares (OLS) method fails to correctly identify the system. We prove that the $l_1$-norm estimator accurately identifies the system under the condition that each disturbance has equal probabilities of being positive or negative. This condition restricts the sign of each disturbance but allows its magnitude to be arbitrary. Second, we consider the case where each disturbance is adversarial with the model that the attack times happen occasionally but the distributions of the attack values are completely arbitrary. We show that when the probability of having an attack at a given time is less than 0.5, the $l_1$-norm estimator prevails against any adversarial noncentral disturbances and the exact recovery is achieved within a finite time. These results pave the way to effectively defend against arbitrarily large noncentral attacks in safety-critical systems.



## **39. Intruding with Words: Towards Understanding Graph Injection Attacks at the Text Level**

cs.LG

Accepted by NeurIPS 2024

**SubmitDate**: 2024-11-01    [abs](http://arxiv.org/abs/2405.16405v2) [paper-pdf](http://arxiv.org/pdf/2405.16405v2)

**Authors**: Runlin Lei, Yuwei Hu, Yuchen Ren, Zhewei Wei

**Abstract**: Graph Neural Networks (GNNs) excel across various applications but remain vulnerable to adversarial attacks, particularly Graph Injection Attacks (GIAs), which inject malicious nodes into the original graph and pose realistic threats. Text-attributed graphs (TAGs), where nodes are associated with textual features, are crucial due to their prevalence in real-world applications and are commonly used to evaluate these vulnerabilities. However, existing research only focuses on embedding-level GIAs, which inject node embeddings rather than actual textual content, limiting their applicability and simplifying detection. In this paper, we pioneer the exploration of GIAs at the text level, presenting three novel attack designs that inject textual content into the graph. Through theoretical and empirical analysis, we demonstrate that text interpretability, a factor previously overlooked at the embedding level, plays a crucial role in attack strength. Among the designs we investigate, the Word-frequency-based Text-level GIA (WTGIA) is particularly notable for its balance between performance and interpretability. Despite the success of WTGIA, we discover that defenders can easily enhance their defenses with customized text embedding methods or large language model (LLM)--based predictors. These insights underscore the necessity for further research into the potential and practical significance of text-level GIAs.



## **40. Improved Generation of Adversarial Examples Against Safety-aligned LLMs**

cs.CR

**SubmitDate**: 2024-11-01    [abs](http://arxiv.org/abs/2405.20778v2) [paper-pdf](http://arxiv.org/pdf/2405.20778v2)

**Authors**: Qizhang Li, Yiwen Guo, Wangmeng Zuo, Hao Chen

**Abstract**: Adversarial prompts generated using gradient-based methods exhibit outstanding performance in performing automatic jailbreak attacks against safety-aligned LLMs. Nevertheless, due to the discrete nature of texts, the input gradient of LLMs struggles to precisely reflect the magnitude of loss change that results from token replacements in the prompt, leading to limited attack success rates against safety-aligned LLMs, even in the white-box setting. In this paper, we explore a new perspective on this problem, suggesting that it can be alleviated by leveraging innovations inspired in transfer-based attacks that were originally proposed for attacking black-box image classification models. For the first time, we appropriate the ideologies of effective methods among these transfer-based attacks, i.e., Skip Gradient Method and Intermediate Level Attack, into gradient-based adversarial prompt generation and achieve significant performance gains without introducing obvious computational cost. Meanwhile, by discussing mechanisms behind the gains, new insights are drawn, and proper combinations of these methods are also developed. Our empirical results show that 87% of the query-specific adversarial suffixes generated by the developed combination can induce Llama-2-7B-Chat to produce the output that exactly matches the target string on AdvBench. This match rate is 33% higher than that of a very strong baseline known as GCG, demonstrating advanced discrete optimization for adversarial prompt generation against LLMs. In addition, without introducing obvious cost, the combination achieves >30% absolute increase in attack success rates compared with GCG when generating both query-specific (38% -> 68%) and universal adversarial prompts (26.68% -> 60.32%) for attacking the Llama-2-7B-Chat model on AdvBench. Code at: https://github.com/qizhangli/Gradient-based-Jailbreak-Attacks.



## **41. Uncertainty-based Offline Variational Bayesian Reinforcement Learning for Robustness under Diverse Data Corruptions**

cs.LG

Accepted to NeurIPS 2024

**SubmitDate**: 2024-11-01    [abs](http://arxiv.org/abs/2411.00465v1) [paper-pdf](http://arxiv.org/pdf/2411.00465v1)

**Authors**: Rui Yang, Jie Wang, Guoping Wu, Bin Li

**Abstract**: Real-world offline datasets are often subject to data corruptions (such as noise or adversarial attacks) due to sensor failures or malicious attacks. Despite advances in robust offline reinforcement learning (RL), existing methods struggle to learn robust agents under high uncertainty caused by the diverse corrupted data (i.e., corrupted states, actions, rewards, and dynamics), leading to performance degradation in clean environments. To tackle this problem, we propose a novel robust variational Bayesian inference for offline RL (TRACER). It introduces Bayesian inference for the first time to capture the uncertainty via offline data for robustness against all types of data corruptions. Specifically, TRACER first models all corruptions as the uncertainty in the action-value function. Then, to capture such uncertainty, it uses all offline data as the observations to approximate the posterior distribution of the action-value function under a Bayesian inference framework. An appealing feature of TRACER is that it can distinguish corrupted data from clean data using an entropy-based uncertainty measure, since corrupted data often induces higher uncertainty and entropy. Based on the aforementioned measure, TRACER can regulate the loss associated with corrupted data to reduce its influence, thereby enhancing robustness and performance in clean environments. Experiments demonstrate that TRACER significantly outperforms several state-of-the-art approaches across both individual and simultaneous data corruptions.



## **42. Adversarial Purification and Fine-tuning for Robust UDC Image Restoration**

eess.IV

Failure to meet expectations

**SubmitDate**: 2024-11-01    [abs](http://arxiv.org/abs/2402.13629v3) [paper-pdf](http://arxiv.org/pdf/2402.13629v3)

**Authors**: Zhenbo Song, Zhenyuan Zhang, Kaihao Zhang, Zhaoxin Fan, Jianfeng Lu

**Abstract**: This study delves into the enhancement of Under-Display Camera (UDC) image restoration models, focusing on their robustness against adversarial attacks. Despite its innovative approach to seamless display integration, UDC technology faces unique image degradation challenges exacerbated by the susceptibility to adversarial perturbations. Our research initially conducts an in-depth robustness evaluation of deep-learning-based UDC image restoration models by employing several white-box and black-box attacking methods. This evaluation is pivotal in understanding the vulnerabilities of current UDC image restoration techniques. Following the assessment, we introduce a defense framework integrating adversarial purification with subsequent fine-tuning processes. First, our approach employs diffusion-based adversarial purification, effectively neutralizing adversarial perturbations. Then, we apply the fine-tuning methodologies to refine the image restoration models further, ensuring that the quality and fidelity of the restored images are maintained. The effectiveness of our proposed approach is validated through extensive experiments, showing marked improvements in resilience against typical adversarial attacks.



## **43. Towards Building Secure UAV Navigation with FHE-aware Knowledge Distillation**

cs.CR

arXiv admin note: text overlap with arXiv:2404.17225

**SubmitDate**: 2024-11-01    [abs](http://arxiv.org/abs/2411.00403v1) [paper-pdf](http://arxiv.org/pdf/2411.00403v1)

**Authors**: Arjun Ramesh Kaushik, Charanjit Jutla, Nalini Ratha

**Abstract**: In safeguarding mission-critical systems, such as Unmanned Aerial Vehicles (UAVs), preserving the privacy of path trajectories during navigation is paramount. While the combination of Reinforcement Learning (RL) and Fully Homomorphic Encryption (FHE) holds promise, the computational overhead of FHE presents a significant challenge. This paper proposes an innovative approach that leverages Knowledge Distillation to enhance the practicality of secure UAV navigation. By integrating RL and FHE, our framework addresses vulnerabilities to adversarial attacks while enabling real-time processing of encrypted UAV camera feeds, ensuring data security. To mitigate FHE's latency, Knowledge Distillation is employed to compress the network, resulting in an impressive 18x speedup without compromising performance, as evidenced by an R-squared score of 0.9499 compared to the original model's score of 0.9631. Our methodology underscores the feasibility of processing encrypted data for UAV navigation tasks, emphasizing security alongside performance efficiency and timely processing. These findings pave the way for deploying autonomous UAVs in sensitive environments, bolstering their resilience against potential security threats.



## **44. Replace-then-Perturb: Targeted Adversarial Attacks With Visual Reasoning for Vision-Language Models**

cs.CV

13 pages, 5 figure

**SubmitDate**: 2024-11-01    [abs](http://arxiv.org/abs/2411.00898v1) [paper-pdf](http://arxiv.org/pdf/2411.00898v1)

**Authors**: Jonggyu Jang, Hyeonsu Lyu, Jungyeon Koh, Hyun Jong Yang

**Abstract**: The conventional targeted adversarial attacks add a small perturbation to an image to make neural network models estimate the image as a predefined target class, even if it is not the correct target class. Recently, for visual-language models (VLMs), the focus of targeted adversarial attacks is to generate a perturbation that makes VLMs answer intended target text outputs. For example, they aim to make a small perturbation on an image to make VLMs' answers change from "there is an apple" to "there is a baseball." However, answering just intended text outputs is insufficient for tricky questions like "if there is a baseball, tell me what is below it." This is because the target of the adversarial attacks does not consider the overall integrity of the original image, thereby leading to a lack of visual reasoning. In this work, we focus on generating targeted adversarial examples with visual reasoning against VLMs. To this end, we propose 1) a novel adversarial attack procedure -- namely, Replace-then-Perturb and 2) a contrastive learning-based adversarial loss -- namely, Contrastive-Adv. In Replace-then-Perturb, we first leverage a text-guided segmentation model to find the target object in the image. Then, we get rid of the target object and inpaint the empty space with the desired prompt. By doing this, we can generate a target image corresponding to the desired prompt, while maintaining the overall integrity of the original image. Furthermore, in Contrastive-Adv, we design a novel loss function to obtain better adversarial examples. Our extensive benchmark results demonstrate that Replace-then-Perturb and Contrastive-Adv outperform the baseline adversarial attack algorithms. We note that the source code to reproduce the results will be available.



## **45. OSLO: One-Shot Label-Only Membership Inference Attacks**

cs.LG

To appear at NeurIPS 2024

**SubmitDate**: 2024-11-01    [abs](http://arxiv.org/abs/2405.16978v3) [paper-pdf](http://arxiv.org/pdf/2405.16978v3)

**Authors**: Yuefeng Peng, Jaechul Roh, Subhransu Maji, Amir Houmansadr

**Abstract**: We introduce One-Shot Label-Only (OSLO) membership inference attacks (MIAs), which accurately infer a given sample's membership in a target model's training set with high precision using just \emph{a single query}, where the target model only returns the predicted hard label. This is in contrast to state-of-the-art label-only attacks which require $\sim6000$ queries, yet get attack precisions lower than OSLO's. OSLO leverages transfer-based black-box adversarial attacks. The core idea is that a member sample exhibits more resistance to adversarial perturbations than a non-member. We compare OSLO against state-of-the-art label-only attacks and demonstrate that, despite requiring only one query, our method significantly outperforms previous attacks in terms of precision and true positive rate (TPR) under the same false positive rates (FPR). For example, compared to previous label-only MIAs, OSLO achieves a TPR that is at least 7$\times$ higher under a 1\% FPR and at least 22$\times$ higher under a 0.1\% FPR on CIFAR100 for a ResNet18 model. We evaluated multiple defense mechanisms against OSLO.



## **46. Quantum Entanglement Path Selection and Qubit Allocation via Adversarial Group Neural Bandits**

quant-ph

Accepted by IEEE/ACM Transactions on Networking

**SubmitDate**: 2024-11-01    [abs](http://arxiv.org/abs/2411.00316v1) [paper-pdf](http://arxiv.org/pdf/2411.00316v1)

**Authors**: Yin Huang, Lei Wang, Jie Xu

**Abstract**: Quantum Data Networks (QDNs) have emerged as a promising framework in the field of information processing and transmission, harnessing the principles of quantum mechanics. QDNs utilize a quantum teleportation technique through long-distance entanglement connections, encoding data information in quantum bits (qubits). Despite being a cornerstone in various quantum applications, quantum entanglement encounters challenges in establishing connections over extended distances due to probabilistic processes influenced by factors like optical fiber losses. The creation of long-distance entanglement connections between quantum computers involves multiple entanglement links and entanglement swapping techniques through successive quantum nodes, including quantum computers and quantum repeaters, necessitating optimal path selection and qubit allocation. Current research predominantly assumes known success rates of entanglement links between neighboring quantum nodes and overlooks potential network attackers. This paper addresses the online challenge of optimal path selection and qubit allocation, aiming to learn the best strategy for achieving the highest success rate of entanglement connections between two chosen quantum computers without prior knowledge of the success rate and in the presence of a QDN attacker. The proposed approach is based on multi-armed bandits, specifically adversarial group neural bandits, which treat each path as a group and view qubit allocation as arm selection. Our contributions encompass formulating an online adversarial optimization problem, introducing the EXPNeuralUCB bandits algorithm with theoretical performance guarantees, and conducting comprehensive simulations to showcase its superiority over established advanced algorithms.



## **47. Detecting Brittle Decisions for Free: Leveraging Margin Consistency in Deep Robust Classifiers**

cs.LG

10 pages, 6 figures, 2 tables. Version Update: Neurips Camera Ready

**SubmitDate**: 2024-11-01    [abs](http://arxiv.org/abs/2406.18451v3) [paper-pdf](http://arxiv.org/pdf/2406.18451v3)

**Authors**: Jonas Ngnawé, Sabyasachi Sahoo, Yann Pequignot, Frédéric Precioso, Christian Gagné

**Abstract**: Despite extensive research on adversarial training strategies to improve robustness, the decisions of even the most robust deep learning models can still be quite sensitive to imperceptible perturbations, creating serious risks when deploying them for high-stakes real-world applications. While detecting such cases may be critical, evaluating a model's vulnerability at a per-instance level using adversarial attacks is computationally too intensive and unsuitable for real-time deployment scenarios. The input space margin is the exact score to detect non-robust samples and is intractable for deep neural networks. This paper introduces the concept of margin consistency -- a property that links the input space margins and the logit margins in robust models -- for efficient detection of vulnerable samples. First, we establish that margin consistency is a necessary and sufficient condition to use a model's logit margin as a score for identifying non-robust samples. Next, through comprehensive empirical analysis of various robustly trained models on CIFAR10 and CIFAR100 datasets, we show that they indicate high margin consistency with a strong correlation between their input space margins and the logit margins. Then, we show that we can effectively and confidently use the logit margin to detect brittle decisions with such models. Finally, we address cases where the model is not sufficiently margin-consistent by learning a pseudo-margin from the feature representation. Our findings highlight the potential of leveraging deep representations to assess adversarial vulnerability in deployment scenarios efficiently.



## **48. Efficient Model Compression for Bayesian Neural Networks**

cs.LG

**SubmitDate**: 2024-11-01    [abs](http://arxiv.org/abs/2411.00273v1) [paper-pdf](http://arxiv.org/pdf/2411.00273v1)

**Authors**: Diptarka Saha, Zihe Liu, Feng Liang

**Abstract**: Model Compression has drawn much attention within the deep learning community recently. Compressing a dense neural network offers many advantages including lower computation cost, deployability to devices of limited storage and memories, and resistance to adversarial attacks. This may be achieved via weight pruning or fully discarding certain input features. Here we demonstrate a novel strategy to emulate principles of Bayesian model selection in a deep learning setup. Given a fully connected Bayesian neural network with spike-and-slab priors trained via a variational algorithm, we obtain the posterior inclusion probability for every node that typically gets lost. We employ these probabilities for pruning and feature selection on a host of simulated and real-world benchmark data and find evidence of better generalizability of the pruned model in all our experiments.



## **49. JailbreakBench: An Open Robustness Benchmark for Jailbreaking Large Language Models**

cs.CR

The camera-ready version of JailbreakBench v1.0 (accepted at NeurIPS  2024 Datasets and Benchmarks Track): more attack artifacts, more test-time  defenses, a more accurate jailbreak judge (Llama-3-70B with a custom prompt),  a larger dataset of human preferences for selecting a jailbreak judge (300  examples), an over-refusal evaluation dataset, a semantic refusal judge based  on Llama-3-8B

**SubmitDate**: 2024-10-31    [abs](http://arxiv.org/abs/2404.01318v5) [paper-pdf](http://arxiv.org/pdf/2404.01318v5)

**Authors**: Patrick Chao, Edoardo Debenedetti, Alexander Robey, Maksym Andriushchenko, Francesco Croce, Vikash Sehwag, Edgar Dobriban, Nicolas Flammarion, George J. Pappas, Florian Tramer, Hamed Hassani, Eric Wong

**Abstract**: Jailbreak attacks cause large language models (LLMs) to generate harmful, unethical, or otherwise objectionable content. Evaluating these attacks presents a number of challenges, which the current collection of benchmarks and evaluation techniques do not adequately address. First, there is no clear standard of practice regarding jailbreaking evaluation. Second, existing works compute costs and success rates in incomparable ways. And third, numerous works are not reproducible, as they withhold adversarial prompts, involve closed-source code, or rely on evolving proprietary APIs. To address these challenges, we introduce JailbreakBench, an open-sourced benchmark with the following components: (1) an evolving repository of state-of-the-art adversarial prompts, which we refer to as jailbreak artifacts; (2) a jailbreaking dataset comprising 100 behaviors -- both original and sourced from prior work (Zou et al., 2023; Mazeika et al., 2023, 2024) -- which align with OpenAI's usage policies; (3) a standardized evaluation framework at https://github.com/JailbreakBench/jailbreakbench that includes a clearly defined threat model, system prompts, chat templates, and scoring functions; and (4) a leaderboard at https://jailbreakbench.github.io/ that tracks the performance of attacks and defenses for various LLMs. We have carefully considered the potential ethical implications of releasing this benchmark, and believe that it will be a net positive for the community.



## **50. Protecting Feed-Forward Networks from Adversarial Attacks Using Predictive Coding**

cs.CR

**SubmitDate**: 2024-10-31    [abs](http://arxiv.org/abs/2411.00222v1) [paper-pdf](http://arxiv.org/pdf/2411.00222v1)

**Authors**: Ehsan Ganjidoost, Jeff Orchard

**Abstract**: An adversarial example is a modified input image designed to cause a Machine Learning (ML) model to make a mistake; these perturbations are often invisible or subtle to human observers and highlight vulnerabilities in a model's ability to generalize from its training data. Several adversarial attacks can create such examples, each with a different perspective, effectiveness, and perceptibility of changes. Conversely, defending against such adversarial attacks improves the robustness of ML models in image processing and other domains of deep learning. Most defence mechanisms require either a level of model awareness, changes to the model, or access to a comprehensive set of adversarial examples during training, which is impractical. Another option is to use an auxiliary model in a preprocessing manner without changing the primary model. This study presents a practical and effective solution -- using predictive coding networks (PCnets) as an auxiliary step for adversarial defence. By seamlessly integrating PCnets into feed-forward networks as a preprocessing step, we substantially bolster resilience to adversarial perturbations. Our experiments on MNIST and CIFAR10 demonstrate the remarkable effectiveness of PCnets in mitigating adversarial examples with about 82% and 65% improvements in robustness, respectively. The PCnet, trained on a small subset of the dataset, leverages its generative nature to effectively counter adversarial efforts, reverting perturbed images closer to their original forms. This innovative approach holds promise for enhancing the security and reliability of neural network classifiers in the face of the escalating threat of adversarial attacks.



