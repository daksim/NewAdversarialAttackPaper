# Latest Adversarial Attack Papers
**update at 2024-11-28 10:19:10**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

[Attacks and Defenses in Large language Models](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM.md)

## **1. GSE: Group-wise Sparse and Explainable Adversarial Attacks**

cs.CV

**SubmitDate**: 2024-11-27    [abs](http://arxiv.org/abs/2311.17434v4) [paper-pdf](http://arxiv.org/pdf/2311.17434v4)

**Authors**: Shpresim Sadiku, Moritz Wagner, Sebastian Pokutta

**Abstract**: Sparse adversarial attacks fool deep neural networks (DNNs) through minimal pixel perturbations, often regularized by the $\ell_0$ norm. Recent efforts have replaced this norm with a structural sparsity regularizer, such as the nuclear group norm, to craft group-wise sparse adversarial attacks. The resulting perturbations are thus explainable and hold significant practical relevance, shedding light on an even greater vulnerability of DNNs. However, crafting such attacks poses an optimization challenge, as it involves computing norms for groups of pixels within a non-convex objective. We address this by presenting a two-phase algorithm that generates group-wise sparse attacks within semantically meaningful areas of an image. Initially, we optimize a quasinorm adversarial loss using the $1/2-$quasinorm proximal operator tailored for non-convex programming. Subsequently, the algorithm transitions to a projected Nesterov's accelerated gradient descent with $2-$norm regularization applied to perturbation magnitudes. Rigorous evaluations on CIFAR-10 and ImageNet datasets demonstrate a remarkable increase in group-wise sparsity, e.g., $50.9\%$ on CIFAR-10 and $38.4\%$ on ImageNet (average case, targeted attack). This performance improvement is accompanied by significantly faster computation times, improved explainability, and a $100\%$ attack success rate.



## **2. Don't Command, Cultivate: An Exploratory Study of System-2 Alignment**

cs.CL

Preprint version, more results will be updated

**SubmitDate**: 2024-11-27    [abs](http://arxiv.org/abs/2411.17075v2) [paper-pdf](http://arxiv.org/pdf/2411.17075v2)

**Authors**: Yuhang Wang, Jitao Sang

**Abstract**: The o1 system card identifies the o1 models as the most robust within OpenAI, with their defining characteristic being the progression from rapid, intuitive thinking to slower, more deliberate reasoning. This observation motivated us to investigate the influence of System-2 thinking patterns on model safety. In our preliminary research, we conducted safety evaluations of the o1 model, including complex jailbreak attack scenarios using adversarial natural language prompts and mathematical encoding prompts. Our findings indicate that the o1 model demonstrates relatively improved safety performance; however, it still exhibits vulnerabilities, particularly against jailbreak attacks employing mathematical encoding. Through detailed case analysis, we identified specific patterns in the o1 model's responses. We also explored the alignment of System-2 safety in open-source models using prompt engineering and supervised fine-tuning techniques. Experimental results show that some simple methods to encourage the model to carefully scrutinize user requests are beneficial for model safety. Additionally, we proposed a implementation plan for process supervision to enhance safety alignment. The implementation details and experimental results will be provided in future versions.



## **3. Visual Adversarial Attack on Vision-Language Models for Autonomous Driving**

cs.CV

**SubmitDate**: 2024-11-27    [abs](http://arxiv.org/abs/2411.18275v1) [paper-pdf](http://arxiv.org/pdf/2411.18275v1)

**Authors**: Tianyuan Zhang, Lu Wang, Xinwei Zhang, Yitong Zhang, Boyi Jia, Siyuan Liang, Shengshan Hu, Qiang Fu, Aishan Liu, Xianglong Liu

**Abstract**: Vision-language models (VLMs) have significantly advanced autonomous driving (AD) by enhancing reasoning capabilities. However, these models remain highly vulnerable to adversarial attacks. While existing research has primarily focused on general VLM attacks, the development of attacks tailored to the safety-critical AD context has been largely overlooked. In this paper, we take the first step toward designing adversarial attacks specifically targeting VLMs in AD, exposing the substantial risks these attacks pose within this critical domain. We identify two unique challenges for effective adversarial attacks on AD VLMs: the variability of textual instructions and the time-series nature of visual scenarios. To this end, we propose ADvLM, the first visual adversarial attack framework specifically designed for VLMs in AD. Our framework introduces Semantic-Invariant Induction, which uses a large language model to create a diverse prompt library of textual instructions with consistent semantic content, guided by semantic entropy. Building on this, we introduce Scenario-Associated Enhancement, an approach where attention mechanisms select key frames and perspectives within driving scenarios to optimize adversarial perturbations that generalize across the entire scenario. Extensive experiments on several AD VLMs over multiple benchmarks show that ADvLM achieves state-of-the-art attack effectiveness. Moreover, real-world attack studies further validate its applicability and potential in practice.



## **4. G-Designer: Architecting Multi-agent Communication Topologies via Graph Neural Networks**

cs.MA

**SubmitDate**: 2024-11-27    [abs](http://arxiv.org/abs/2410.11782v2) [paper-pdf](http://arxiv.org/pdf/2410.11782v2)

**Authors**: Guibin Zhang, Yanwei Yue, Xiangguo Sun, Guancheng Wan, Miao Yu, Junfeng Fang, Kun Wang, Dawei Cheng

**Abstract**: Recent advancements in large language model (LLM)-based agents have demonstrated that collective intelligence can significantly surpass the capabilities of individual agents, primarily due to well-crafted inter-agent communication topologies. Despite the diverse and high-performing designs available, practitioners often face confusion when selecting the most effective pipeline for their specific task: \textit{Which topology is the best choice for my task, avoiding unnecessary communication token overhead while ensuring high-quality solution?} In response to this dilemma, we introduce G-Designer, an adaptive, efficient, and robust solution for multi-agent deployment, which dynamically designs task-aware, customized communication topologies. Specifically, G-Designer models the multi-agent system as a multi-agent network, leveraging a variational graph auto-encoder to encode both the nodes (agents) and a task-specific virtual node, and decodes a task-adaptive and high-performing communication topology. Extensive experiments on six benchmarks showcase that G-Designer is: \textbf{(1) high-performing}, achieving superior results on MMLU with accuracy at $84.50\%$ and on HumanEval with pass@1 at $89.90\%$; \textbf{(2) task-adaptive}, architecting communication protocols tailored to task difficulty, reducing token consumption by up to $95.33\%$ on HumanEval; and \textbf{(3) adversarially robust}, defending against agent adversarial attacks with merely $0.3\%$ accuracy drop.



## **5. R-MTLLMF: Resilient Multi-Task Large Language Model Fusion at the Wireless Edge**

eess.SP

**SubmitDate**: 2024-11-27    [abs](http://arxiv.org/abs/2411.18220v1) [paper-pdf](http://arxiv.org/pdf/2411.18220v1)

**Authors**: Aladin Djuhera, Vlad C. Andrei, Mohsen Pourghasemian, Haris Gacanin, Holger Boche, Walid Saad

**Abstract**: Multi-task large language models (MTLLMs) are important for many applications at the wireless edge, where users demand specialized models to handle multiple tasks efficiently. However, training MTLLMs is complex and exhaustive, particularly when tasks are subject to change. Recently, the concept of model fusion via task vectors has emerged as an efficient approach for combining fine-tuning parameters to produce an MTLLM. In this paper, the problem of enabling edge users to collaboratively craft such MTTLMs via tasks vectors is studied, under the assumption of worst-case adversarial attacks. To this end, first the influence of adversarial noise to multi-task model fusion is investigated and a relationship between the so-called weight disentanglement error and the mean squared error (MSE) is derived. Using hypothesis testing, it is directly shown that the MSE increases interference between task vectors, thereby rendering model fusion ineffective. Then, a novel resilient MTLLM fusion (R-MTLLMF) is proposed, which leverages insights about the LLM architecture and fine-tuning process to safeguard task vector aggregation under adversarial noise by realigning the MTLLM. The proposed R-MTLLMF is then compared for both worst-case and ideal transmission scenarios to study the impact of the wireless channel. Extensive model fusion experiments with vision LLMs demonstrate R-MTLLMF's effectiveness, achieving close-to-baseline performance across eight different tasks in ideal noise scenarios and significantly outperforming unprotected model fusion in worst-case scenarios. The results further advocate for additional physical layer protection for a holistic approach to resilience, from both a wireless and LLM perspective.



## **6. Privacy-preserving Robotic-based Multi-factor Authentication Scheme for Secure Automated Delivery System**

cs.CR

**SubmitDate**: 2024-11-27    [abs](http://arxiv.org/abs/2411.18027v1) [paper-pdf](http://arxiv.org/pdf/2411.18027v1)

**Authors**: Yang Yang, Aryan Mohammadi Pasikhani, Prosanta Gope, Biplab Sikdar

**Abstract**: Package delivery is a critical aspect of various industries, but it often incurs high financial costs and inefficiencies when relying solely on human resources. The last-mile transport problem, in particular, contributes significantly to the expenditure of human resources in major companies. Robot-based delivery systems have emerged as a potential solution for last-mile delivery to address this challenge. However, robotic delivery systems still face security and privacy issues, like impersonation, replay, man-in-the-middle attacks (MITM), unlinkability, and identity theft. In this context, we propose a privacy-preserving multi-factor authentication scheme specifically designed for robot delivery systems. Additionally, AI-assisted robotic delivery systems are susceptible to machine learning-based attacks (e.g. FGSM, PGD, etc.). We introduce the \emph{first} transformer-based audio-visual fusion defender to tackle this issue, which effectively provides resilience against adversarial samples. Furthermore, we provide a rigorous formal analysis of the proposed protocol and also analyse the protocol security using a popular symbolic proof tool called ProVerif and Scyther. Finally, we present a real-world implementation of the proposed robotic system with the computation cost and energy consumption analysis. Code and pre-trained models are available at: https://drive.google.com/drive/folders/18B2YbxtV0Pyj5RSFX-ZzCGtFOyorBHil



## **7. Leveraging A New GAN-based Transformer with ECDH Crypto-system for Enhancing Energy Theft Detection in Smart Grid**

cs.CR

**SubmitDate**: 2024-11-27    [abs](http://arxiv.org/abs/2411.18023v1) [paper-pdf](http://arxiv.org/pdf/2411.18023v1)

**Authors**: Yang Yang, Xun Yuan, Arwa Alromih, Aryan Mohammadi Pasikhani, Prosanta Gope, Biplab Sikdar

**Abstract**: Detecting energy theft is vital for effectively managing power grids, as it ensures precise billing and prevents financial losses. Split-learning emerges as a promising decentralized machine learning technique for identifying energy theft while preserving user data confidentiality. Nevertheless, traditional split learning approaches are vulnerable to privacy leakage attacks, which significantly threaten data confidentiality. To address this challenge, we propose a novel GAN-Transformer-based split learning framework in this paper. This framework leverages the strengths of the transformer architecture, which is known for its capability to process long-range dependencies in energy consumption data. Thus, it enhances the accuracy of energy theft detection without compromising user privacy. A distinctive feature of our approach is the deployment of a novel mask-based method, marking a first in its field to effectively combat privacy leakage in split learning scenarios targeted at AI-enabled adversaries. This method protects sensitive information during the model's training phase. Our experimental evaluations indicate that the proposed framework not only achieves accuracy levels comparable to conventional methods but also significantly enhances privacy protection. The results underscore the potential of the GAN-Transformer split learning framework as an effective and secure tool in the domain of energy theft detection.



## **8. Exploring Visual Vulnerabilities via Multi-Loss Adversarial Search for Jailbreaking Vision-Language Models**

cs.CV

**SubmitDate**: 2024-11-27    [abs](http://arxiv.org/abs/2411.18000v1) [paper-pdf](http://arxiv.org/pdf/2411.18000v1)

**Authors**: Shuyang Hao, Bryan Hooi, Jun Liu, Kai-Wei Chang, Zi Huang, Yujun Cai

**Abstract**: Despite inheriting security measures from underlying language models, Vision-Language Models (VLMs) may still be vulnerable to safety alignment issues. Through empirical analysis, we uncover two critical findings: scenario-matched images can significantly amplify harmful outputs, and contrary to common assumptions in gradient-based attacks, minimal loss values do not guarantee optimal attack effectiveness. Building on these insights, we introduce MLAI (Multi-Loss Adversarial Images), a novel jailbreak framework that leverages scenario-aware image generation for semantic alignment, exploits flat minima theory for robust adversarial image selection, and employs multi-image collaborative attacks for enhanced effectiveness. Extensive experiments demonstrate MLAI's significant impact, achieving attack success rates of 77.75% on MiniGPT-4 and 82.80% on LLaVA-2, substantially outperforming existing methods by margins of 34.37% and 12.77% respectively. Furthermore, MLAI shows considerable transferability to commercial black-box VLMs, achieving up to 60.11% success rate. Our work reveals fundamental visual vulnerabilities in current VLMs safety mechanisms and underscores the need for stronger defenses. Warning: This paper contains potentially harmful example text.



## **9. Adversarial Training in Low-Label Regimes with Margin-Based Interpolation**

cs.LG

**SubmitDate**: 2024-11-27    [abs](http://arxiv.org/abs/2411.17959v1) [paper-pdf](http://arxiv.org/pdf/2411.17959v1)

**Authors**: Tian Ye, Rajgopal Kannan, Viktor Prasanna

**Abstract**: Adversarial training has emerged as an effective approach to train robust neural network models that are resistant to adversarial attacks, even in low-label regimes where labeled data is scarce. In this paper, we introduce a novel semi-supervised adversarial training approach that enhances both robustness and natural accuracy by generating effective adversarial examples. Our method begins by applying linear interpolation between clean and adversarial examples to create interpolated adversarial examples that cross decision boundaries by a controlled margin. This sample-aware strategy tailors adversarial examples to the characteristics of each data point, enabling the model to learn from the most informative perturbations. Additionally, we propose a global epsilon scheduling strategy that progressively adjusts the upper bound of perturbation strengths during training. The combination of these strategies allows the model to develop increasingly complex decision boundaries with better robustness and natural accuracy. Empirical evaluations show that our approach effectively enhances performance against various adversarial attacks, such as PGD and AutoAttack.



## **10. Stealthy Multi-Task Adversarial Attacks**

cs.CR

**SubmitDate**: 2024-11-26    [abs](http://arxiv.org/abs/2411.17936v1) [paper-pdf](http://arxiv.org/pdf/2411.17936v1)

**Authors**: Jiacheng Guo, Tianyun Zhang, Lei Li, Haochen Yang, Hongkai Yu, Minghai Qin

**Abstract**: Deep Neural Networks exhibit inherent vulnerabilities to adversarial attacks, which can significantly compromise their outputs and reliability. While existing research primarily focuses on attacking single-task scenarios or indiscriminately targeting all tasks in multi-task environments, we investigate selectively targeting one task while preserving performance in others within a multi-task framework. This approach is motivated by varying security priorities among tasks in real-world applications, such as autonomous driving, where misinterpreting critical objects (e.g., signs, traffic lights) poses a greater security risk than minor depth miscalculations. Consequently, attackers may hope to target security-sensitive tasks while avoiding non-critical tasks from being compromised, thus evading being detected before compromising crucial functions. In this paper, we propose a method for the stealthy multi-task attack framework that utilizes multiple algorithms to inject imperceptible noise into the input. This novel method demonstrates remarkable efficacy in compromising the target task while simultaneously maintaining or even enhancing performance across non-targeted tasks - a criterion hitherto unexplored in the field. Additionally, we introduce an automated approach for searching the weighting factors in the loss function, further enhancing attack efficiency. Experimental results validate our framework's ability to successfully attack the target task while preserving the performance of non-targeted tasks. The automated loss function weight searching method demonstrates comparable efficacy to manual tuning, establishing a state-of-the-art multi-task attack framework.



## **11. Enhancing Robustness in Deep Reinforcement Learning: A Lyapunov Exponent Approach**

cs.LG

**SubmitDate**: 2024-11-26    [abs](http://arxiv.org/abs/2410.10674v2) [paper-pdf](http://arxiv.org/pdf/2410.10674v2)

**Authors**: Rory Young, Nicolas Pugeault

**Abstract**: Deep reinforcement learning agents achieve state-of-the-art performance in a wide range of simulated control tasks. However, successful applications to real-world problems remain limited. One reason for this dichotomy is because the learnt policies are not robust to observation noise or adversarial attacks. In this paper, we investigate the robustness of deep RL policies to a single small state perturbation in deterministic continuous control tasks. We demonstrate that RL policies can be deterministically chaotic, as small perturbations to the system state have a large impact on subsequent state and reward trajectories. This unstable non-linear behaviour has two consequences: first, inaccuracies in sensor readings, or adversarial attacks, can cause significant performance degradation; second, even policies that show robust performance in terms of rewards may have unpredictable behaviour in practice. These two facets of chaos in RL policies drastically restrict the application of deep RL to real-world problems. To address this issue, we propose an improvement on the successful Dreamer V3 architecture, implementing Maximal Lyapunov Exponent regularisation. This new approach reduces the chaotic state dynamics, rendering the learnt policies more resilient to sensor noise or adversarial attacks and thereby improving the suitability of deep reinforcement learning for real-world applications.



## **12. TrackPGD: Efficient Adversarial Attack using Object Binary Masks against Robust Transformer Trackers**

cs.CV

Accepted in The 3rd New Frontiers in Adversarial Machine Learning  (AdvML Frontiers @NeurIPS2024)

**SubmitDate**: 2024-11-26    [abs](http://arxiv.org/abs/2407.03946v2) [paper-pdf](http://arxiv.org/pdf/2407.03946v2)

**Authors**: Fatemeh Nourilenjan Nokabadi, Yann Batiste Pequignot, Jean-Francois Lalonde, Christian Gagné

**Abstract**: Adversarial perturbations can deceive neural networks by adding small, imperceptible noise to the input. Recent object trackers with transformer backbones have shown strong performance on tracking datasets, but their adversarial robustness has not been thoroughly evaluated. While transformer trackers are resilient to black-box attacks, existing white-box adversarial attacks are not universally applicable against these new transformer trackers due to differences in backbone architecture. In this work, we introduce TrackPGD, a novel white-box attack that utilizes predicted object binary masks to target robust transformer trackers. Built upon the powerful segmentation attack SegPGD, our proposed TrackPGD effectively influences the decisions of transformer-based trackers. Our method addresses two primary challenges in adapting a segmentation attack for trackers: limited class numbers and extreme pixel class imbalance. TrackPGD uses the same number of iterations as other attack methods for tracker networks and produces competitive adversarial examples that mislead transformer and non-transformer trackers such as MixFormerM, OSTrackSTS, TransT-SEG, and RTS on datasets including VOT2022STS, DAVIS2016, UAV123, and GOT-10k.



## **13. Mitigating the Impact of Noisy Edges on Graph-Based Algorithms via Adversarial Robustness Evaluation**

cs.LG

**SubmitDate**: 2024-11-26    [abs](http://arxiv.org/abs/2401.15615v2) [paper-pdf](http://arxiv.org/pdf/2401.15615v2)

**Authors**: Yongyu Wang, Xiaotian Zhuang

**Abstract**: Given that no existing graph construction method can generate a perfect graph for a given dataset, graph-based algorithms are often affected by redundant and erroneous edges present within the constructed graphs. In this paper, we view these noisy edges as adversarial attack and propose to use a spectral adversarial robustness evaluation method to mitigate the impact of noisy edges on the performance of graph-based algorithms. Our method identifies the points that are less vulnerable to noisy edges and leverages only these robust points to perform graph-based algorithms. Our experiments demonstrate that our methodology is highly effective and outperforms state-of-the-art denoising methods by a large margin.



## **14. Adversarial Bounding Boxes Generation (ABBG) Attack against Visual Object Trackers**

cs.CV

Accepted in The 3rd New Frontiers in Adversarial Machine Learning  (AdvML Frontiers @NeurIPS2024)

**SubmitDate**: 2024-11-26    [abs](http://arxiv.org/abs/2411.17468v1) [paper-pdf](http://arxiv.org/pdf/2411.17468v1)

**Authors**: Fatemeh Nourilenjan Nokabadi, Jean-Francois Lalonde, Christian Gagné

**Abstract**: Adversarial perturbations aim to deceive neural networks into predicting inaccurate results. For visual object trackers, adversarial attacks have been developed to generate perturbations by manipulating the outputs. However, transformer trackers predict a specific bounding box instead of an object candidate list, which limits the applicability of many existing attack scenarios. To address this issue, we present a novel white-box approach to attack visual object trackers with transformer backbones using only one bounding box. From the tracker predicted bounding box, we generate a list of adversarial bounding boxes and compute the adversarial loss for those bounding boxes. Experimental results demonstrate that our simple yet effective attack outperforms existing attacks against several robust transformer trackers, including TransT-M, ROMTrack, and MixFormer, on popular benchmark tracking datasets such as GOT-10k, UAV123, and VOT2022STS.



## **15. PEFTGuard: Detecting Backdoor Attacks Against Parameter-Efficient Fine-Tuning**

cs.CR

20 pages, 8 figures

**SubmitDate**: 2024-11-26    [abs](http://arxiv.org/abs/2411.17453v1) [paper-pdf](http://arxiv.org/pdf/2411.17453v1)

**Authors**: Zhen Sun, Tianshuo Cong, Yule Liu, Chenhao Lin, Xinlei He, Rongmao Chen, Xingshuo Han, Xinyi Huang

**Abstract**: Fine-tuning is an essential process to improve the performance of Large Language Models (LLMs) in specific domains, with Parameter-Efficient Fine-Tuning (PEFT) gaining popularity due to its capacity to reduce computational demands through the integration of low-rank adapters. These lightweight adapters, such as LoRA, can be shared and utilized on open-source platforms. However, adversaries could exploit this mechanism to inject backdoors into these adapters, resulting in malicious behaviors like incorrect or harmful outputs, which pose serious security risks to the community. Unfortunately, few of the current efforts concentrate on analyzing the backdoor patterns or detecting the backdoors in the adapters.   To fill this gap, we first construct (and will release) PADBench, a comprehensive benchmark that contains 13,300 benign and backdoored adapters fine-tuned with various datasets, attack strategies, PEFT methods, and LLMs. Moreover, we propose PEFTGuard, the first backdoor detection framework against PEFT-based adapters. Extensive evaluation upon PADBench shows that PEFTGuard outperforms existing detection methods, achieving nearly perfect detection accuracy (100%) in most cases. Notably, PEFTGuard exhibits zero-shot transferability on three aspects, including different attacks, PEFT methods, and adapter ranks. In addition, we consider various adaptive attacks to demonstrate the high robustness of PEFTGuard. We further explore several possible backdoor mitigation defenses, finding fine-mixing to be the most effective method. We envision our benchmark and method can shed light on future LLM backdoor detection research.



## **16. Breaking the Illusion: Real-world Challenges for Adversarial Patches in Object Detection**

cs.CV

This paper has been accepted by the 1st Workshop on Enabling Machine  Learning Operations for next-Gen Embedded Wireless Networked Devices  (EMERGE), 2024

**SubmitDate**: 2024-11-26    [abs](http://arxiv.org/abs/2410.19863v2) [paper-pdf](http://arxiv.org/pdf/2410.19863v2)

**Authors**: Jakob Shack, Katarina Petrovic, Olga Saukh

**Abstract**: Adversarial attacks pose a significant threat to the robustness and reliability of machine learning systems, particularly in computer vision applications. This study investigates the performance of adversarial patches for the YOLO object detection network in the physical world. Two attacks were tested: a patch designed to be placed anywhere within the scene - global patch, and another patch intended to partially overlap with specific object targeted for removal from detection - local patch. Various factors such as patch size, position, rotation, brightness, and hue were analyzed to understand their impact on the effectiveness of the adversarial patches. The results reveal a notable dependency on these parameters, highlighting the challenges in maintaining attack efficacy in real-world conditions. Learning to align digitally applied transformation parameters with those measured in the real world still results in up to a 64\% discrepancy in patch performance. These findings underscore the importance of understanding environmental influences on adversarial attacks, which can inform the development of more robust defenses for practical machine learning applications.



## **17. Enhancing generalization in high energy physics using white-box adversarial attacks**

hep-ph

10 pages, 4 figures, 8 tables, 3 algorithms, to be published in  Physical Review D (PRD), presented at the ML4Jets 2024 conference

**SubmitDate**: 2024-11-26    [abs](http://arxiv.org/abs/2411.09296v2) [paper-pdf](http://arxiv.org/pdf/2411.09296v2)

**Authors**: Franck Rothen, Samuel Klein, Matthew Leigh, Tobias Golling

**Abstract**: Machine learning is becoming increasingly popular in the context of particle physics. Supervised learning, which uses labeled Monte Carlo (MC) simulations, remains one of the most widely used methods for discriminating signals beyond the Standard Model. However, this paper suggests that supervised models may depend excessively on artifacts and approximations from Monte Carlo simulations, potentially limiting their ability to generalize well to real data. This study aims to enhance the generalization properties of supervised models by reducing the sharpness of local minima. It reviews the application of four distinct white-box adversarial attacks in the context of classifying Higgs boson decay signals. The attacks are divided into weight space attacks, and feature space attacks. To study and quantify the sharpness of different local minima this paper presents two analysis methods: gradient ascent and reduced Hessian eigenvalue analysis. The results show that white-box adversarial attacks significantly improve generalization performance, albeit with increased computational complexity.



## **18. BadScan: An Architectural Backdoor Attack on Visual State Space Models**

cs.CV

**SubmitDate**: 2024-11-26    [abs](http://arxiv.org/abs/2411.17283v1) [paper-pdf](http://arxiv.org/pdf/2411.17283v1)

**Authors**: Om Suhas Deshmukh, Sankalp Nagaonkar, Achyut Mani Tripathi, Ashish Mishra

**Abstract**: The newly introduced Visual State Space Model (VMamba), which employs \textit{State Space Mechanisms} (SSM) to interpret images as sequences of patches, has shown exceptional performance compared to Vision Transformers (ViT) across various computer vision tasks. However, recent studies have highlighted that deep models are susceptible to adversarial attacks. One common approach is to embed a trigger in the training data to retrain the model, causing it to misclassify data samples into a target class, a phenomenon known as a backdoor attack. In this paper, we first evaluate the robustness of the VMamba model against existing backdoor attacks. Based on this evaluation, we introduce a novel architectural backdoor attack, termed BadScan, designed to deceive the VMamba model. This attack utilizes bit plane slicing to create visually imperceptible backdoored images. During testing, if a trigger is detected by performing XOR operations between the $k^{th}$ bit planes of the modified triggered patches, the traditional 2D selective scan (SS2D) mechanism in the visual state space (VSS) block of VMamba is replaced with our newly designed BadScan block, which incorporates four newly developed scanning patterns. We demonstrate that the BadScan backdoor attack represents a significant threat to visual state space models and remains effective even after complete retraining from scratch. Experimental results on two widely used image classification datasets, CIFAR-10, and ImageNet-1K, reveal that while visual state space models generally exhibit robustness against current backdoor attacks, the BadScan attack is particularly effective, achieving a higher Triggered Accuracy Ratio (TAR) in misleading the VMamba model and its variants.



## **19. BadSFL: Backdoor Attack against Scaffold Federated Learning**

cs.LG

**SubmitDate**: 2024-11-26    [abs](http://arxiv.org/abs/2411.16167v2) [paper-pdf](http://arxiv.org/pdf/2411.16167v2)

**Authors**: Xingshuo Han, Xuanye Zhang, Xiang Lan, Haozhao Wang, Shengmin Xu, Shen Ren, Jason Zeng, Ming Wu, Michael Heinrich, Tianwei Zhang

**Abstract**: Federated learning (FL) enables the training of deep learning models on distributed clients to preserve data privacy. However, this learning paradigm is vulnerable to backdoor attacks, where malicious clients can upload poisoned local models to embed backdoors into the global model, leading to attacker-desired predictions. Existing backdoor attacks mainly focus on FL with independently and identically distributed (IID) scenarios, while real-world FL training data are typically non-IID. Current strategies for non-IID backdoor attacks suffer from limitations in maintaining effectiveness and durability. To address these challenges, we propose a novel backdoor attack method, BadSFL, specifically designed for the FL framework using the scaffold aggregation algorithm in non-IID settings. BadSFL leverages a Generative Adversarial Network (GAN) based on the global model to complement the training set, achieving high accuracy on both backdoor and benign samples. It utilizes a specific feature as the backdoor trigger to ensure stealthiness, and exploits the Scaffold's control variate to predict the global model's convergence direction, ensuring the backdoor's persistence. Extensive experiments on three benchmark datasets demonstrate the high effectiveness, stealthiness, and durability of BadSFL. Notably, our attack remains effective over 60 rounds in the global model and up to 3 times longer than existing baseline attacks after stopping the injection of malicious updates.



## **20. Practical Membership Inference Attacks against Fine-tuned Large Language Models via Self-prompt Calibration**

cs.CL

Repo: https://github.com/tsinghua-fib-lab/NeurIPS2024_SPV-MIA

**SubmitDate**: 2024-11-26    [abs](http://arxiv.org/abs/2311.06062v4) [paper-pdf](http://arxiv.org/pdf/2311.06062v4)

**Authors**: Wenjie Fu, Huandong Wang, Chen Gao, Guanghua Liu, Yong Li, Tao Jiang

**Abstract**: Membership Inference Attacks (MIA) aim to infer whether a target data record has been utilized for model training or not. Existing MIAs designed for large language models (LLMs) can be bifurcated into two types: reference-free and reference-based attacks. Although reference-based attacks appear promising performance by calibrating the probability measured on the target model with reference models, this illusion of privacy risk heavily depends on a reference dataset that closely resembles the training set. Both two types of attacks are predicated on the hypothesis that training records consistently maintain a higher probability of being sampled. However, this hypothesis heavily relies on the overfitting of target models, which will be mitigated by multiple regularization methods and the generalization of LLMs. Thus, these reasons lead to high false-positive rates of MIAs in practical scenarios. We propose a Membership Inference Attack based on Self-calibrated Probabilistic Variation (SPV-MIA). Specifically, we introduce a self-prompt approach, which constructs the dataset to fine-tune the reference model by prompting the target LLM itself. In this manner, the adversary can collect a dataset with a similar distribution from public APIs. Furthermore, we introduce probabilistic variation, a more reliable membership signal based on LLM memorization rather than overfitting, from which we rediscover the neighbour attack with theoretical grounding. Comprehensive evaluation conducted on three datasets and four exemplary LLMs shows that SPV-MIA raises the AUC of MIAs from 0.7 to a significantly high level of 0.9. Our code and dataset are available at: https://github.com/tsinghua-fib-lab/NeurIPS2024_SPV-MIA



## **21. RED: Robust Environmental Design**

cs.CV

**SubmitDate**: 2024-11-26    [abs](http://arxiv.org/abs/2411.17026v1) [paper-pdf](http://arxiv.org/pdf/2411.17026v1)

**Authors**: Jinghan Yang

**Abstract**: The classification of road signs by autonomous systems, especially those reliant on visual inputs, is highly susceptible to adversarial attacks. Traditional approaches to mitigating such vulnerabilities have focused on enhancing the robustness of classification models. In contrast, this paper adopts a fundamentally different strategy aimed at increasing robustness through the redesign of road signs themselves. We propose an attacker-agnostic learning scheme to automatically design road signs that are robust to a wide array of patch-based attacks. Empirical tests conducted in both digital and physical environments demonstrate that our approach significantly reduces vulnerability to patch attacks, outperforming existing techniques.



## **22. Unlocking The Potential of Adaptive Attacks on Diffusion-Based Purification**

cs.CR

**SubmitDate**: 2024-11-25    [abs](http://arxiv.org/abs/2411.16598v1) [paper-pdf](http://arxiv.org/pdf/2411.16598v1)

**Authors**: Andre Kassis, Urs Hengartner, Yaoliang Yu

**Abstract**: Diffusion-based purification (DBP) is a defense against adversarial examples (AEs), amassing popularity for its ability to protect classifiers in an attack-oblivious manner and resistance to strong adversaries with access to the defense. Its robustness has been claimed to ensue from the reliance on diffusion models (DMs) that project the AEs onto the natural distribution. We revisit this claim, focusing on gradient-based strategies that back-propagate the loss gradients through the defense, commonly referred to as ``adaptive attacks". Analytically, we show that such an optimization method invalidates DBP's core foundations, effectively targeting the DM rather than the classifier and restricting the purified outputs to a distribution over malicious samples instead. Thus, we reassess the reported empirical robustness, uncovering implementation flaws in the gradient back-propagation techniques used thus far for DBP. We fix these issues, providing the first reliable gradient library for DBP and demonstrating how adaptive attacks drastically degrade its robustness. We then study a less efficient yet stricter majority-vote setting where the classifier evaluates multiple purified copies of the input to make its decision. Here, DBP's stochasticity enables it to remain partially robust against traditional norm-bounded AEs. We propose a novel adaptation of a recent optimization method against deepfake watermarking that crafts systemic malicious perturbations while ensuring imperceptibility. When integrated with the adaptive attack, it completely defeats DBP, even in the majority-vote setup. Our findings prove that DBP, in its current state, is not a viable defense against AEs.



## **23. Adversarial Attacks for Drift Detection**

cs.LG

**SubmitDate**: 2024-11-25    [abs](http://arxiv.org/abs/2411.16591v1) [paper-pdf](http://arxiv.org/pdf/2411.16591v1)

**Authors**: Fabian Hinder, Valerie Vaquet, Barbara Hammer

**Abstract**: Concept drift refers to the change of data distributions over time. While drift poses a challenge for learning models, requiring their continual adaption, it is also relevant in system monitoring to detect malfunctions, system failures, and unexpected behavior. In the latter case, the robust and reliable detection of drifts is imperative. This work studies the shortcomings of commonly used drift detection schemes. We show how to construct data streams that are drifting without being detected. We refer to those as drift adversarials. In particular, we compute all possible adversairals for common detection schemes and underpin our theoretical findings with empirical evaluations.



## **24. XAI and Android Malware Models**

cs.CR

**SubmitDate**: 2024-11-25    [abs](http://arxiv.org/abs/2411.16817v1) [paper-pdf](http://arxiv.org/pdf/2411.16817v1)

**Authors**: Maithili Kulkarni, Mark Stamp

**Abstract**: Android malware detection based on machine learning (ML) and deep learning (DL) models is widely used for mobile device security. Such models offer benefits in terms of detection accuracy and efficiency, but it is often difficult to understand how such learning models make decisions. As a result, these popular malware detection strategies are generally treated as black boxes, which can result in a lack of trust in the decisions made, as well as making adversarial attacks more difficult to detect. The field of eXplainable Artificial Intelligence (XAI) attempts to shed light on such black box models. In this paper, we apply XAI techniques to ML and DL models that have been trained on a challenging Android malware classification problem. Specifically, the classic ML models considered are Support Vector Machines (SVM), Random Forest, and $k$-Nearest Neighbors ($k$-NN), while the DL models we consider are Multi-Layer Perceptrons (MLP) and Convolutional Neural Networks (CNN). The state-of-the-art XAI techniques that we apply to these trained models are Local Interpretable Model-agnostic Explanations (LIME), Shapley Additive exPlanations (SHAP), PDP plots, ELI5, and Class Activation Mapping (CAM). We obtain global and local explanation results, and we discuss the utility of XAI techniques in this problem domain. We also provide a literature review of XAI work related to Android malware.



## **25. Privacy Protection in Personalized Diffusion Models via Targeted Cross-Attention Adversarial Attack**

cs.CV

Accepted at Safe Generative AI Workshop (NeurIPS 2024)

**SubmitDate**: 2024-11-25    [abs](http://arxiv.org/abs/2411.16437v1) [paper-pdf](http://arxiv.org/pdf/2411.16437v1)

**Authors**: Xide Xu, Muhammad Atif Butt, Sandesh Kamath, Bogdan Raducanu

**Abstract**: The growing demand for customized visual content has led to the rise of personalized text-to-image (T2I) diffusion models. Despite their remarkable potential, they pose significant privacy risk when misused for malicious purposes. In this paper, we propose a novel and efficient adversarial attack method, Concept Protection by Selective Attention Manipulation (CoPSAM) which targets only the cross-attention layers of a T2I diffusion model. For this purpose, we carefully construct an imperceptible noise to be added to clean samples to get their adversarial counterparts. This is obtained during the fine-tuning process by maximizing the discrepancy between the corresponding cross-attention maps of the user-specific token and the class-specific token, respectively. Experimental validation on a subset of CelebA-HQ face images dataset demonstrates that our approach outperforms existing methods. Besides this, our method presents two important advantages derived from the qualitative evaluation: (i) we obtain better protection results for lower noise levels than our competitors; and (ii) we protect the content from unauthorized use thereby protecting the individual's identity from potential misuse.



## **26. Dark Miner: Defend against undesired generation for text-to-image diffusion models**

cs.CV

**SubmitDate**: 2024-11-25    [abs](http://arxiv.org/abs/2409.17682v2) [paper-pdf](http://arxiv.org/pdf/2409.17682v2)

**Authors**: Zheling Meng, Bo Peng, Xiaochuan Jin, Yue Jiang, Jing Dong, Wei Wang

**Abstract**: Text-to-image diffusion models have been demonstrated with undesired generation due to unfiltered large-scale training data, such as sexual images and copyrights, necessitating the erasure of undesired concepts. Most existing methods focus on modifying the generation probabilities conditioned on the texts containing target concepts. However, they fail to guarantee the desired generation of texts unseen in the training phase, especially for the adversarial texts from malicious attacks. In this paper, we analyze the erasure task and point out that existing methods cannot guarantee the minimization of the total probabilities of undesired generation. To tackle this problem, we propose Dark Miner. It entails a recurring three-stage process that comprises mining, verifying, and circumventing. This method greedily mines embeddings with maximum generation probabilities of target concepts and more effectively reduces their generation. In the experiments, we evaluate its performance on the inappropriateness, object, and style concepts. Compared with the previous methods, our method achieves better erasure and defense results, especially under multiple adversarial attacks, while preserving the native generation capability of the models. Our code will be available at https://github.com/RichardSunnyMeng/DarkMiner-offical-codes.



## **27. Scaling Laws for Black box Adversarial Attacks**

cs.LG

**SubmitDate**: 2024-11-25    [abs](http://arxiv.org/abs/2411.16782v1) [paper-pdf](http://arxiv.org/pdf/2411.16782v1)

**Authors**: Chuan Liu, Huanran Chen, Yichi Zhang, Yinpeng Dong, Jun Zhu

**Abstract**: A longstanding problem of deep learning models is their vulnerability to adversarial examples, which are often generated by applying imperceptible perturbations to natural examples. Adversarial examples exhibit cross-model transferability, enabling to attack black-box models with limited information about their architectures and parameters. Model ensembling is an effective strategy to improve the transferability by attacking multiple surrogate models simultaneously. However, as prior studies usually adopt few models in the ensemble, there remains an open question of whether scaling the number of models can further improve black-box attacks. Inspired by the findings in large foundation models, we investigate the scaling laws of black-box adversarial attacks in this work. By analyzing the relationship between the number of surrogate models and transferability of adversarial examples, we conclude with clear scaling laws, emphasizing the potential of using more surrogate models to enhance adversarial transferability. Extensive experiments verify the claims on standard image classifiers, multimodal large language models, and even proprietary models like GPT-4o, demonstrating consistent scaling effects and impressive attack success rates with more surrogate models. Further studies by visualization indicate that scaled attacks bring better interpretability in semantics, indicating that the common features of models are captured.



## **28. Sparse patches adversarial attacks via extrapolating point-wise information**

cs.CV

AdvML-Frontiers 24: The 3nd Workshop on New Frontiers in Adversarial  Machine Learning, NeurIPS 24

**SubmitDate**: 2024-11-25    [abs](http://arxiv.org/abs/2411.16162v1) [paper-pdf](http://arxiv.org/pdf/2411.16162v1)

**Authors**: Yaniv Nemcovsky, Avi Mendelson, Chaim Baskin

**Abstract**: Sparse and patch adversarial attacks were previously shown to be applicable in realistic settings and are considered a security risk to autonomous systems. Sparse adversarial perturbations constitute a setting in which the adversarial perturbations are limited to affecting a relatively small number of points in the input. Patch adversarial attacks denote the setting where the sparse attacks are limited to a given structure, i.e., sparse patches with a given shape and number. However, previous patch adversarial attacks do not simultaneously optimize multiple patches' locations and perturbations. This work suggests a novel approach for sparse patches adversarial attacks via point-wise trimming dense adversarial perturbations. Our approach enables simultaneous optimization of multiple sparse patches' locations and perturbations for any given number and shape. Moreover, our approach is also applicable for standard sparse adversarial attacks, where we show that it significantly improves the state-of-the-art over multiple extensive settings. A reference implementation of the proposed method and the reported experiments is provided at \url{https://github.com/yanemcovsky/SparsePatches.git}



## **29. Unlearn to Relearn Backdoors: Deferred Backdoor Functionality Attacks on Deep Learning Models**

cs.CR

**SubmitDate**: 2024-11-25    [abs](http://arxiv.org/abs/2411.14449v2) [paper-pdf](http://arxiv.org/pdf/2411.14449v2)

**Authors**: Jeongjin Shin, Sangdon Park

**Abstract**: Deep learning models are vulnerable to backdoor attacks, where adversaries inject malicious functionality during training that activates on trigger inputs at inference time. Extensive research has focused on developing stealthy backdoor attacks to evade detection and defense mechanisms. However, these approaches still have limitations that leave the door open for detection and mitigation due to their inherent design to cause malicious behavior in the presence of a trigger. To address this limitation, we introduce Deferred Activated Backdoor Functionality (DABF), a new paradigm in backdoor attacks. Unlike conventional attacks, DABF initially conceals its backdoor, producing benign outputs even when triggered. This stealthy behavior allows DABF to bypass multiple detection and defense methods, remaining undetected during initial inspections. The backdoor functionality is strategically activated only after the model undergoes subsequent updates, such as retraining on benign data. DABF attacks exploit the common practice in the life cycle of machine learning models to perform model updates and fine-tuning after initial deployment. To implement DABF attacks, we approach the problem by making the unlearning of the backdoor fragile, allowing it to be easily cancelled and subsequently reactivate the backdoor functionality. To achieve this, we propose a novel two-stage training scheme, called DeferBad. Our extensive experiments across various fine-tuning scenarios, backdoor attack types, datasets, and model architectures demonstrate the effectiveness and stealthiness of DeferBad.



## **30. Flexible Physical Camouflage Generation Based on a Differential Approach**

cs.CV

**SubmitDate**: 2024-11-25    [abs](http://arxiv.org/abs/2402.13575v2) [paper-pdf](http://arxiv.org/pdf/2402.13575v2)

**Authors**: Yang Li, Wenyi Tan, Chenxing Zhao, Shuangju Zhou, Xinkai Liang, Quan Pan

**Abstract**: This study introduces a novel approach to neural rendering, specifically tailored for adversarial camouflage, within an extensive 3D rendering framework. Our method, named FPA, goes beyond traditional techniques by faithfully simulating lighting conditions and material variations, ensuring a nuanced and realistic representation of textures on a 3D target. To achieve this, we employ a generative approach that learns adversarial patterns from a diffusion model. This involves incorporating a specially designed adversarial loss and covert constraint loss to guarantee the adversarial and covert nature of the camouflage in the physical world. Furthermore, we showcase the effectiveness of the proposed camouflage in sticker mode, demonstrating its ability to cover the target without compromising adversarial information. Through empirical and physical experiments, FPA exhibits strong performance in terms of attack success rate and transferability. Additionally, the designed sticker-mode camouflage, coupled with a concealment constraint, adapts to the environment, yielding diverse styles of texture. Our findings highlight the versatility and efficacy of the FPA approach in adversarial camouflage applications.



## **31. AgentDojo: A Dynamic Environment to Evaluate Prompt Injection Attacks and Defenses for LLM Agents**

cs.CR

Updated version after fixing a bug in the Llama implementation and  updating the travel suite

**SubmitDate**: 2024-11-24    [abs](http://arxiv.org/abs/2406.13352v3) [paper-pdf](http://arxiv.org/pdf/2406.13352v3)

**Authors**: Edoardo Debenedetti, Jie Zhang, Mislav Balunović, Luca Beurer-Kellner, Marc Fischer, Florian Tramèr

**Abstract**: AI agents aim to solve complex tasks by combining text-based reasoning with external tool calls. Unfortunately, AI agents are vulnerable to prompt injection attacks where data returned by external tools hijacks the agent to execute malicious tasks. To measure the adversarial robustness of AI agents, we introduce AgentDojo, an evaluation framework for agents that execute tools over untrusted data. To capture the evolving nature of attacks and defenses, AgentDojo is not a static test suite, but rather an extensible environment for designing and evaluating new agent tasks, defenses, and adaptive attacks. We populate the environment with 97 realistic tasks (e.g., managing an email client, navigating an e-banking website, or making travel bookings), 629 security test cases, and various attack and defense paradigms from the literature. We find that AgentDojo poses a challenge for both attacks and defenses: state-of-the-art LLMs fail at many tasks (even in the absence of attacks), and existing prompt injection attacks break some security properties but not all. We hope that AgentDojo can foster research on new design principles for AI agents that solve common tasks in a reliable and robust manner.. We release the code for AgentDojo at https://github.com/ethz-spylab/agentdojo.



## **32. AmpleGCG: Learning a Universal and Transferable Generative Model of Adversarial Suffixes for Jailbreaking Both Open and Closed LLMs**

cs.CL

Published as a conference paper at COLM 2024  (https://colmweb.org/index.html)

**SubmitDate**: 2024-11-24    [abs](http://arxiv.org/abs/2404.07921v3) [paper-pdf](http://arxiv.org/pdf/2404.07921v3)

**Authors**: Zeyi Liao, Huan Sun

**Abstract**: As large language models (LLMs) become increasingly prevalent and integrated into autonomous systems, ensuring their safety is imperative. Despite significant strides toward safety alignment, recent work GCG~\citep{zou2023universal} proposes a discrete token optimization algorithm and selects the single suffix with the lowest loss to successfully jailbreak aligned LLMs. In this work, we first discuss the drawbacks of solely picking the suffix with the lowest loss during GCG optimization for jailbreaking and uncover the missed successful suffixes during the intermediate steps. Moreover, we utilize those successful suffixes as training data to learn a generative model, named AmpleGCG, which captures the distribution of adversarial suffixes given a harmful query and enables the rapid generation of hundreds of suffixes for any harmful queries in seconds. AmpleGCG achieves near 100\% attack success rate (ASR) on two aligned LLMs (Llama-2-7B-chat and Vicuna-7B), surpassing two strongest attack baselines. More interestingly, AmpleGCG also transfers seamlessly to attack different models, including closed-source LLMs, achieving a 99\% ASR on the latest GPT-3.5. To summarize, our work amplifies the impact of GCG by training a generative model of adversarial suffixes that is universal to any harmful queries and transferable from attacking open-source LLMs to closed-source LLMs. In addition, it can generate 200 adversarial suffixes for one harmful query in only 4 seconds, rendering it more challenging to defend.



## **33. A Framework for Differential Privacy Against Timing Attacks**

cs.CR

**SubmitDate**: 2024-11-24    [abs](http://arxiv.org/abs/2409.05623v2) [paper-pdf](http://arxiv.org/pdf/2409.05623v2)

**Authors**: Zachary Ratliff, Salil Vadhan

**Abstract**: The standard definition of differential privacy (DP) ensures that a mechanism's output distribution on adjacent datasets is indistinguishable. However, real-world implementations of DP can, and often do, reveal information through their runtime distributions, making them susceptible to timing attacks. In this work, we establish a general framework for ensuring differential privacy in the presence of timing side channels. We define a new notion of timing privacy, which captures programs that remain differentially private to an adversary that observes the program's runtime in addition to the output. Our framework enables chaining together component programs that are timing-stable followed by a random delay to obtain DP programs that achieve timing privacy. Importantly, our definitions allow for measuring timing privacy and output privacy using different privacy measures. We illustrate how to instantiate our framework by giving programs for standard DP computations in the RAM and Word RAM models of computation. Furthermore, we show how our framework can be realized in code through a natural extension of the OpenDP Programming Framework.



## **34. A Tunable Despeckling Neural Network Stabilized via Diffusion Equation**

cs.CV

**SubmitDate**: 2024-11-24    [abs](http://arxiv.org/abs/2411.15921v1) [paper-pdf](http://arxiv.org/pdf/2411.15921v1)

**Authors**: Yi Ran, Zhichang Guo, Jia Li, Yao Li, Martin Burger, Boying Wu

**Abstract**: Multiplicative Gamma noise remove is a critical research area in the application of synthetic aperture radar (SAR) imaging, where neural networks serve as a potent tool. However, real-world data often diverges from theoretical models, exhibiting various disturbances, which makes the neural network less effective. Adversarial attacks work by finding perturbations that significantly disrupt functionality of neural networks, as the inherent instability of neural networks makes them highly susceptible. A network designed to withstand such extreme cases can more effectively mitigate general disturbances in real SAR data. In this work, the dissipative nature of diffusion equations is employed to underpin a novel approach for countering adversarial attacks and improve the resistance of real noise disturbance. We propose a tunable, regularized neural network that unrolls a denoising unit and a regularization unit into a single network for end-to-end training. In the network, the denoising unit and the regularization unit are composed of the denoising network and the simplest linear diffusion equation respectively. The regularization unit enhances network stability, allowing post-training time step adjustments to effectively mitigate the adverse impacts of adversarial attacks. The stability and convergence of our model are theoretically proven, and in the experiments, we compare our model with several state-of-the-art denoising methods on simulated images, adversarial samples, and real SAR images, yielding superior results in both quantitative and visual evaluations.



## **35. ExAL: An Exploration Enhanced Adversarial Learning Algorithm**

cs.LG

**SubmitDate**: 2024-11-24    [abs](http://arxiv.org/abs/2411.15878v1) [paper-pdf](http://arxiv.org/pdf/2411.15878v1)

**Authors**: A Vinil, Aneesh Sreevallabh Chivukula, Pranav Chintareddy

**Abstract**: Adversarial learning is critical for enhancing model robustness, aiming to defend against adversarial attacks that jeopardize machine learning systems. Traditional methods often lack efficient mechanisms to explore diverse adversarial perturbations, leading to limited model resilience. Inspired by game-theoretic principles, where adversarial dynamics are analyzed through frameworks like Nash equilibrium, exploration mechanisms in such setups allow for the discovery of diverse strategies, enhancing system robustness. However, existing adversarial learning methods often fail to incorporate structured exploration effectively, reducing their ability to improve model defense comprehensively. To address these challenges, we propose a novel Exploration-enhanced Adversarial Learning Algorithm (ExAL), leveraging the Exponentially Weighted Momentum Particle Swarm Optimizer (EMPSO) to generate optimized adversarial perturbations. ExAL integrates exploration-driven mechanisms to discover perturbations that maximize impact on the model's decision boundary while preserving structural coherence in the data. We evaluate the performance of ExAL on the MNIST Handwritten Digits and Blended Malware datasets. Experimental results demonstrate that ExAL significantly enhances model resilience to adversarial attacks by improving robustness through adversarial learning.



## **36. Data Lineage Inference: Uncovering Privacy Vulnerabilities of Dataset Pruning**

cs.CR

**SubmitDate**: 2024-11-24    [abs](http://arxiv.org/abs/2411.15796v1) [paper-pdf](http://arxiv.org/pdf/2411.15796v1)

**Authors**: Qi Li, Cheng-Long Wang, Yinzhi Cao, Di Wang

**Abstract**: In this work, we systematically explore the data privacy issues of dataset pruning in machine learning systems. Our findings reveal, for the first time, that even if data in the redundant set is solely used before model training, its pruning-phase membership status can still be detected through attacks. Since this is a fully upstream process before model training, traditional model output-based privacy inference methods are completely unsuitable. To address this, we introduce a new task called Data-Centric Membership Inference and propose the first ever data-centric privacy inference paradigm named Data Lineage Inference (DaLI). Under this paradigm, four threshold-based attacks are proposed, named WhoDis, CumDis, ArraDis and SpiDis. We show that even without access to downstream models, adversaries can accurately identify the redundant set with only limited prior knowledge. Furthermore, we find that different pruning methods involve varying levels of privacy leakage, and even the same pruning method can present different privacy risks at different pruning fractions. We conducted an in-depth analysis of these phenomena and introduced a metric called the Brimming score to offer guidance for selecting pruning methods with privacy protection in mind.



## **37. JailBreakV: A Benchmark for Assessing the Robustness of MultiModal Large Language Models against Jailbreak Attacks**

cs.CR

**SubmitDate**: 2024-11-24    [abs](http://arxiv.org/abs/2404.03027v4) [paper-pdf](http://arxiv.org/pdf/2404.03027v4)

**Authors**: Weidi Luo, Siyuan Ma, Xiaogeng Liu, Xiaoyu Guo, Chaowei Xiao

**Abstract**: With the rapid advancements in Multimodal Large Language Models (MLLMs), securing these models against malicious inputs while aligning them with human values has emerged as a critical challenge. In this paper, we investigate an important and unexplored question of whether techniques that successfully jailbreak Large Language Models (LLMs) can be equally effective in jailbreaking MLLMs. To explore this issue, we introduce JailBreakV-28K, a pioneering benchmark designed to assess the transferability of LLM jailbreak techniques to MLLMs, thereby evaluating the robustness of MLLMs against diverse jailbreak attacks. Utilizing a dataset of 2, 000 malicious queries that is also proposed in this paper, we generate 20, 000 text-based jailbreak prompts using advanced jailbreak attacks on LLMs, alongside 8, 000 image-based jailbreak inputs from recent MLLMs jailbreak attacks, our comprehensive dataset includes 28, 000 test cases across a spectrum of adversarial scenarios. Our evaluation of 10 open-source MLLMs reveals a notably high Attack Success Rate (ASR) for attacks transferred from LLMs, highlighting a critical vulnerability in MLLMs that stems from their text-processing capabilities. Our findings underscore the urgent need for future research to address alignment vulnerabilities in MLLMs from both textual and visual inputs.



## **38. Chain of Attack: On the Robustness of Vision-Language Models Against Transfer-Based Adversarial Attacks**

cs.CV

**SubmitDate**: 2024-11-24    [abs](http://arxiv.org/abs/2411.15720v1) [paper-pdf](http://arxiv.org/pdf/2411.15720v1)

**Authors**: Peng Xie, Yequan Bie, Jianda Mao, Yangqiu Song, Yang Wang, Hao Chen, Kani Chen

**Abstract**: Pre-trained vision-language models (VLMs) have showcased remarkable performance in image and natural language understanding, such as image captioning and response generation. As the practical applications of vision-language models become increasingly widespread, their potential safety and robustness issues raise concerns that adversaries may evade the system and cause these models to generate toxic content through malicious attacks. Therefore, evaluating the robustness of open-source VLMs against adversarial attacks has garnered growing attention, with transfer-based attacks as a representative black-box attacking strategy. However, most existing transfer-based attacks neglect the importance of the semantic correlations between vision and text modalities, leading to sub-optimal adversarial example generation and attack performance. To address this issue, we present Chain of Attack (CoA), which iteratively enhances the generation of adversarial examples based on the multi-modal semantic update using a series of intermediate attacking steps, achieving superior adversarial transferability and efficiency. A unified attack success rate computing method is further proposed for automatic evasion evaluation. Extensive experiments conducted under the most realistic and high-stakes scenario, demonstrate that our attacking strategy can effectively mislead models to generate targeted responses using only black-box attacks without any knowledge of the victim models. The comprehensive robustness evaluation in our paper provides insight into the vulnerabilities of VLMs and offers a reference for the safety considerations of future model developments.



## **39. Benchmarking Vision Language Model Unlearning via Fictitious Facial Identity Dataset**

cs.CV

**SubmitDate**: 2024-11-24    [abs](http://arxiv.org/abs/2411.03554v2) [paper-pdf](http://arxiv.org/pdf/2411.03554v2)

**Authors**: Yingzi Ma, Jiongxiao Wang, Fei Wang, Siyuan Ma, Jiazhao Li, Xiujun Li, Furong Huang, Lichao Sun, Bo Li, Yejin Choi, Muhao Chen, Chaowei Xiao

**Abstract**: Machine unlearning has emerged as an effective strategy for forgetting specific information in the training data. However, with the increasing integration of visual data, privacy concerns in Vision Language Models (VLMs) remain underexplored. To address this, we introduce Facial Identity Unlearning Benchmark (FIUBench), a novel VLM unlearning benchmark designed to robustly evaluate the effectiveness of unlearning algorithms under the Right to be Forgotten setting. Specifically, we formulate the VLM unlearning task via constructing the Fictitious Facial Identity VQA dataset and apply a two-stage evaluation pipeline that is designed to precisely control the sources of information and their exposure levels. In terms of evaluation, since VLM supports various forms of ways to ask questions with the same semantic meaning, we also provide robust evaluation metrics including membership inference attacks and carefully designed adversarial privacy attacks to evaluate the performance of algorithms. Through the evaluation of four baseline VLM unlearning algorithms within FIUBench, we find that all methods remain limited in their unlearning performance, with significant trade-offs between model utility and forget quality. Furthermore, our findings also highlight the importance of privacy attacks for robust evaluations. We hope FIUBench will drive progress in developing more effective VLM unlearning algorithms.



## **40. Game-Theoretic Neyman-Pearson Detection to Combat Strategic Evasion**

cs.CR

**SubmitDate**: 2024-11-24    [abs](http://arxiv.org/abs/2206.05276v4) [paper-pdf](http://arxiv.org/pdf/2206.05276v4)

**Authors**: Yinan Hu, Juntao Chen, Quanyan Zhu

**Abstract**: The security in networked systems depends greatly on recognizing and identifying adversarial behaviors. Traditional detection methods focus on specific categories of attacks and have become inadequate for increasingly stealthy and deceptive attacks that are designed to bypass detection strategically. This work aims to develop a holistic theory to countermeasure such evasive attacks. We focus on extending a fundamental class of statistical-based detection methods based on Neyman-Pearson's (NP) hypothesis testing formulation. We propose game-theoretic frameworks to capture the conflicting relationship between a strategic evasive attacker and an evasion-aware NP detector. By analyzing both the equilibrium behaviors of the attacker and the NP detector, we characterize their performance using Equilibrium Receiver-Operational-Characteristic (EROC) curves. We show that the evasion-aware NP detectors outperform the passive ones in the way that the former can act strategically against the attacker's behavior and adaptively modify their decision rules based on the received messages. In addition, we extend our framework to a sequential setting where the user sends out identically distributed messages. We corroborate the analytical results with a case study of anomaly detection.



## **41. Constructing Semantics-Aware Adversarial Examples with a Probabilistic Perspective**

stat.ML

21 pages, 9 figures

**SubmitDate**: 2024-11-24    [abs](http://arxiv.org/abs/2306.00353v3) [paper-pdf](http://arxiv.org/pdf/2306.00353v3)

**Authors**: Andi Zhang, Mingtian Zhang, Damon Wischik

**Abstract**: We propose a probabilistic perspective on adversarial examples, allowing us to embed subjective understanding of semantics as a distribution into the process of generating adversarial examples, in a principled manner. Despite significant pixel-level modifications compared to traditional adversarial attacks, our method preserves the overall semantics of the image, making the changes difficult for humans to detect. This extensive pixel-level modification enhances our method's ability to deceive classifiers designed to defend against adversarial attacks. Our empirical findings indicate that the proposed methods achieve higher success rates in circumventing adversarial defense mechanisms, while remaining difficult for human observers to detect.



## **42. Enhancing the Transferability of Adversarial Attacks on Face Recognition with Diverse Parameters Augmentation**

cs.CV

**SubmitDate**: 2024-11-23    [abs](http://arxiv.org/abs/2411.15555v1) [paper-pdf](http://arxiv.org/pdf/2411.15555v1)

**Authors**: Fengfan Zhou, Bangjie Yin, Hefei Ling, Qianyu Zhou, Wenxuan Wang

**Abstract**: Face Recognition (FR) models are vulnerable to adversarial examples that subtly manipulate benign face images, underscoring the urgent need to improve the transferability of adversarial attacks in order to expose the blind spots of these systems. Existing adversarial attack methods often overlook the potential benefits of augmenting the surrogate model with diverse initializations, which limits the transferability of the generated adversarial examples. To address this gap, we propose a novel method called Diverse Parameters Augmentation (DPA) attack method, which enhances surrogate models by incorporating diverse parameter initializations, resulting in a broader and more diverse set of surrogate models. Specifically, DPA consists of two key stages: Diverse Parameters Optimization (DPO) and Hard Model Aggregation (HMA). In the DPO stage, we initialize the parameters of the surrogate model using both pre-trained and random parameters. Subsequently, we save the models in the intermediate training process to obtain a diverse set of surrogate models. During the HMA stage, we enhance the feature maps of the diversified surrogate models by incorporating beneficial perturbations, thereby further improving the transferability. Experimental results demonstrate that our proposed attack method can effectively enhance the transferability of the crafted adversarial face examples.



## **43. Improving Transferable Targeted Attacks with Feature Tuning Mixup**

cs.CV

**SubmitDate**: 2024-11-23    [abs](http://arxiv.org/abs/2411.15553v1) [paper-pdf](http://arxiv.org/pdf/2411.15553v1)

**Authors**: Kaisheng Liang, Xuelong Dai, Yanjie Li, Dong Wang, Bin Xiao

**Abstract**: Deep neural networks exhibit vulnerability to adversarial examples that can transfer across different models. A particularly challenging problem is developing transferable targeted attacks that can mislead models into predicting specific target classes. While various methods have been proposed to enhance attack transferability, they often incur substantial computational costs while yielding limited improvements. Recent clean feature mixup methods use random clean features to perturb the feature space but lack optimization for disrupting adversarial examples, overlooking the advantages of attack-specific perturbations. In this paper, we propose Feature Tuning Mixup (FTM), a novel method that enhances targeted attack transferability by combining both random and optimized noises in the feature space. FTM introduces learnable feature perturbations and employs an efficient stochastic update strategy for optimization. These learnable perturbations facilitate the generation of more robust adversarial examples with improved transferability. We further demonstrate that attack performance can be enhanced through an ensemble of multiple FTM-perturbed surrogate models. Extensive experiments on the ImageNet-compatible dataset across various models demonstrate that our method achieves significant improvements over state-of-the-art methods while maintaining low computational cost.



## **44. MUNBa: Machine Unlearning via Nash Bargaining**

cs.CV

**SubmitDate**: 2024-11-23    [abs](http://arxiv.org/abs/2411.15537v1) [paper-pdf](http://arxiv.org/pdf/2411.15537v1)

**Authors**: Jing Wu, Mehrtash Harandi

**Abstract**: Machine Unlearning (MU) aims to selectively erase harmful behaviors from models while retaining the overall utility of the model. As a multi-task learning problem, MU involves balancing objectives related to forgetting specific concepts/data and preserving general performance. A naive integration of these forgetting and preserving objectives can lead to gradient conflicts, impeding MU algorithms from reaching optimal solutions. To address the gradient conflict issue, we reformulate MU as a two-player cooperative game, where the two players, namely, the forgetting player and the preservation player, contribute via their gradient proposals to maximize their overall gain. To this end, inspired by the Nash bargaining theory, we derive a closed-form solution to guide the model toward the Pareto front, effectively avoiding the gradient conflicts. Our formulation of MU guarantees an equilibrium solution, where any deviation from the final state would lead to a reduction in the overall objectives for both players, ensuring optimality in each objective. We evaluate our algorithm's effectiveness on a diverse set of tasks across image classification and image generation. Extensive experiments with ResNet, vision-language model CLIP, and text-to-image diffusion models demonstrate that our method outperforms state-of-the-art MU algorithms, achieving superior performance on several benchmarks. For example, in the challenging scenario of sample-wise forgetting, our algorithm approaches the gold standard retrain baseline. Our results also highlight improvements in forgetting precision, preservation of generalization, and robustness against adversarial attacks.



## **45. Harnessing LLM to Attack LLM-Guarded Text-to-Image Models**

cs.AI

10 pages, 7 figures, under review

**SubmitDate**: 2024-11-23    [abs](http://arxiv.org/abs/2312.07130v4) [paper-pdf](http://arxiv.org/pdf/2312.07130v4)

**Authors**: Yimo Deng, Huangxun Chen

**Abstract**: To prevent Text-to-Image (T2I) models from generating unethical images, people deploy safety filters to block inappropriate drawing prompts. Previous works have employed token replacement to search adversarial prompts that attempt to bypass these filters, but they have become ineffective as nonsensical tokens fail semantic logic checks. In this paper, we approach adversarial prompts from a different perspective. We demonstrate that rephrasing a drawing intent into multiple benign descriptions of individual visual components can obtain an effective adversarial prompt. We propose a LLM-piloted multi-agent method named DACA to automatically complete intended rephrasing. Our method successfully bypasses the safety filters of DALL-E 3 and Midjourney to generate the intended images, achieving success rates of up to 76.7% and 64% in the one-time attack, and 98% and 84% in the re-use attack, respectively. We open-source our code and dataset on [this link](https://github.com/researchcode003/DACA).



## **46. Steering Away from Harm: An Adaptive Approach to Defending Vision Language Model Against Jailbreaks**

cs.CV

**SubmitDate**: 2024-11-23    [abs](http://arxiv.org/abs/2411.16721v1) [paper-pdf](http://arxiv.org/pdf/2411.16721v1)

**Authors**: Han Wang, Gang Wang, Huan Zhang

**Abstract**: Vision Language Models (VLMs) can produce unintended and harmful content when exposed to adversarial attacks, particularly because their vision capabilities create new vulnerabilities. Existing defenses, such as input preprocessing, adversarial training, and response evaluation-based methods, are often impractical for real-world deployment due to their high costs. To address this challenge, we propose ASTRA, an efficient and effective defense by adaptively steering models away from adversarial feature directions to resist VLM attacks. Our key procedures involve finding transferable steering vectors representing the direction of harmful response and applying adaptive activation steering to remove these directions at inference time. To create effective steering vectors, we randomly ablate the visual tokens from the adversarial images and identify those most strongly associated with jailbreaks. These tokens are then used to construct steering vectors. During inference, we perform the adaptive steering method that involves the projection between the steering vectors and calibrated activation, resulting in little performance drops on benign inputs while strongly avoiding harmful outputs under adversarial inputs. Extensive experiments across multiple models and baselines demonstrate our state-of-the-art performance and high efficiency in mitigating jailbreak risks. Additionally, ASTRA exhibits good transferability, defending against both unseen attacks at design time (i.e., structured-based attacks) and adversarial images from diverse distributions.



## **47. Scalable and Optimal Security Allocation in Networks against Stealthy Injection Attacks**

eess.SY

8 pages, 5 figures, journal submission

**SubmitDate**: 2024-11-22    [abs](http://arxiv.org/abs/2411.15319v1) [paper-pdf](http://arxiv.org/pdf/2411.15319v1)

**Authors**: Anh Tung Nguyen, Sribalaji C. Anand, André M. H. Teixeira

**Abstract**: This paper addresses the security allocation problem in a networked control system under stealthy injection attacks. The networked system is comprised of interconnected subsystems which are represented by nodes in a digraph. An adversary compromises the system by injecting false data into several nodes with the aim of maximally disrupting the performance of the network while remaining stealthy to a defender. To minimize the impact of such stealthy attacks, the defender, with limited knowledge about attack policies and attack resources, allocates several sensors on nodes to impose the stealthiness constraint governing the attack policy. We provide an optimal security allocation algorithm to minimize the expected attack impact on the entire network. Furthermore, under a suitable local control design, the proposed security allocation algorithm can be executed in a scalable way. Finally, the obtained results are validated through several numerical examples.



## **48. UnMarker: A Universal Attack on Defensive Image Watermarking**

cs.CR

To appear at IEEE S&P 2025

**SubmitDate**: 2024-11-22    [abs](http://arxiv.org/abs/2405.08363v2) [paper-pdf](http://arxiv.org/pdf/2405.08363v2)

**Authors**: Andre Kassis, Urs Hengartner

**Abstract**: Reports regarding the misuse of Generative AI (GenAI) to create deepfakes are frequent. Defensive watermarking enables GenAI providers to hide fingerprints in their images and use them later for deepfake detection. Yet, its potential has not been fully explored. We present UnMarker -- the first practical universal attack on defensive watermarking. Unlike existing attacks, UnMarker requires no detector feedback, no unrealistic knowledge of the watermarking scheme or similar models, and no advanced denoising pipelines that may not be available. Instead, being the product of an in-depth analysis of the watermarking paradigm revealing that robust schemes must construct their watermarks in the spectral amplitudes, UnMarker employs two novel adversarial optimizations to disrupt the spectra of watermarked images, erasing the watermarks. Evaluations against SOTA schemes prove UnMarker's effectiveness. It not only defeats traditional schemes while retaining superior quality compared to existing attacks but also breaks semantic watermarks that alter an image's structure, reducing the best detection rate to $43\%$ and rendering them useless. To our knowledge, UnMarker is the first practical attack on semantic watermarks, which have been deemed the future of defensive watermarking. Our findings show that defensive watermarking is not a viable defense against deepfakes, and we urge the community to explore alternatives.



## **49. Benchmarking the Robustness of Optical Flow Estimation to Corruptions**

eess.IV

The benchmarks and source code will be released at  https://github.com/ZhonghuaYi/optical_flow_robustness_benchmark

**SubmitDate**: 2024-11-22    [abs](http://arxiv.org/abs/2411.14865v1) [paper-pdf](http://arxiv.org/pdf/2411.14865v1)

**Authors**: Zhonghua Yi, Hao Shi, Qi Jiang, Yao Gao, Ze Wang, Yufan Zhang, Kailun Yang, Kaiwei Wang

**Abstract**: Optical flow estimation is extensively used in autonomous driving and video editing. While existing models demonstrate state-of-the-art performance across various benchmarks, the robustness of these methods has been infrequently investigated. Despite some research focusing on the robustness of optical flow models against adversarial attacks, there has been a lack of studies investigating their robustness to common corruptions. Taking into account the unique temporal characteristics of optical flow, we introduce 7 temporal corruptions specifically designed for benchmarking the robustness of optical flow models, in addition to 17 classical single-image corruptions, in which advanced PSF Blur simulation method is performed. Two robustness benchmarks, KITTI-FC and GoPro-FC, are subsequently established as the first corruption robustness benchmark for optical flow estimation, with Out-Of-Domain (OOD) and In-Domain (ID) settings to facilitate comprehensive studies. Robustness metrics, Corruption Robustness Error (CRE), Corruption Robustness Error ratio (CREr), and Relative Corruption Robustness Error (RCRE) are further introduced to quantify the optical flow estimation robustness. 29 model variants from 15 optical flow methods are evaluated, yielding 10 intriguing observations, such as 1) the absolute robustness of the model is heavily dependent on the estimation performance; 2) the corruptions that diminish local information are more serious than that reduce visual effects. We also give suggestions for the design and application of optical flow models. We anticipate that our benchmark will serve as a foundational resource for advancing research in robust optical flow estimation. The benchmarks and source code will be released at https://github.com/ZhonghuaYi/optical_flow_robustness_benchmark.



## **50. Derivative-Free Diffusion Manifold-Constrained Gradient for Unified XAI**

cs.CV

19 pages, 5 figures

**SubmitDate**: 2024-11-22    [abs](http://arxiv.org/abs/2411.15265v1) [paper-pdf](http://arxiv.org/pdf/2411.15265v1)

**Authors**: Won Jun Kim, Hyungjin Chung, Jaemin Kim, Sangmin Lee, Byeongsu Sim, Jong Chul Ye

**Abstract**: Gradient-based methods are a prototypical family of explainability techniques, especially for image-based models. Nonetheless, they have several shortcomings in that they (1) require white-box access to models, (2) are vulnerable to adversarial attacks, and (3) produce attributions that lie off the image manifold, leading to explanations that are not actually faithful to the model and do not align well with human perception. To overcome these challenges, we introduce Derivative-Free Diffusion Manifold-Constrainted Gradients (FreeMCG), a novel method that serves as an improved basis for explainability of a given neural network than the traditional gradient. Specifically, by leveraging ensemble Kalman filters and diffusion models, we derive a derivative-free approximation of the model's gradient projected onto the data manifold, requiring access only to the model's outputs. We demonstrate the effectiveness of FreeMCG by applying it to both counterfactual generation and feature attribution, which have traditionally been treated as distinct tasks. Through comprehensive evaluation on both tasks, counterfactual explanation and feature attribution, we show that our method yields state-of-the-art results while preserving the essential properties expected of XAI tools.



