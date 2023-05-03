# Latest Adversarial Attack Papers
**update at 2023-05-03 17:36:49**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. Sentiment Perception Adversarial Attacks on Neural Machine Translation Systems**

cs.CL

**SubmitDate**: 2023-05-02    [abs](http://arxiv.org/abs/2305.01437v1) [paper-pdf](http://arxiv.org/pdf/2305.01437v1)

**Authors**: Vyas Raina, Mark Gales

**Abstract**: With the advent of deep learning methods, Neural Machine Translation (NMT) systems have become increasingly powerful. However, deep learning based systems are susceptible to adversarial attacks, where imperceptible changes to the input can cause undesirable changes at the output of the system. To date there has been little work investigating adversarial attacks on sequence-to-sequence systems, such as NMT models. Previous work in NMT has examined attacks with the aim of introducing target phrases in the output sequence. In this work, adversarial attacks for NMT systems are explored from an output perception perspective. Thus the aim of an attack is to change the perception of the output sequence, without altering the perception of the input sequence. For example, an adversary may distort the sentiment of translated reviews to have an exaggerated positive sentiment. In practice it is challenging to run extensive human perception experiments, so a proxy deep-learning classifier applied to the NMT output is used to measure perception changes. Experiments demonstrate that the sentiment perception of NMT systems' output sequences can be changed significantly.



## **2. Boosting Adversarial Transferability via Fusing Logits of Top-1 Decomposed Feature**

cs.CV

**SubmitDate**: 2023-05-02    [abs](http://arxiv.org/abs/2305.01361v1) [paper-pdf](http://arxiv.org/pdf/2305.01361v1)

**Authors**: Juanjuan Weng, Zhiming Luo, Dazhen Lin, Shaozi Li, Zhun Zhong

**Abstract**: Recent research has shown that Deep Neural Networks (DNNs) are highly vulnerable to adversarial samples, which are highly transferable and can be used to attack other unknown black-box models. To improve the transferability of adversarial samples, several feature-based adversarial attack methods have been proposed to disrupt neuron activation in middle layers. However, current state-of-the-art feature-based attack methods typically require additional computation costs for estimating the importance of neurons. To address this challenge, we propose a Singular Value Decomposition (SVD)-based feature-level attack method. Our approach is inspired by the discovery that eigenvectors associated with the larger singular values decomposed from the middle layer features exhibit superior generalization and attention properties. Specifically, we conduct the attack by retaining the decomposed Top-1 singular value-associated feature for computing the output logits, which are then combined with the original logits to optimize adversarial perturbations. Our extensive experimental results verify the effectiveness of our proposed method, which significantly enhances the transferability of adversarial samples against various baseline models and defense strategies.The source code of this study is available at \href{https://anonymous.4open.science/r/SVD-SSA-13BF/README.md}.



## **3. Improving adversarial robustness by putting more regularizations on less robust samples**

stat.ML

**SubmitDate**: 2023-05-02    [abs](http://arxiv.org/abs/2206.03353v3) [paper-pdf](http://arxiv.org/pdf/2206.03353v3)

**Authors**: Dongyoon Yang, Insung Kong, Yongdai Kim

**Abstract**: Adversarial training, which is to enhance robustness against adversarial attacks, has received much attention because it is easy to generate human-imperceptible perturbations of data to deceive a given deep neural network. In this paper, we propose a new adversarial training algorithm that is theoretically well motivated and empirically superior to other existing algorithms. A novel feature of the proposed algorithm is to apply more regularization to data vulnerable to adversarial attacks than other existing regularization algorithms do. Theoretically, we show that our algorithm can be understood as an algorithm of minimizing the regularized empirical risk motivated from a newly derived upper bound of the robust risk. Numerical experiments illustrate that our proposed algorithm improves the generalization (accuracy on examples) and robustness (accuracy on adversarial attacks) simultaneously to achieve the state-of-the-art performance.



## **4. StyleFool: Fooling Video Classification Systems via Style Transfer**

cs.CV

18 pages, 9 figures. Accepted to S&P 2023

**SubmitDate**: 2023-05-02    [abs](http://arxiv.org/abs/2203.16000v3) [paper-pdf](http://arxiv.org/pdf/2203.16000v3)

**Authors**: Yuxin Cao, Xi Xiao, Ruoxi Sun, Derui Wang, Minhui Xue, Sheng Wen

**Abstract**: Video classification systems are vulnerable to adversarial attacks, which can create severe security problems in video verification. Current black-box attacks need a large number of queries to succeed, resulting in high computational overhead in the process of attack. On the other hand, attacks with restricted perturbations are ineffective against defenses such as denoising or adversarial training. In this paper, we focus on unrestricted perturbations and propose StyleFool, a black-box video adversarial attack via style transfer to fool the video classification system. StyleFool first utilizes color theme proximity to select the best style image, which helps avoid unnatural details in the stylized videos. Meanwhile, the target class confidence is additionally considered in targeted attacks to influence the output distribution of the classifier by moving the stylized video closer to or even across the decision boundary. A gradient-free method is then employed to further optimize the adversarial perturbations. We carry out extensive experiments to evaluate StyleFool on two standard datasets, UCF-101 and HMDB-51. The experimental results demonstrate that StyleFool outperforms the state-of-the-art adversarial attacks in terms of both the number of queries and the robustness against existing defenses. Moreover, 50% of the stylized videos in untargeted attacks do not need any query since they can already fool the video classification model. Furthermore, we evaluate the indistinguishability through a user study to show that the adversarial samples of StyleFool look imperceptible to human eyes, despite unrestricted perturbations.



## **5. Exposing Fine-Grained Adversarial Vulnerability of Face Anti-Spoofing Models**

cs.CV

Accepted by IEEE/CVF Conference on Computer Vision and Pattern  Recognition (CVPR) Workshop, 2023

**SubmitDate**: 2023-05-02    [abs](http://arxiv.org/abs/2205.14851v3) [paper-pdf](http://arxiv.org/pdf/2205.14851v3)

**Authors**: Songlin Yang, Wei Wang, Chenye Xu, Ziwen He, Bo Peng, Jing Dong

**Abstract**: Face anti-spoofing aims to discriminate the spoofing face images (e.g., printed photos) from live ones. However, adversarial examples greatly challenge its credibility, where adding some perturbation noise can easily change the predictions. Previous works conducted adversarial attack methods to evaluate the face anti-spoofing performance without any fine-grained analysis that which model architecture or auxiliary feature is vulnerable to the adversary. To handle this problem, we propose a novel framework to expose the fine-grained adversarial vulnerability of the face anti-spoofing models, which consists of a multitask module and a semantic feature augmentation (SFA) module. The multitask module can obtain different semantic features for further evaluation, but only attacking these semantic features fails to reflect the discrimination-related vulnerability. We then design the SFA module to introduce the data distribution prior for more discrimination-related gradient directions for generating adversarial examples. Comprehensive experiments show that SFA module increases the attack success rate by nearly 40$\%$ on average. We conduct this fine-grained adversarial analysis on different annotations, geometric maps, and backbone networks (e.g., Resnet network). These fine-grained adversarial examples can be used for selecting robust backbone networks and auxiliary features. They also can be used for adversarial training, which makes it practical to further improve the accuracy and robustness of the face anti-spoofing models.



## **6. Stratified Adversarial Robustness with Rejection**

cs.LG

Paper published at International Conference on Machine Learning  (ICML'23)

**SubmitDate**: 2023-05-02    [abs](http://arxiv.org/abs/2305.01139v1) [paper-pdf](http://arxiv.org/pdf/2305.01139v1)

**Authors**: Jiefeng Chen, Jayaram Raghuram, Jihye Choi, Xi Wu, Yingyu Liang, Somesh Jha

**Abstract**: Recently, there is an emerging interest in adversarially training a classifier with a rejection option (also known as a selective classifier) for boosting adversarial robustness. While rejection can incur a cost in many applications, existing studies typically associate zero cost with rejecting perturbed inputs, which can result in the rejection of numerous slightly-perturbed inputs that could be correctly classified. In this work, we study adversarially-robust classification with rejection in the stratified rejection setting, where the rejection cost is modeled by rejection loss functions monotonically non-increasing in the perturbation magnitude. We theoretically analyze the stratified rejection setting and propose a novel defense method -- Adversarial Training with Consistent Prediction-based Rejection (CPR) -- for building a robust selective classifier. Experiments on image datasets demonstrate that the proposed method significantly outperforms existing methods under strong adaptive attacks. For instance, on CIFAR-10, CPR reduces the total robust loss (for different rejection losses) by at least 7.3% under both seen and unseen attacks.



## **7. Randomized Reversible Gate-Based Obfuscation for Secured Compilation of Quantum Circuit**

quant-ph

11 pages, 12 figures, conference

**SubmitDate**: 2023-05-02    [abs](http://arxiv.org/abs/2305.01133v1) [paper-pdf](http://arxiv.org/pdf/2305.01133v1)

**Authors**: Subrata Das, Swaroop Ghosh

**Abstract**: The success of quantum circuits in providing reliable outcomes for a given problem depends on the gate count and depth in near-term noisy quantum computers. Quantum circuit compilers that decompose high-level gates to native gates of the hardware and optimize the circuit play a key role in quantum computing. However, the quality and time complexity of the optimization process can vary significantly especially for practically relevant large-scale quantum circuits. As a result, third-party (often less-trusted/untrusted) compilers have emerged, claiming to provide better and faster optimization of complex quantum circuits than so-called trusted compilers. However, untrusted compilers can pose severe security risks, such as the theft of sensitive intellectual property (IP) embedded within the quantum circuit. We propose an obfuscation technique for quantum circuits using randomized reversible gates to protect them from such attacks during compilation. The idea is to insert a small random circuit into the original circuit and send it to the untrusted compiler. Since the circuit function is corrupted, the adversary may get incorrect IP. However, the user may also get incorrect output post-compilation. To circumvent this issue, we concatenate the inverse of the random circuit in the compiled circuit to recover the original functionality. We demonstrate the practicality of our method by conducting exhaustive experiments on a set of benchmark circuits and measuring the quality of obfuscation by calculating the Total Variation Distance (TVD) metric. Our method achieves TVD of up to 1.92 and performs at least 2X better than a previously reported obfuscation method. We also propose a novel adversarial reverse engineering (RE) approach and show that the proposed obfuscation is resilient against RE attacks. The proposed technique introduces minimal degradation in fidelity (~1% to ~3% on average).



## **8. Evaluating Adversarial Robustness on Document Image Classification**

cs.CV

The 17th International Conference on Document Analysis and  Recognition

**SubmitDate**: 2023-05-01    [abs](http://arxiv.org/abs/2304.12486v2) [paper-pdf](http://arxiv.org/pdf/2304.12486v2)

**Authors**: Timothée Fronteau, Arnaud Paran, Aymen Shabou

**Abstract**: Adversarial attacks and defenses have gained increasing interest on computer vision systems in recent years, but as of today, most investigations are limited to images. However, many artificial intelligence models actually handle documentary data, which is very different from real world images. Hence, in this work, we try to apply the adversarial attack philosophy on documentary and natural data and to protect models against such attacks. We focus our work on untargeted gradient-based, transfer-based and score-based attacks and evaluate the impact of adversarial training, JPEG input compression and grey-scale input transformation on the robustness of ResNet50 and EfficientNetB0 model architectures. To the best of our knowledge, no such work has been conducted by the community in order to study the impact of these attacks on the document image classification task.



## **9. Physical Adversarial Attacks for Surveillance: A Survey**

cs.CV

**SubmitDate**: 2023-05-01    [abs](http://arxiv.org/abs/2305.01074v1) [paper-pdf](http://arxiv.org/pdf/2305.01074v1)

**Authors**: Kien Nguyen, Tharindu Fernando, Clinton Fookes, Sridha Sridharan

**Abstract**: Modern automated surveillance techniques are heavily reliant on deep learning methods. Despite the superior performance, these learning systems are inherently vulnerable to adversarial attacks - maliciously crafted inputs that are designed to mislead, or trick, models into making incorrect predictions. An adversary can physically change their appearance by wearing adversarial t-shirts, glasses, or hats or by specific behavior, to potentially avoid various forms of detection, tracking and recognition of surveillance systems; and obtain unauthorized access to secure properties and assets. This poses a severe threat to the security and safety of modern surveillance systems. This paper reviews recent attempts and findings in learning and designing physical adversarial attacks for surveillance applications. In particular, we propose a framework to analyze physical adversarial attacks and provide a comprehensive survey of physical adversarial attacks on four key surveillance tasks: detection, identification, tracking, and action recognition under this framework. Furthermore, we review and analyze strategies to defend against the physical adversarial attacks and the methods for evaluating the strengths of the defense. The insights in this paper present an important step in building resilience within surveillance systems to physical adversarial attacks.



## **10. IoTFlowGenerator: Crafting Synthetic IoT Device Traffic Flows for Cyber Deception**

cs.CR

FLAIRS-36

**SubmitDate**: 2023-05-01    [abs](http://arxiv.org/abs/2305.00925v1) [paper-pdf](http://arxiv.org/pdf/2305.00925v1)

**Authors**: Joseph Bao, Murat Kantarcioglu, Yevgeniy Vorobeychik, Charles Kamhoua

**Abstract**: Over the years, honeypots emerged as an important security tool to understand attacker intent and deceive attackers to spend time and resources. Recently, honeypots are being deployed for Internet of things (IoT) devices to lure attackers, and learn their behavior. However, most of the existing IoT honeypots, even the high interaction ones, are easily detected by an attacker who can observe honeypot traffic due to lack of real network traffic originating from the honeypot. This implies that, to build better honeypots and enhance cyber deception capabilities, IoT honeypots need to generate realistic network traffic flows. To achieve this goal, we propose a novel deep learning based approach for generating traffic flows that mimic real network traffic due to user and IoT device interactions. A key technical challenge that our approach overcomes is scarcity of device-specific IoT traffic data to effectively train a generator. We address this challenge by leveraging a core generative adversarial learning algorithm for sequences along with domain specific knowledge common to IoT devices. Through an extensive experimental evaluation with 18 IoT devices, we demonstrate that the proposed synthetic IoT traffic generation tool significantly outperforms state of the art sequence and packet generators in remaining indistinguishable from real traffic even to an adaptive attacker.



## **11. Attack-SAM: Towards Evaluating Adversarial Robustness of Segment Anything Model**

cs.CV

The first work to evaluate the adversarial robustness of Segment  Anything Model (ongoing)

**SubmitDate**: 2023-05-01    [abs](http://arxiv.org/abs/2305.00866v1) [paper-pdf](http://arxiv.org/pdf/2305.00866v1)

**Authors**: Chenshuang Zhang, Chaoning Zhang, Taegoo Kang, Donghun Kim, Sung-Ho Bae, In So Kweon

**Abstract**: Segment Anything Model (SAM) has attracted significant attention recently, due to its impressive performance on various downstream tasks in a zero-short manner. Computer vision (CV) area might follow the natural language processing (NLP) area to embark on a path from task-specific vision models toward foundation models. However, previous task-specific models are widely recognized as vulnerable to adversarial examples, which fool the model to make wrong predictions with imperceptible perturbation. Such vulnerability to adversarial attacks causes serious concerns when applying deep models to security-sensitive applications. Therefore, it is critical to know whether the vision foundation model SAM can also be easily fooled by adversarial attacks. To the best of our knowledge, our work is the first of its kind to conduct a comprehensive investigation on how to attack SAM with adversarial examples. Specifically, we find that SAM is vulnerable to white-box attacks while maintaining robustness to some extent in the black-box setting. This is an ongoing project and more results and findings will be updated soon through https://github.com/chenshuang-zhang/attack-sam.



## **12. Visual Prompting for Adversarial Robustness**

cs.CV

ICASSP 2023

**SubmitDate**: 2023-05-01    [abs](http://arxiv.org/abs/2210.06284v4) [paper-pdf](http://arxiv.org/pdf/2210.06284v4)

**Authors**: Aochuan Chen, Peter Lorenz, Yuguang Yao, Pin-Yu Chen, Sijia Liu

**Abstract**: In this work, we leverage visual prompting (VP) to improve adversarial robustness of a fixed, pre-trained model at testing time. Compared to conventional adversarial defenses, VP allows us to design universal (i.e., data-agnostic) input prompting templates, which have plug-and-play capabilities at testing time to achieve desired model performance without introducing much computation overhead. Although VP has been successfully applied to improving model generalization, it remains elusive whether and how it can be used to defend against adversarial attacks. We investigate this problem and show that the vanilla VP approach is not effective in adversarial defense since a universal input prompt lacks the capacity for robust learning against sample-specific adversarial perturbations. To circumvent it, we propose a new VP method, termed Class-wise Adversarial Visual Prompting (C-AVP), to generate class-wise visual prompts so as to not only leverage the strengths of ensemble prompts but also optimize their interrelations to improve model robustness. Our experiments show that C-AVP outperforms the conventional VP method, with 2.1X standard accuracy gain and 2X robust accuracy gain. Compared to classical test-time defenses, C-AVP also yields a 42X inference time speedup.



## **13. Robustness of Graph Neural Networks at Scale**

cs.LG

39 pages, 22 figures, 17 tables NeurIPS 2021

**SubmitDate**: 2023-04-30    [abs](http://arxiv.org/abs/2110.14038v4) [paper-pdf](http://arxiv.org/pdf/2110.14038v4)

**Authors**: Simon Geisler, Tobias Schmidt, Hakan Şirin, Daniel Zügner, Aleksandar Bojchevski, Stephan Günnemann

**Abstract**: Graph Neural Networks (GNNs) are increasingly important given their popularity and the diversity of applications. Yet, existing studies of their vulnerability to adversarial attacks rely on relatively small graphs. We address this gap and study how to attack and defend GNNs at scale. We propose two sparsity-aware first-order optimization attacks that maintain an efficient representation despite optimizing over a number of parameters which is quadratic in the number of nodes. We show that common surrogate losses are not well-suited for global attacks on GNNs. Our alternatives can double the attack strength. Moreover, to improve GNNs' reliability we design a robust aggregation function, Soft Median, resulting in an effective defense at all scales. We evaluate our attacks and defense with standard GNNs on graphs more than 100 times larger compared to previous work. We even scale one order of magnitude further by extending our techniques to a scalable GNN.



## **14. Assessing Vulnerabilities of Adversarial Learning Algorithm through Poisoning Attacks**

cs.CR

**SubmitDate**: 2023-04-30    [abs](http://arxiv.org/abs/2305.00399v1) [paper-pdf](http://arxiv.org/pdf/2305.00399v1)

**Authors**: Jingfeng Zhang, Bo Song, Bo Han, Lei Liu, Gang Niu, Masashi Sugiyama

**Abstract**: Adversarial training (AT) is a robust learning algorithm that can defend against adversarial attacks in the inference phase and mitigate the side effects of corrupted data in the training phase. As such, it has become an indispensable component of many artificial intelligence (AI) systems. However, in high-stake AI applications, it is crucial to understand AT's vulnerabilities to ensure reliable deployment. In this paper, we investigate AT's susceptibility to poisoning attacks, a type of malicious attack that manipulates training data to compromise the performance of the trained model. Previous work has focused on poisoning attacks against standard training, but little research has been done on their effectiveness against AT. To fill this gap, we design and test effective poisoning attacks against AT. Specifically, we investigate and design clean-label poisoning attacks, allowing attackers to imperceptibly modify a small fraction of training data to control the algorithm's behavior on a specific target data point. Additionally, we propose the clean-label untargeted attack, enabling attackers can attach tiny stickers on training data to degrade the algorithm's performance on all test data, where the stickers could serve as a signal against unauthorized data collection. Our experiments demonstrate that AT can still be poisoned, highlighting the need for caution when using vanilla AT algorithms in security-related applications. The code is at https://github.com/zjfheart/Poison-adv-training.git.



## **15. Enhancing Adversarial Contrastive Learning via Adversarial Invariant Regularization**

cs.LG

**SubmitDate**: 2023-04-30    [abs](http://arxiv.org/abs/2305.00374v1) [paper-pdf](http://arxiv.org/pdf/2305.00374v1)

**Authors**: Xilie Xu, Jingfeng Zhang, Feng Liu, Masashi Sugiyama, Mohan Kankanhalli

**Abstract**: Adversarial contrastive learning (ACL), without requiring labels, incorporates adversarial data with standard contrastive learning (SCL) and outputs a robust representation which is generalizable and resistant to adversarial attacks and common corruptions. The style-independence property of representations has been validated to be beneficial in improving robustness transferability. Standard invariant regularization (SIR) has been proposed to make the learned representations via SCL to be independent of the style factors. However, how to equip robust representations learned via ACL with the style-independence property is still unclear so far. To this end, we leverage the technique of causal reasoning to propose an adversarial invariant regularization (AIR) that enforces robust representations learned via ACL to be style-independent. Then, we enhance ACL using invariant regularization (IR), which is a weighted sum of SIR and AIR. Theoretically, we show that AIR implicitly encourages the prediction of adversarial data and consistency between adversarial and natural data to be independent of data augmentations. We also theoretically demonstrate that the style-independence property of robust representation learned via ACL still holds in downstream tasks, providing generalization guarantees. Empirically, our comprehensive experimental results corroborate that IR can significantly improve the performance of ACL and its variants on various datasets.



## **16. MetaShard: A Novel Sharding Blockchain Platform for Metaverse Applications**

cs.CR

**SubmitDate**: 2023-04-30    [abs](http://arxiv.org/abs/2305.00367v1) [paper-pdf](http://arxiv.org/pdf/2305.00367v1)

**Authors**: Cong T. Nguyen, Dinh Thai Hoang, Diep N. Nguyen, Yong Xiao, Dusit Niyato, Eryk Dutkiewicz

**Abstract**: Due to its security, transparency, and flexibility in verifying virtual assets, blockchain has been identified as one of the key technologies for Metaverse. Unfortunately, blockchain-based Metaverse faces serious challenges such as massive resource demands, scalability, and security concerns. To address these issues, this paper proposes a novel sharding-based blockchain framework, namely MetaShard, for Metaverse applications. Particularly, we first develop an effective consensus mechanism, namely Proof-of-Engagement, that can incentivize MUs' data and computing resource contribution. Moreover, to improve the scalability of MetaShard, we propose an innovative sharding management scheme to maximize the network's throughput while protecting the shards from 51% attacks. Since the optimization problem is NP-complete, we develop a hybrid approach that decomposes the problem (using the binary search method) into sub-problems that can be solved effectively by the Lagrangian method. As a result, the proposed approach can obtain solutions in polynomial time, thereby enabling flexible shard reconfiguration and reducing the risk of corruption from the adversary. Extensive numerical experiments show that, compared to the state-of-the-art commercial solvers, our proposed approach can achieve up to 66.6% higher throughput in less than 1/30 running time. Moreover, the proposed approach can achieve global optimal solutions in most experiments.



## **17. FedGrad: Mitigating Backdoor Attacks in Federated Learning Through Local Ultimate Gradients Inspection**

cs.CV

Accepted for presentation at the International Joint Conference on  Neural Networks (IJCNN 2023)

**SubmitDate**: 2023-04-29    [abs](http://arxiv.org/abs/2305.00328v1) [paper-pdf](http://arxiv.org/pdf/2305.00328v1)

**Authors**: Thuy Dung Nguyen, Anh Duy Nguyen, Kok-Seng Wong, Huy Hieu Pham, Thanh Hung Nguyen, Phi Le Nguyen, Truong Thao Nguyen

**Abstract**: Federated learning (FL) enables multiple clients to train a model without compromising sensitive data. The decentralized nature of FL makes it susceptible to adversarial attacks, especially backdoor insertion during training. Recently, the edge-case backdoor attack employing the tail of the data distribution has been proposed as a powerful one, raising questions about the shortfall in current defenses' robustness guarantees. Specifically, most existing defenses cannot eliminate edge-case backdoor attacks or suffer from a trade-off between backdoor-defending effectiveness and overall performance on the primary task. To tackle this challenge, we propose FedGrad, a novel backdoor-resistant defense for FL that is resistant to cutting-edge backdoor attacks, including the edge-case attack, and performs effectively under heterogeneous client data and a large number of compromised clients. FedGrad is designed as a two-layer filtering mechanism that thoroughly analyzes the ultimate layer's gradient to identify suspicious local updates and remove them from the aggregation process. We evaluate FedGrad under different attack scenarios and show that it significantly outperforms state-of-the-art defense mechanisms. Notably, FedGrad can almost 100% correctly detect the malicious participants, thus providing a significant reduction in the backdoor effect (e.g., backdoor accuracy is less than 8%) while not reducing the main accuracy on the primary task.



## **18. Game Theoretic Mixed Experts for Combinational Adversarial Machine Learning**

cs.LG

17pages, 10 figures

**SubmitDate**: 2023-04-29    [abs](http://arxiv.org/abs/2211.14669v2) [paper-pdf](http://arxiv.org/pdf/2211.14669v2)

**Authors**: Ethan Rathbun, Kaleel Mahmood, Sohaib Ahmad, Caiwen Ding, Marten van Dijk

**Abstract**: Recent advances in adversarial machine learning have shown that defenses considered to be robust are actually susceptible to adversarial attacks which are specifically customized to target their weaknesses. These defenses include Barrage of Random Transforms (BaRT), Friendly Adversarial Training (FAT), Trash is Treasure (TiT) and ensemble models made up of Vision Transformers (ViTs), Big Transfer models and Spiking Neural Networks (SNNs). We first conduct a transferability analysis, to demonstrate the adversarial examples generated by customized attacks on one defense, are not often misclassified by another defense.   This finding leads to two important questions. First, how can the low transferability between defenses be utilized in a game theoretic framework to improve the robustness? Second, how can an adversary within this framework develop effective multi-model attacks? In this paper, we provide a game-theoretic framework for ensemble adversarial attacks and defenses. Our framework is called Game theoretic Mixed Experts (GaME). It is designed to find the Mixed-Nash strategy for both a detector based and standard defender, when facing an attacker employing compositional adversarial attacks. We further propose three new attack algorithms, specifically designed to target defenses with randomized transformations, multi-model voting schemes, and adversarial detector architectures. These attacks serve to both strengthen defenses generated by the GaME framework and verify their robustness against unforeseen attacks. Overall, our framework and analyses advance the field of adversarial machine learning by yielding new insights into compositional attack and defense formulations.



## **19. Improving Hyperspectral Adversarial Robustness Under Multiple Attacks**

cs.LG

6 pages, 2 figures, 1 table, 1 algorithm

**SubmitDate**: 2023-04-28    [abs](http://arxiv.org/abs/2210.16346v3) [paper-pdf](http://arxiv.org/pdf/2210.16346v3)

**Authors**: Nicholas Soucy, Salimeh Yasaei Sekeh

**Abstract**: Semantic segmentation models classifying hyperspectral images (HSI) are vulnerable to adversarial examples. Traditional approaches to adversarial robustness focus on training or retraining a single network on attacked data, however, in the presence of multiple attacks these approaches decrease in performance compared to networks trained individually on each attack. To combat this issue we propose an Adversarial Discriminator Ensemble Network (ADE-Net) which focuses on attack type detection and adversarial robustness under a unified model to preserve per data-type weight optimally while robustifiying the overall network. In the proposed method, a discriminator network is used to separate data by attack type into their specific attack-expert ensemble network.



## **20. The Power of Typed Affine Decision Structures: A Case Study**

cs.LG

**SubmitDate**: 2023-04-28    [abs](http://arxiv.org/abs/2304.14888v1) [paper-pdf](http://arxiv.org/pdf/2304.14888v1)

**Authors**: Gerrit Nolte, Maximilian Schlüter, Alnis Murtovi, Bernhard Steffen

**Abstract**: TADS are a novel, concise white-box representation of neural networks. In this paper, we apply TADS to the problem of neural network verification, using them to generate either proofs or concise error characterizations for desirable neural network properties. In a case study, we consider the robustness of neural networks to adversarial attacks, i.e., small changes to an input that drastically change a neural networks perception, and show that TADS can be used to provide precise diagnostics on how and where robustness errors a occur. We achieve these results by introducing Precondition Projection, a technique that yields a TADS describing network behavior precisely on a given subset of its input space, and combining it with PCA, a traditional, well-understood dimensionality reduction technique. We show that PCA is easily compatible with TADS. All analyses can be implemented in a straightforward fashion using the rich algebraic properties of TADS, demonstrating the utility of the TADS framework for neural network explainability and verification. While TADS do not yet scale as efficiently as state-of-the-art neural network verifiers, we show that, using PCA-based simplifications, they can still scale to mediumsized problems and yield concise explanations for potential errors that can be used for other purposes such as debugging a network or generating new training samples.



## **21. Topic-oriented Adversarial Attacks against Black-box Neural Ranking Models**

cs.IR

Accepted by SIGIR 2023

**SubmitDate**: 2023-04-28    [abs](http://arxiv.org/abs/2304.14867v1) [paper-pdf](http://arxiv.org/pdf/2304.14867v1)

**Authors**: Yu-An Liu, Ruqing Zhang, Jiafeng Guo, Maarten de Rijke, Wei Chen, Yixing Fan, Xueqi Cheng

**Abstract**: Neural ranking models (NRMs) have attracted considerable attention in information retrieval. Unfortunately, NRMs may inherit the adversarial vulnerabilities of general neural networks, which might be leveraged by black-hat search engine optimization practitioners. Recently, adversarial attacks against NRMs have been explored in the paired attack setting, generating an adversarial perturbation to a target document for a specific query. In this paper, we focus on a more general type of perturbation and introduce the topic-oriented adversarial ranking attack task against NRMs, which aims to find an imperceptible perturbation that can promote a target document in ranking for a group of queries with the same topic. We define both static and dynamic settings for the task and focus on decision-based black-box attacks. We propose a novel framework to improve topic-oriented attack performance based on a surrogate ranking model. The attack problem is formalized as a Markov decision process (MDP) and addressed using reinforcement learning. Specifically, a topic-oriented reward function guides the policy to find a successful adversarial example that can be promoted in rankings to as many queries as possible in a group. Experimental results demonstrate that the proposed framework can significantly outperform existing attack strategies, and we conclude by re-iterating that there exist potential risks for applying NRMs in the real world.



## **22. False Claims against Model Ownership Resolution**

cs.CR

13pages,3 figures

**SubmitDate**: 2023-04-28    [abs](http://arxiv.org/abs/2304.06607v2) [paper-pdf](http://arxiv.org/pdf/2304.06607v2)

**Authors**: Jian Liu, Rui Zhang, Sebastian Szyller, Kui Ren, N. Asokan

**Abstract**: Deep neural network (DNN) models are valuable intellectual property of model owners, constituting a competitive advantage. Therefore, it is crucial to develop techniques to protect against model theft. Model ownership resolution (MOR) is a class of techniques that can deter model theft. A MOR scheme enables an accuser to assert an ownership claim for a suspect model by presenting evidence, such as a watermark or fingerprint, to show that the suspect model was stolen or derived from a source model owned by the accuser. Most of the existing MOR schemes prioritize robustness against malicious suspects, ensuring that the accuser will win if the suspect model is indeed a stolen model.   In this paper, we show that common MOR schemes in the literature are vulnerable to a different, equally important but insufficiently explored, robustness concern: a malicious accuser. We show how malicious accusers can successfully make false claims against independent suspect models that were not stolen. Our core idea is that a malicious accuser can deviate (without detection) from the specified MOR process by finding (transferable) adversarial examples that successfully serve as evidence against independent suspect models. To this end, we first generalize the procedures of common MOR schemes and show that, under this generalization, defending against false claims is as challenging as preventing (transferable) adversarial examples. Via systematic empirical evaluation we demonstrate that our false claim attacks always succeed in all prominent MOR schemes with realistic configurations, including against a real-world model: Amazon's Rekognition API.



## **23. Certified Robustness of Quantum Classifiers against Adversarial Examples through Quantum Noise**

quant-ph

Accepted to IEEE ICASSP 2023

**SubmitDate**: 2023-04-28    [abs](http://arxiv.org/abs/2211.00887v2) [paper-pdf](http://arxiv.org/pdf/2211.00887v2)

**Authors**: Jhih-Cing Huang, Yu-Lin Tsai, Chao-Han Huck Yang, Cheng-Fang Su, Chia-Mu Yu, Pin-Yu Chen, Sy-Yen Kuo

**Abstract**: Recently, quantum classifiers have been found to be vulnerable to adversarial attacks, in which quantum classifiers are deceived by imperceptible noises, leading to misclassification. In this paper, we propose the first theoretical study demonstrating that adding quantum random rotation noise can improve robustness in quantum classifiers against adversarial attacks. We link the definition of differential privacy and show that the quantum classifier trained with the natural presence of additive noise is differentially private. Finally, we derive a certified robustness bound to enable quantum classifiers to defend against adversarial examples, supported by experimental results simulated with noises from IBM's 7-qubits device.



## **24. Fusion is Not Enough: Single-Modal Attacks to Compromise Fusion Models in Autonomous Driving**

cs.CV

**SubmitDate**: 2023-04-28    [abs](http://arxiv.org/abs/2304.14614v1) [paper-pdf](http://arxiv.org/pdf/2304.14614v1)

**Authors**: Zhiyuan Cheng, Hongjun Choi, James Liang, Shiwei Feng, Guanhong Tao, Dongfang Liu, Michael Zuzak, Xiangyu Zhang

**Abstract**: Multi-sensor fusion (MSF) is widely adopted for perception in autonomous vehicles (AVs), particularly for the task of 3D object detection with camera and LiDAR sensors. The rationale behind fusion is to capitalize on the strengths of each modality while mitigating their limitations. The exceptional and leading performance of fusion models has been demonstrated by advanced deep neural network (DNN)-based fusion techniques. Fusion models are also perceived as more robust to attacks compared to single-modal ones due to the redundant information in multiple modalities. In this work, we challenge this perspective with single-modal attacks that targets the camera modality, which is considered less significant in fusion but more affordable for attackers. We argue that the weakest link of fusion models depends on their most vulnerable modality, and propose an attack framework that targets advanced camera-LiDAR fusion models with adversarial patches. Our approach employs a two-stage optimization-based strategy that first comprehensively assesses vulnerable image areas under adversarial attacks, and then applies customized attack strategies to different fusion models, generating deployable patches. Evaluations with five state-of-the-art camera-LiDAR fusion models on a real-world dataset show that our attacks successfully compromise all models. Our approach can either reduce the mean average precision (mAP) of detection performance from 0.824 to 0.353 or degrade the detection score of the target object from 0.727 to 0.151 on average, demonstrating the effectiveness and practicality of our proposed attack framework.



## **25. Efficient Reward Poisoning Attacks on Online Deep Reinforcement Learning**

cs.LG

**SubmitDate**: 2023-04-28    [abs](http://arxiv.org/abs/2205.14842v2) [paper-pdf](http://arxiv.org/pdf/2205.14842v2)

**Authors**: Yinglun Xu, Qi Zeng, Gagandeep Singh

**Abstract**: We study reward poisoning attacks on online deep reinforcement learning (DRL), where the attacker is oblivious to the learning algorithm used by the agent and the dynamics of the environment. We demonstrate the intrinsic vulnerability of state-of-the-art DRL algorithms by designing a general, black-box reward poisoning framework called adversarial MDP attacks. We instantiate our framework to construct two new attacks which only corrupt the rewards for a small fraction of the total training timesteps and make the agent learn a low-performing policy. We provide a theoretical analysis of the efficiency of our attack and perform an extensive empirical evaluation. Our results show that our attacks efficiently poison agents learning in several popular classical control and MuJoCo environments with a variety of state-of-the-art DRL algorithms, such as DQN, PPO, SAC, etc.



## **26. Adversary Aware Continual Learning**

cs.LG

**SubmitDate**: 2023-04-27    [abs](http://arxiv.org/abs/2304.14483v1) [paper-pdf](http://arxiv.org/pdf/2304.14483v1)

**Authors**: Muhammad Umer, Robi Polikar

**Abstract**: Class incremental learning approaches are useful as they help the model to learn new information (classes) sequentially, while also retaining the previously acquired information (classes). However, it has been shown that such approaches are extremely vulnerable to the adversarial backdoor attacks, where an intelligent adversary can introduce small amount of misinformation to the model in the form of imperceptible backdoor pattern during training to cause deliberate forgetting of a specific task or class at test time. In this work, we propose a novel defensive framework to counter such an insidious attack where, we use the attacker's primary strength-hiding the backdoor pattern by making it imperceptible to humans-against it, and propose to learn a perceptible (stronger) pattern (also during the training) that can overpower the attacker's imperceptible (weaker) pattern. We demonstrate the effectiveness of the proposed defensive mechanism through various commonly used Replay-based (both generative and exact replay-based) class incremental learning algorithms using continual learning benchmark variants of CIFAR-10, CIFAR-100, and MNIST datasets. Most noteworthy, our proposed defensive framework does not assume that the attacker's target task and target class is known to the defender. The defender is also unaware of the shape, size, and location of the attacker's pattern. We show that our proposed defensive framework considerably improves the performance of class incremental learning algorithms with no knowledge of the attacker's target task, attacker's target class, and attacker's imperceptible pattern. We term our defensive framework as Adversary Aware Continual Learning (AACL).



## **27. Attacking Fake News Detectors via Manipulating News Social Engagement**

cs.SI

ACM Web Conference 2023 (WWW'23)

**SubmitDate**: 2023-04-27    [abs](http://arxiv.org/abs/2302.07363v3) [paper-pdf](http://arxiv.org/pdf/2302.07363v3)

**Authors**: Haoran Wang, Yingtong Dou, Canyu Chen, Lichao Sun, Philip S. Yu, Kai Shu

**Abstract**: Social media is one of the main sources for news consumption, especially among the younger generation. With the increasing popularity of news consumption on various social media platforms, there has been a surge of misinformation which includes false information or unfounded claims. As various text- and social context-based fake news detectors are proposed to detect misinformation on social media, recent works start to focus on the vulnerabilities of fake news detectors. In this paper, we present the first adversarial attack framework against Graph Neural Network (GNN)-based fake news detectors to probe their robustness. Specifically, we leverage a multi-agent reinforcement learning (MARL) framework to simulate the adversarial behavior of fraudsters on social media. Research has shown that in real-world settings, fraudsters coordinate with each other to share different news in order to evade the detection of fake news detectors. Therefore, we modeled our MARL framework as a Markov Game with bot, cyborg, and crowd worker agents, which have their own distinctive cost, budget, and influence. We then use deep Q-learning to search for the optimal policy that maximizes the rewards. Extensive experimental results on two real-world fake news propagation datasets demonstrate that our proposed framework can effectively sabotage the GNN-based fake news detector performance. We hope this paper can provide insights for future research on fake news detection.



## **28. On the (In)security of Peer-to-Peer Decentralized Machine Learning**

cs.CR

IEEE S&P'23 (Previous title: "On the Privacy of Decentralized Machine  Learning")

**SubmitDate**: 2023-04-27    [abs](http://arxiv.org/abs/2205.08443v2) [paper-pdf](http://arxiv.org/pdf/2205.08443v2)

**Authors**: Dario Pasquini, Mathilde Raynal, Carmela Troncoso

**Abstract**: In this work, we carry out the first, in-depth, privacy analysis of Decentralized Learning -- a collaborative machine learning framework aimed at addressing the main limitations of federated learning. We introduce a suite of novel attacks for both passive and active decentralized adversaries. We demonstrate that, contrary to what is claimed by decentralized learning proposers, decentralized learning does not offer any security advantage over federated learning. Rather, it increases the attack surface enabling any user in the system to perform privacy attacks such as gradient inversion, and even gain full control over honest users' local model. We also show that, given the state of the art in protections, privacy-preserving configurations of decentralized learning require fully connected networks, losing any practical advantage over the federated setup and therefore completely defeating the objective of the decentralized approach.



## **29. Robust Resilient Signal Reconstruction under Adversarial Attacks**

math.OC

7 pages

**SubmitDate**: 2023-04-27    [abs](http://arxiv.org/abs/1807.08004v2) [paper-pdf](http://arxiv.org/pdf/1807.08004v2)

**Authors**: Yu Zheng, Olugbenga Moses Anubi, Lalit Mestha, Hema Achanta

**Abstract**: We consider the problem of signal reconstruction for a system under sparse signal corruption by a malicious agent. The reconstruction problem follows the standard error coding problem that has been studied extensively in the literature. We include a new challenge of robust estimation of the attack support. The problem is then cast as a constrained optimization problem merging promising techniques in the area of deep learning and estimation theory. A pruning algorithm is developed to reduce the ``false positive" uncertainty of data-driven attack localization results, thereby improving the probability of correct signal reconstruction. Sufficient conditions for the correct reconstruction and the associated reconstruction error bounds are obtained for both exact and inexact attack support estimation. Moreover, a simulation of a water distribution system is presented to validate the proposed techniques.



## **30. QEVSEC: Quick Electric Vehicle SEcure Charging via Dynamic Wireless Power Transfer**

cs.CR

6 pages, conference

**SubmitDate**: 2023-04-27    [abs](http://arxiv.org/abs/2205.10292v2) [paper-pdf](http://arxiv.org/pdf/2205.10292v2)

**Authors**: Tommaso Bianchi, Surudhi Asokraj, Alessandro Brighente, Mauro Conti, Radha Poovendran

**Abstract**: Dynamic Wireless Power Transfer (DWPT) can be used for on-demand recharging of Electric Vehicles (EV) while driving. However, DWPT raises numerous security and privacy concerns. Recently, researchers demonstrated that DWPT systems are vulnerable to adversarial attacks. In an EV charging scenario, an attacker can prevent the authorized customer from charging, obtain a free charge by billing a victim user and track a target vehicle. State-of-the-art authentication schemes relying on centralized solutions are either vulnerable to various attacks or have high computational complexity, making them unsuitable for a dynamic scenario. In this paper, we propose Quick Electric Vehicle SEcure Charging (QEVSEC), a novel, secure, and efficient authentication protocol for the dynamic charging of EVs. Our idea for QEVSEC originates from multiple vulnerabilities we found in the state-of-the-art protocol that allows tracking of user activity and is susceptible to replay attacks. Based on these observations, the proposed protocol solves these issues and achieves lower computational complexity by using only primitive cryptographic operations in a very short message exchange. QEVSEC provides scalability and a reduced cost in each iteration, thus lowering the impact on the power needed from the grid.



## **31. Boosting Big Brother: Attacking Search Engines with Encodings**

cs.CR

**SubmitDate**: 2023-04-27    [abs](http://arxiv.org/abs/2304.14031v1) [paper-pdf](http://arxiv.org/pdf/2304.14031v1)

**Authors**: Nicholas Boucher, Luca Pajola, Ilia Shumailov, Ross Anderson, Mauro Conti

**Abstract**: Search engines are vulnerable to attacks against indexing and searching via text encoding manipulation. By imperceptibly perturbing text using uncommon encoded representations, adversaries can control results across search engines for specific search queries. We demonstrate that this attack is successful against two major commercial search engines - Google and Bing - and one open source search engine - Elasticsearch. We further demonstrate that this attack is successful against LLM chat search including Bing's GPT-4 chatbot and Google's Bard chatbot. We also present a variant of the attack targeting text summarization and plagiarism detection models, two ML tasks closely tied to search. We provide a set of defenses against these techniques and warn that adversaries can leverage these attacks to launch disinformation campaigns against unsuspecting users, motivating the need for search engine maintainers to patch deployed systems.



## **32. You Can't Always Check What You Wanted: Selective Checking and Trusted Execution to Prevent False Actuations in Cyber-Physical Systems**

cs.CR

Extended version of SCATE published in ISORC'23

**SubmitDate**: 2023-04-27    [abs](http://arxiv.org/abs/2304.13956v1) [paper-pdf](http://arxiv.org/pdf/2304.13956v1)

**Authors**: Monowar Hasan, Sibin Mohan

**Abstract**: Cyber-physical systems (CPS) are vulnerable to attacks targeting outgoing actuation commands that modify their physical behaviors. The limited resources in such systems, coupled with their stringent timing constraints, often prevents the checking of every outgoing command. We present a "selective checking" mechanism that uses game-theoretic modeling to identify the right subset of commands to be checked in order to deter an adversary. This mechanism is coupled with a "delay-aware" trusted execution environment (TEE) to ensure that only verified actuation commands are ever sent to the physical system, thus maintaining their safety and integrity. The selective checking and trusted execution (SCATE) framework is implemented on an off-the-shelf ARM platform running standard embedded Linux. We demonstrate the effectiveness of SCATE using four realistic cyber-physical systems (a ground rover, a flight controller, a robotic arm and an automated syringe pump) and study design trade-offs. Not only does SCATE provide a high level of security and high performance, it also suffers from significantly lower overheads (30.48%-47.32% less) in the process. In fact, SCATE can work with more systems without negatively affecting the safety of the system. Considering that most CPS do not have any such checking mechanisms, and SCATE is guaranteed to meet all the timing requirements (i.e., ensure the safety/integrity of the system), our methods can significantly improve the security (and, hence, safety) of the system.



## **33. Network Cascade Vulnerability using Constrained Bayesian Optimization**

cs.SI

11 pages, 3 figures

**SubmitDate**: 2023-04-27    [abs](http://arxiv.org/abs/2304.14420v1) [paper-pdf](http://arxiv.org/pdf/2304.14420v1)

**Authors**: Albert Lam, Mihai Anitescu, Anirudh Subramanyam

**Abstract**: Measures of power grid vulnerability are often assessed by the amount of damage an adversary can exact on the network. However, the cascading impact of such attacks is often overlooked, even though cascades are one of the primary causes of large-scale blackouts. This paper explores modifications of transmission line protection settings as candidates for adversarial attacks, which can remain undetectable as long as the network equilibrium state remains unaltered. This forms the basis of a black-box function in a Bayesian optimization procedure, where the objective is to find protection settings that maximize network degradation due to cascading. Extensive experiments reveal that, against conventional wisdom, maximally misconfiguring the protection settings of all network lines does not cause the most cascading. More surprisingly, even when the degree of misconfiguration is resource constrained, it is still possible to find settings that produce cascades comparable in severity to instances where there are no constraints.



## **34. Detection of Adversarial Physical Attacks in Time-Series Image Data**

cs.CV

**SubmitDate**: 2023-04-27    [abs](http://arxiv.org/abs/2304.13919v1) [paper-pdf](http://arxiv.org/pdf/2304.13919v1)

**Authors**: Ramneet Kaur, Yiannis Kantaros, Wenwen Si, James Weimer, Insup Lee

**Abstract**: Deep neural networks (DNN) have become a common sensing modality in autonomous systems as they allow for semantically perceiving the ambient environment given input images. Nevertheless, DNN models have proven to be vulnerable to adversarial digital and physical attacks. To mitigate this issue, several detection frameworks have been proposed to detect whether a single input image has been manipulated by adversarial digital noise or not. In our prior work, we proposed a real-time detector, called VisionGuard (VG), for adversarial physical attacks against single input images to DNN models. Building upon that work, we propose VisionGuard* (VG), which couples VG with majority-vote methods, to detect adversarial physical attacks in time-series image data, e.g., videos. This is motivated by autonomous systems applications where images are collected over time using onboard sensors for decision-making purposes. We emphasize that majority-vote mechanisms are quite common in autonomous system applications (among many other applications), as e.g., in autonomous driving stacks for object detection. In this paper, we investigate, both theoretically and experimentally, how this widely used mechanism can be leveraged to enhance the performance of adversarial detectors. We have evaluated VG* on videos of both clean and physically attacked traffic signs generated by a state-of-the-art robust physical attack. We provide extensive comparative experiments against detectors that have been designed originally for out-of-distribution data and digitally attacked images.



## **35. Learning Robust Deep Equilibrium Models**

cs.LG

**SubmitDate**: 2023-04-26    [abs](http://arxiv.org/abs/2304.12707v2) [paper-pdf](http://arxiv.org/pdf/2304.12707v2)

**Authors**: Haoyu Chu, Shikui Wei, Ting Liu, Yao Zhao

**Abstract**: Deep equilibrium (DEQ) models have emerged as a promising class of implicit layer models in deep learning, which abandon traditional depth by solving for the fixed points of a single nonlinear layer. Despite their success, the stability of the fixed points for these models remains poorly understood. Recently, Lyapunov theory has been applied to Neural ODEs, another type of implicit layer model, to confer adversarial robustness. By considering DEQ models as nonlinear dynamic systems, we propose a robust DEQ model named LyaDEQ with guaranteed provable stability via Lyapunov theory. The crux of our method is ensuring the fixed points of the DEQ models are Lyapunov stable, which enables the LyaDEQ models to resist minor initial perturbations. To avoid poor adversarial defense due to Lyapunov-stable fixed points being located near each other, we add an orthogonal fully connected layer after the Lyapunov stability module to separate different fixed points. We evaluate LyaDEQ models on several widely used datasets under well-known adversarial attacks, and experimental results demonstrate significant improvement in robustness. Furthermore, we show that the LyaDEQ model can be combined with other defense methods, such as adversarial training, to achieve even better adversarial robustness.



## **36. One-vs-the-Rest Loss to Focus on Important Samples in Adversarial Training**

cs.LG

ICML2023, 26 pages, 19 figures

**SubmitDate**: 2023-04-26    [abs](http://arxiv.org/abs/2207.10283v3) [paper-pdf](http://arxiv.org/pdf/2207.10283v3)

**Authors**: Sekitoshi Kanai, Shin'ya Yamaguchi, Masanori Yamada, Hiroshi Takahashi, Kentaro Ohno, Yasutoshi Ida

**Abstract**: This paper proposes a new loss function for adversarial training. Since adversarial training has difficulties, e.g., necessity of high model capacity, focusing on important data points by weighting cross-entropy loss has attracted much attention. However, they are vulnerable to sophisticated attacks, e.g., Auto-Attack. This paper experimentally reveals that the cause of their vulnerability is their small margins between logits for the true label and the other labels. Since neural networks classify the data points based on the logits, logit margins should be large enough to avoid flipping the largest logit by the attacks. Importance-aware methods do not increase logit margins of important samples but decrease those of less-important samples compared with cross-entropy loss. To increase logit margins of important samples, we propose switching one-vs-the-rest loss (SOVR), which switches from cross-entropy to one-vs-the-rest loss for important samples that have small logit margins. We prove that one-vs-the-rest loss increases logit margins two times larger than the weighted cross-entropy loss for a simple problem. We experimentally confirm that SOVR increases logit margins of important samples unlike existing methods and achieves better robustness against Auto-Attack than importance-aware methods.



## **37. Improving Adversarial Transferability by Intermediate-level Perturbation Decay**

cs.LG

Revision of ICML '23 submission for better clarity

**SubmitDate**: 2023-04-26    [abs](http://arxiv.org/abs/2304.13410v1) [paper-pdf](http://arxiv.org/pdf/2304.13410v1)

**Authors**: Qizhang Li, Yiwen Guo, Wangmeng Zuo, Hao Chen

**Abstract**: Intermediate-level attacks that attempt to perturb feature representations following an adversarial direction drastically have shown favorable performance in crafting transferable adversarial examples. Existing methods in this category are normally formulated with two separate stages, where a directional guide is required to be determined at first and the scalar projection of the intermediate-level perturbation onto the directional guide is enlarged thereafter. The obtained perturbation deviates from the guide inevitably in the feature space, and it is revealed in this paper that such a deviation may lead to sub-optimal attack. To address this issue, we develop a novel intermediate-level method that crafts adversarial examples within a single stage of optimization. In particular, the proposed method, named intermediate-level perturbation decay (ILPD), encourages the intermediate-level perturbation to be in an effective adversarial direction and to possess a great magnitude simultaneously. In-depth discussion verifies the effectiveness of our method. Experimental results show that it outperforms state-of-the-arts by large margins in attacking various victim models on ImageNet (+10.07% on average) and CIFAR-10 (+3.88% on average). Our code is at https://github.com/qizhangli/ILPD-attack.



## **38. Blockchain-based Access Control for Secure Smart Industry Management Systems**

cs.CR

**SubmitDate**: 2023-04-26    [abs](http://arxiv.org/abs/2304.13379v1) [paper-pdf](http://arxiv.org/pdf/2304.13379v1)

**Authors**: Aditya Pribadi Kalapaaking, Ibrahim Khalil, Mohammad Saidur Rahman, Abdelaziz Bouras

**Abstract**: Smart manufacturing systems involve a large number of interconnected devices resulting in massive data generation. Cloud computing technology has recently gained increasing attention in smart manufacturing systems for facilitating cost-effective service provisioning and massive data management. In a cloud-based manufacturing system, ensuring authorized access to the data is crucial. A cloud platform is operated under a single authority. Hence, a cloud platform is prone to a single point of failure and vulnerable to adversaries. An internal or external adversary can easily modify users' access to allow unauthorized users to access the data. This paper proposes a role-based access control to prevent modification attacks by leveraging blockchain and smart contracts in a cloud-based smart manufacturing system. The role-based access control is developed to determine users' roles and rights in smart contracts. The smart contracts are then deployed to the private blockchain network. We evaluate our solution by utilizing Ethereum private blockchain network to deploy the smart contract. The experimental results demonstrate the feasibility and evaluation of the proposed framework's performance.



## **39. Blockchain-based Federated Learning with SMPC Model Verification Against Poisoning Attack for Healthcare Systems**

cs.CR

**SubmitDate**: 2023-04-26    [abs](http://arxiv.org/abs/2304.13360v1) [paper-pdf](http://arxiv.org/pdf/2304.13360v1)

**Authors**: Aditya Pribadi Kalapaaking, Ibrahim Khalil, Xun Yi

**Abstract**: Due to the rising awareness of privacy and security in machine learning applications, federated learning (FL) has received widespread attention and applied to several areas, e.g., intelligence healthcare systems, IoT-based industries, and smart cities. FL enables clients to train a global model collaboratively without accessing their local training data. However, the current FL schemes are vulnerable to adversarial attacks. Its architecture makes detecting and defending against malicious model updates difficult. In addition, most recent studies to detect FL from malicious updates while maintaining the model's privacy have not been sufficiently explored. This paper proposed blockchain-based federated learning with SMPC model verification against poisoning attacks for healthcare systems. First, we check the machine learning model from the FL participants through an encrypted inference process and remove the compromised model. Once the participants' local models have been verified, the models are sent to the blockchain node to be securely aggregated. We conducted several experiments with different medical datasets to evaluate our proposed framework.



## **40. On the Risks of Stealing the Decoding Algorithms of Language Models**

cs.LG

**SubmitDate**: 2023-04-26    [abs](http://arxiv.org/abs/2303.04729v3) [paper-pdf](http://arxiv.org/pdf/2303.04729v3)

**Authors**: Ali Naseh, Kalpesh Krishna, Mohit Iyyer, Amir Houmansadr

**Abstract**: A key component of generating text from modern language models (LM) is the selection and tuning of decoding algorithms. These algorithms determine how to generate text from the internal probability distribution generated by the LM. The process of choosing a decoding algorithm and tuning its hyperparameters takes significant time, manual effort, and computation, and it also requires extensive human evaluation. Therefore, the identity and hyperparameters of such decoding algorithms are considered to be extremely valuable to their owners. In this work, we show, for the first time, that an adversary with typical API access to an LM can steal the type and hyperparameters of its decoding algorithms at very low monetary costs. Our attack is effective against popular LMs used in text generation APIs, including GPT-2 and GPT-3. We demonstrate the feasibility of stealing such information with only a few dollars, e.g., $\$0.8$, $\$1$, $\$4$, and $\$40$ for the four versions of GPT-3.



## **41. SHIELD: Thwarting Code Authorship Attribution**

cs.CR

12 pages, 13 figures

**SubmitDate**: 2023-04-26    [abs](http://arxiv.org/abs/2304.13255v1) [paper-pdf](http://arxiv.org/pdf/2304.13255v1)

**Authors**: Mohammed Abuhamad, Changhun Jung, David Mohaisen, DaeHun Nyang

**Abstract**: Authorship attribution has become increasingly accurate, posing a serious privacy risk for programmers who wish to remain anonymous. In this paper, we introduce SHIELD to examine the robustness of different code authorship attribution approaches against adversarial code examples. We define four attacks on attribution techniques, which include targeted and non-targeted attacks, and realize them using adversarial code perturbation. We experiment with a dataset of 200 programmers from the Google Code Jam competition to validate our methods targeting six state-of-the-art authorship attribution methods that adopt a variety of techniques for extracting authorship traits from source-code, including RNN, CNN, and code stylometry. Our experiments demonstrate the vulnerability of current authorship attribution methods against adversarial attacks. For the non-targeted attack, our experiments demonstrate the vulnerability of current authorship attribution methods against the attack with an attack success rate exceeds 98.5\% accompanied by a degradation of the identification confidence that exceeds 13\%. For the targeted attacks, we show the possibility of impersonating a programmer using targeted-adversarial perturbations with a success rate ranging from 66\% to 88\% for different authorship attribution techniques under several adversarial scenarios.



## **42. Generating Adversarial Examples with Task Oriented Multi-Objective Optimization**

cs.LG

**SubmitDate**: 2023-04-26    [abs](http://arxiv.org/abs/2304.13229v1) [paper-pdf](http://arxiv.org/pdf/2304.13229v1)

**Authors**: Anh Bui, Trung Le, He Zhao, Quan Tran, Paul Montague, Dinh Phung

**Abstract**: Deep learning models, even the-state-of-the-art ones, are highly vulnerable to adversarial examples. Adversarial training is one of the most efficient methods to improve the model's robustness. The key factor for the success of adversarial training is the capability to generate qualified and divergent adversarial examples which satisfy some objectives/goals (e.g., finding adversarial examples that maximize the model losses for simultaneously attacking multiple models). Therefore, multi-objective optimization (MOO) is a natural tool for adversarial example generation to achieve multiple objectives/goals simultaneously. However, we observe that a naive application of MOO tends to maximize all objectives/goals equally, without caring if an objective/goal has been achieved yet. This leads to useless effort to further improve the goal-achieved tasks, while putting less focus on the goal-unachieved tasks. In this paper, we propose \emph{Task Oriented MOO} to address this issue, in the context where we can explicitly define the goal achievement for a task. Our principle is to only maintain the goal-achieved tasks, while letting the optimizer spend more effort on improving the goal-unachieved tasks. We conduct comprehensive experiments for our Task Oriented MOO on various adversarial example generation schemes. The experimental results firmly demonstrate the merit of our proposed approach. Our code is available at \url{https://github.com/tuananhbui89/TAMOO}.



## **43. Uncovering the Representation of Spiking Neural Networks Trained with Surrogate Gradient**

cs.LG

Published in Transactions on Machine Learning Research (TMLR)

**SubmitDate**: 2023-04-25    [abs](http://arxiv.org/abs/2304.13098v1) [paper-pdf](http://arxiv.org/pdf/2304.13098v1)

**Authors**: Yuhang Li, Youngeun Kim, Hyoungseob Park, Priyadarshini Panda

**Abstract**: Spiking Neural Networks (SNNs) are recognized as the candidate for the next-generation neural networks due to their bio-plausibility and energy efficiency. Recently, researchers have demonstrated that SNNs are able to achieve nearly state-of-the-art performance in image recognition tasks using surrogate gradient training. However, some essential questions exist pertaining to SNNs that are little studied: Do SNNs trained with surrogate gradient learn different representations from traditional Artificial Neural Networks (ANNs)? Does the time dimension in SNNs provide unique representation power? In this paper, we aim to answer these questions by conducting a representation similarity analysis between SNNs and ANNs using Centered Kernel Alignment (CKA). We start by analyzing the spatial dimension of the networks, including both the width and the depth. Furthermore, our analysis of residual connections shows that SNNs learn a periodic pattern, which rectifies the representations in SNNs to be ANN-like. We additionally investigate the effect of the time dimension on SNN representation, finding that deeper layers encourage more dynamics along the time dimension. We also investigate the impact of input data such as event-stream data and adversarial attacks. Our work uncovers a host of new findings of representations in SNNs. We hope this work will inspire future research to fully comprehend the representation power of SNNs. Code is released at https://github.com/Intelligent-Computing-Lab-Yale/SNNCKA.



## **44. Improving Robustness Against Adversarial Attacks with Deeply Quantized Neural Networks**

cs.LG

Accepted at IJCNN 2023. 8 pages, 5 figures

**SubmitDate**: 2023-04-25    [abs](http://arxiv.org/abs/2304.12829v1) [paper-pdf](http://arxiv.org/pdf/2304.12829v1)

**Authors**: Ferheen Ayaz, Idris Zakariyya, José Cano, Sye Loong Keoh, Jeremy Singer, Danilo Pau, Mounia Kharbouche-Harrari

**Abstract**: Reducing the memory footprint of Machine Learning (ML) models, particularly Deep Neural Networks (DNNs), is essential to enable their deployment into resource-constrained tiny devices. However, a disadvantage of DNN models is their vulnerability to adversarial attacks, as they can be fooled by adding slight perturbations to the inputs. Therefore, the challenge is how to create accurate, robust, and tiny DNN models deployable on resource-constrained embedded devices. This paper reports the results of devising a tiny DNN model, robust to adversarial black and white box attacks, trained with an automatic quantizationaware training framework, i.e. QKeras, with deep quantization loss accounted in the learning loop, thereby making the designed DNNs more accurate for deployment on tiny devices. We investigated how QKeras and an adversarial robustness technique, Jacobian Regularization (JR), can provide a co-optimization strategy by exploiting the DNN topology and the per layer JR approach to produce robust yet tiny deeply quantized DNN models. As a result, a new DNN model implementing this cooptimization strategy was conceived, developed and tested on three datasets containing both images and audio inputs, as well as compared its performance with existing benchmarks against various white-box and black-box attacks. Experimental results demonstrated that on average our proposed DNN model resulted in 8.3% and 79.5% higher accuracy than MLCommons/Tiny benchmarks in the presence of white-box and black-box attacks on the CIFAR-10 image dataset and a subset of the Google Speech Commands audio dataset respectively. It was also 6.5% more accurate for black-box attacks on the SVHN image dataset.



## **45. RobCaps: Evaluating the Robustness of Capsule Networks against Affine Transformations and Adversarial Attacks**

cs.LG

To appear at the 2023 International Joint Conference on Neural  Networks (IJCNN), Queensland, Australia, June 2023

**SubmitDate**: 2023-04-25    [abs](http://arxiv.org/abs/2304.03973v2) [paper-pdf](http://arxiv.org/pdf/2304.03973v2)

**Authors**: Alberto Marchisio, Antonio De Marco, Alessio Colucci, Maurizio Martina, Muhammad Shafique

**Abstract**: Capsule Networks (CapsNets) are able to hierarchically preserve the pose relationships between multiple objects for image classification tasks. Other than achieving high accuracy, another relevant factor in deploying CapsNets in safety-critical applications is the robustness against input transformations and malicious adversarial attacks.   In this paper, we systematically analyze and evaluate different factors affecting the robustness of CapsNets, compared to traditional Convolutional Neural Networks (CNNs). Towards a comprehensive comparison, we test two CapsNet models and two CNN models on the MNIST, GTSRB, and CIFAR10 datasets, as well as on the affine-transformed versions of such datasets. With a thorough analysis, we show which properties of these architectures better contribute to increasing the robustness and their limitations. Overall, CapsNets achieve better robustness against adversarial examples and affine transformations, compared to a traditional CNN with a similar number of parameters. Similar conclusions have been derived for deeper versions of CapsNets and CNNs. Moreover, our results unleash a key finding that the dynamic routing does not contribute much to improving the CapsNets' robustness. Indeed, the main generalization contribution is due to the hierarchical feature learning through capsules.



## **46. StratDef: Strategic Defense Against Adversarial Attacks in ML-based Malware Detection**

cs.LG

**SubmitDate**: 2023-04-24    [abs](http://arxiv.org/abs/2202.07568v6) [paper-pdf](http://arxiv.org/pdf/2202.07568v6)

**Authors**: Aqib Rashid, Jose Such

**Abstract**: Over the years, most research towards defenses against adversarial attacks on machine learning models has been in the image recognition domain. The ML-based malware detection domain has received less attention despite its importance. Moreover, most work exploring these defenses has focused on several methods but with no strategy when applying them. In this paper, we introduce StratDef, which is a strategic defense system based on a moving target defense approach. We overcome challenges related to the systematic construction, selection, and strategic use of models to maximize adversarial robustness. StratDef dynamically and strategically chooses the best models to increase the uncertainty for the attacker while minimizing critical aspects in the adversarial ML domain, like attack transferability. We provide the first comprehensive evaluation of defenses against adversarial attacks on machine learning for malware detection, where our threat model explores different levels of threat, attacker knowledge, capabilities, and attack intensities. We show that StratDef performs better than other defenses even when facing the peak adversarial threat. We also show that, of the existing defenses, only a few adversarially-trained models provide substantially better protection than just using vanilla models but are still outperformed by StratDef.



## **47. On Adversarial Robustness of Point Cloud Semantic Segmentation**

cs.CV

**SubmitDate**: 2023-04-23    [abs](http://arxiv.org/abs/2112.05871v4) [paper-pdf](http://arxiv.org/pdf/2112.05871v4)

**Authors**: Jiacen Xu, Zhe Zhou, Boyuan Feng, Yufei Ding, Zhou Li

**Abstract**: Recent research efforts on 3D point cloud semantic segmentation (PCSS) have achieved outstanding performance by adopting neural networks. However, the robustness of these complex models have not been systematically analyzed. Given that PCSS has been applied in many safety-critical applications like autonomous driving, it is important to fill this knowledge gap, especially, how these models are affected under adversarial samples. As such, we present a comparative study of PCSS robustness. First, we formally define the attacker's objective under performance degradation and object hiding. Then, we develop new attack by whether to bound the norm. We evaluate different attack options on two datasets and three PCSS models. We found all the models are vulnerable and attacking point color is more effective. With this study, we call the attention of the research community to develop new approaches to harden PCSS models.



## **48. Evading DeepFake Detectors via Adversarial Statistical Consistency**

cs.CV

Accepted by CVPR 2023

**SubmitDate**: 2023-04-23    [abs](http://arxiv.org/abs/2304.11670v1) [paper-pdf](http://arxiv.org/pdf/2304.11670v1)

**Authors**: Yang Hou, Qing Guo, Yihao Huang, Xiaofei Xie, Lei Ma, Jianjun Zhao

**Abstract**: In recent years, as various realistic face forgery techniques known as DeepFake improves by leaps and bounds,more and more DeepFake detection techniques have been proposed. These methods typically rely on detecting statistical differences between natural (i.e., real) and DeepFakegenerated images in both spatial and frequency domains. In this work, we propose to explicitly minimize the statistical differences to evade state-of-the-art DeepFake detectors. To this end, we propose a statistical consistency attack (StatAttack) against DeepFake detectors, which contains two main parts. First, we select several statistical-sensitive natural degradations (i.e., exposure, blur, and noise) and add them to the fake images in an adversarial way. Second, we find that the statistical differences between natural and DeepFake images are positively associated with the distribution shifting between the two kinds of images, and we propose to use a distribution-aware loss to guide the optimization of different degradations. As a result, the feature distributions of generated adversarial examples is close to the natural images.Furthermore, we extend the StatAttack to a more powerful version, MStatAttack, where we extend the single-layer degradation to multi-layer degradations sequentially and use the loss to tune the combination weights jointly. Comprehensive experimental results on four spatial-based detectors and two frequency-based detectors with four datasets demonstrate the effectiveness of our proposed attack method in both white-box and black-box settings.



## **49. Partial-Information, Longitudinal Cyber Attacks on LiDAR in Autonomous Vehicles**

cs.CR

**SubmitDate**: 2023-04-23    [abs](http://arxiv.org/abs/2303.03470v2) [paper-pdf](http://arxiv.org/pdf/2303.03470v2)

**Authors**: R. Spencer Hallyburton, Qingzhao Zhang, Z. Morley Mao, Miroslav Pajic

**Abstract**: What happens to an autonomous vehicle (AV) if its data are adversarially compromised? Prior security studies have addressed this question through mostly unrealistic threat models, with limited practical relevance, such as white-box adversarial learning or nanometer-scale laser aiming and spoofing. With growing evidence that cyber threats pose real, imminent danger to AVs and cyber-physical systems (CPS) in general, we present and evaluate a novel AV threat model: a cyber-level attacker capable of disrupting sensor data but lacking any situational awareness. We demonstrate that even though the attacker has minimal knowledge and only access to raw data from a single sensor (i.e., LiDAR), she can design several attacks that critically compromise perception and tracking in multi-sensor AVs. To mitigate vulnerabilities and advance secure architectures in AVs, we introduce two improvements for security-aware fusion: a probabilistic data-asymmetry monitor and a scalable track-to-track fusion of 3D LiDAR and monocular detections (T2T-3DLM); we demonstrate that the approaches significantly reduce attack effectiveness. To support objective safety and security evaluations in AVs, we release our security evaluation platform, AVsec, which is built on security-relevant metrics to benchmark AVs on gold-standard longitudinal AV datasets and AV simulators.



## **50. Disco Intelligent Reflecting Surfaces: Active Channel Aging for Fully-Passive Jamming Attacks**

eess.SP

**SubmitDate**: 2023-04-23    [abs](http://arxiv.org/abs/2302.00415v2) [paper-pdf](http://arxiv.org/pdf/2302.00415v2)

**Authors**: Huan Huang, Ying Zhang, Hongliang Zhang, Yi Cai, A. Lee Swindlehurst, Zhu Han

**Abstract**: Due to the open communications environment in wireless channels, wireless networks are vulnerable to jamming attacks. However, existing approaches for jamming rely on knowledge of the legitimate users' (LUs') channels, extra jamming power, or both. To raise concerns about the potential threats posed by illegitimate intelligent reflecting surfaces (IRSs), we propose an alternative method to launch jamming attacks on LUs without either LU channel state information (CSI) or jamming power. The proposed approach employs an adversarial IRS with random phase shifts, referred to as a "disco" IRS (DIRS), that acts like a "disco ball" to actively age the LUs' channels. Such active channel aging (ACA) interference can be used to launch jamming attacks on multi-user multiple-input single-output (MU-MISO) systems. The proposed DIRS-based fully-passive jammer (FPJ) can jam LUs with no additional jamming power or knowledge of the LU CSI, and it can not be mitigated by classical anti-jamming approaches. A theoretical analysis of the proposed DIRS-based FPJ that provides an evaluation of the DIRS-based jamming attacks is derived. Based on this detailed theoretical analysis, some unique properties of the proposed DIRS-based FPJ can be obtained. Furthermore, a design example of the proposed DIRS-based FPJ based on one-bit quantization of the IRS phases is demonstrated to be sufficient for implementing the jamming attack. In addition, numerical results are provided to show the effectiveness of the derived theoretical analysis and the jamming impact of the proposed DIRS-based FPJ.



