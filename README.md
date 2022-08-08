# Latest Adversarial Attack Papers
**update at 2022-08-09 06:31:23**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. Attacking Adversarial Defences by Smoothing the Loss Landscape**

cs.LG

**SubmitDate**: 2022-08-05    [paper-pdf](http://arxiv.org/pdf/2208.00862v2)

**Authors**: Panagiotis Eustratiadis, Henry Gouk, Da Li, Timothy Hospedales

**Abstracts**: This paper investigates a family of methods for defending against adversarial attacks that owe part of their success to creating a noisy, discontinuous, or otherwise rugged loss landscape that adversaries find difficult to navigate. A common, but not universal, way to achieve this effect is via the use of stochastic neural networks. We show that this is a form of gradient obfuscation, and propose a general extension to gradient-based adversaries based on the Weierstrass transform, which smooths the surface of the loss function and provides more reliable gradient estimates. We further show that the same principle can strengthen gradient-free adversaries. We demonstrate the efficacy of our loss-smoothing method against both stochastic and non-stochastic adversarial defences that exhibit robustness due to this type of obfuscation. Furthermore, we provide analysis of how it interacts with Expectation over Transformation; a popular gradient-sampling method currently used to attack stochastic defences.



## **2. Adversarial Robustness of MR Image Reconstruction under Realistic Perturbations**

eess.IV

Accepted at the MICCAI-2022 workshop: Machine Learning for Medical  Image Reconstruction

**SubmitDate**: 2022-08-05    [paper-pdf](http://arxiv.org/pdf/2208.03161v1)

**Authors**: Jan Nikolas Morshuis, Sergios Gatidis, Matthias Hein, Christian F. Baumgartner

**Abstracts**: Deep Learning (DL) methods have shown promising results for solving ill-posed inverse problems such as MR image reconstruction from undersampled $k$-space data. However, these approaches currently have no guarantees for reconstruction quality and the reliability of such algorithms is only poorly understood. Adversarial attacks offer a valuable tool to understand possible failure modes and worst case performance of DL-based reconstruction algorithms. In this paper we describe adversarial attacks on multi-coil $k$-space measurements and evaluate them on the recently proposed E2E-VarNet and a simpler UNet-based model. In contrast to prior work, the attacks are targeted to specifically alter diagnostically relevant regions. Using two realistic attack models (adversarial $k$-space noise and adversarial rotations) we are able to show that current state-of-the-art DL-based reconstruction algorithms are indeed sensitive to such perturbations to a degree where relevant diagnostic information may be lost. Surprisingly, in our experiments the UNet and the more sophisticated E2E-VarNet were similarly sensitive to such attacks. Our findings add further to the evidence that caution must be exercised as DL-based methods move closer to clinical practice.



## **3. A Systematic Survey of Attack Detection and Prevention in Connected and Autonomous Vehicles**

cs.CR

This article is published in the Vehicular Communications journal

**SubmitDate**: 2022-08-05    [paper-pdf](http://arxiv.org/pdf/2203.14965v2)

**Authors**: Trupil Limbasiya, Ko Zheng Teng, Sudipta Chattopadhyay, Jianying Zhou

**Abstracts**: The number of Connected and Autonomous Vehicles (CAVs) is increasing rapidly in various smart transportation services and applications, considering many benefits to society, people, and the environment. Several research surveys for CAVs were conducted by primarily focusing on various security threats and vulnerabilities in the domain of CAVs to classify different types of attacks, impacts of attacks, attack features, cyber-risk, defense methodologies against attacks, and safety standards. However, the importance of attack detection and prevention approaches for CAVs has not been discussed extensively in the state-of-the-art surveys, and there is a clear gap in the existing literature on such methodologies to detect new and conventional threats and protect the CAV systems from unexpected hazards on the road. Some surveys have a limited discussion on Attacks Detection and Prevention Systems (ADPS), but such surveys provide only partial coverage of different types of ADPS for CAVs. Furthermore, there is a scope for discussing security, privacy, and efficiency challenges in ADPS that can give an overview of important security and performance attributes.   This survey paper, therefore, presents the significance of CAVs in the market, potential challenges in CAVs, key requirements of essential security and privacy properties, various capabilities of adversaries, possible attacks in CAVs, and performance evaluation parameters for ADPS. An extensive analysis is discussed of different ADPS categories for CAVs and state-of-the-art research works based on each ADPS category that gives the latest findings in this research domain. This survey also discusses crucial and open security research problems that are required to be focused on the secure deployment of CAVs in the market.



## **4. Differentially Private Counterfactuals via Functional Mechanism**

cs.LG

**SubmitDate**: 2022-08-04    [paper-pdf](http://arxiv.org/pdf/2208.02878v1)

**Authors**: Fan Yang, Qizhang Feng, Kaixiong Zhou, Jiahao Chen, Xia Hu

**Abstracts**: Counterfactual, serving as one emerging type of model explanation, has attracted tons of attentions recently from both industry and academia. Different from the conventional feature-based explanations (e.g., attributions), counterfactuals are a series of hypothetical samples which can flip model decisions with minimal perturbations on queries. Given valid counterfactuals, humans are capable of reasoning under ``what-if'' circumstances, so as to better understand the model decision boundaries. However, releasing counterfactuals could be detrimental, since it may unintentionally leak sensitive information to adversaries, which brings about higher risks on both model security and data privacy. To bridge the gap, in this paper, we propose a novel framework to generate differentially private counterfactual (DPC) without touching the deployed model or explanation set, where noises are injected for protection while maintaining the explanation roles of counterfactual. In particular, we train an autoencoder with the functional mechanism to construct noisy class prototypes, and then derive the DPC from the latent prototypes based on the post-processing immunity of differential privacy. Further evaluations demonstrate the effectiveness of the proposed framework, showing that DPC can successfully relieve the risks on both extraction and inference attacks.



## **5. Self-Ensembling Vision Transformer (SEViT) for Robust Medical Image Classification**

cs.CV

**SubmitDate**: 2022-08-04    [paper-pdf](http://arxiv.org/pdf/2208.02851v1)

**Authors**: Faris Almalik, Mohammad Yaqub, Karthik Nandakumar

**Abstracts**: Vision Transformers (ViT) are competing to replace Convolutional Neural Networks (CNN) for various computer vision tasks in medical imaging such as classification and segmentation. While the vulnerability of CNNs to adversarial attacks is a well-known problem, recent works have shown that ViTs are also susceptible to such attacks and suffer significant performance degradation under attack. The vulnerability of ViTs to carefully engineered adversarial samples raises serious concerns about their safety in clinical settings. In this paper, we propose a novel self-ensembling method to enhance the robustness of ViT in the presence of adversarial attacks. The proposed Self-Ensembling Vision Transformer (SEViT) leverages the fact that feature representations learned by initial blocks of a ViT are relatively unaffected by adversarial perturbations. Learning multiple classifiers based on these intermediate feature representations and combining these predictions with that of the final ViT classifier can provide robustness against adversarial attacks. Measuring the consistency between the various predictions can also help detect adversarial samples. Experiments on two modalities (chest X-ray and fundoscopy) demonstrate the efficacy of SEViT architecture to defend against various adversarial attacks in the gray-box (attacker has full knowledge of the target model, but not the defense mechanism) setting. Code: https://github.com/faresmalik/SEViT



## **6. Mass Exit Attacks on the Lightning Network**

cs.CR

**SubmitDate**: 2022-08-04    [paper-pdf](http://arxiv.org/pdf/2208.01908v2)

**Authors**: Cosimo Sguanci, Anastasios Sidiropoulos

**Abstracts**: The Lightning Network (LN) has enjoyed rapid growth over recent years, and has become the most popular scaling solution for the Bitcoin blockchain. The security of the LN hinges on the ability of the nodes to close a channel by settling their balances, which requires confirming a transaction on the Bitcoin blockchain within a pre-agreed time period. This inherent timing restriction that the LN must satisfy, make it susceptible to attacks that seek to increase the congestion on the Bitcoin blockchain, thus preventing correct protocol execution. We study the susceptibility of the LN to \emph{mass exit} attacks, in the presence of a small coalition of adversarial nodes. This is a scenario where an adversary forces a large set of honest protocol participants to interact with the blockchain. We focus on two types of attacks: (i) The first is a \emph{zombie} attack, where a set of $k$ nodes become unresponsive with the goal to lock the funds of many channels for a period of time longer than what the LN protocol dictates. (ii) The second is a \emph{mass double-spend} attack, where a set of $k$ nodes attempt to steal funds by submitting many closing transactions that settle channels using expired protocol states; this causes many honest nodes to have to quickly respond by submitting invalidating transactions. We show via simulations that, under historically-plausible congestion conditions, with mild statistical assumptions on channel balances, both of the attacks can be performed by a very small coalition. To perform our simulations, we formulate the problem of finding a worst-case coalition of $k$ adversarial nodes as a graph cut problem. Our experimental findings are supported by a theoretical justification based on the scale-free topology of the LN.



## **7. Design Considerations and Architecture for a Resilient Risk based Adaptive Authentication and Authorization (RAD-AA) Framework**

cs.CR

**SubmitDate**: 2022-08-04    [paper-pdf](http://arxiv.org/pdf/2208.02592v1)

**Authors**: Jaimandeep Singh, Chintan Patel, Naveen Kumar Chaudhary

**Abstracts**: A strong cyber attack is capable of degrading the performance of any Information Technology (IT) or Operational Technology (OT) system. In recent cyber attacks, credential theft emerged as one of the primary vectors of gaining entry into the system. Once, an attacker has a foothold in the system, they use token manipulation techniques to elevate the privileges and access protected resources. This makes authentication and authorization a critical component for a secure and resilient cyber system. In this paper we consider the design considerations for such a secure and resilient authentication and authorization framework capable of self-adapting based on the risk scores and trust profiles. We compare this design with the existing standards such as OAuth 2.0, OpenID Connect and SAML 2.0. We then study popular threat models such as STRIDE and PASTA and summarize the resilience of the proposed architecture against common and relevant threat vectors. We call this framework Resilient Risk-based Adaptive Authentication and Authorization (RAD-AA). The proposed framework excessively increases the cost for an adversary to launch any cyber attack and provides much-needed strength to critical infrastructure.



## **8. Prompt Tuning for Generative Multimodal Pretrained Models**

cs.CL

Work in progress

**SubmitDate**: 2022-08-04    [paper-pdf](http://arxiv.org/pdf/2208.02532v1)

**Authors**: Hao Yang, Junyang Lin, An Yang, Peng Wang, Chang Zhou, Hongxia Yang

**Abstracts**: Prompt tuning has become a new paradigm for model tuning and it has demonstrated success in natural language pretraining and even vision pretraining. In this work, we explore the transfer of prompt tuning to multimodal pretraining, with a focus on generative multimodal pretrained models, instead of contrastive ones. Specifically, we implement prompt tuning on the unified sequence-to-sequence pretrained model adaptive to both understanding and generation tasks. Experimental results demonstrate that the light-weight prompt tuning can achieve comparable performance with finetuning and surpass other light-weight tuning methods. Besides, in comparison with finetuned models, the prompt-tuned models demonstrate improved robustness against adversarial attacks. We further figure out that experimental factors, including the prompt length, prompt depth, and reparameteratization, have great impacts on the model performance, and thus we empirically provide a recommendation for the setups of prompt tuning. Despite the observed advantages, we still find some limitations in prompt tuning, and we correspondingly point out the directions for future studies. Codes are available at \url{https://github.com/OFA-Sys/OFA}



## **9. NoiLIn: Improving Adversarial Training and Correcting Stereotype of Noisy Labels**

cs.LG

Accepted at Transactions on Machine Learning Research (TMLR) at June  2022

**SubmitDate**: 2022-08-04    [paper-pdf](http://arxiv.org/pdf/2105.14676v2)

**Authors**: Jingfeng Zhang, Xilie Xu, Bo Han, Tongliang Liu, Gang Niu, Lizhen Cui, Masashi Sugiyama

**Abstracts**: Adversarial training (AT) formulated as the minimax optimization problem can effectively enhance the model's robustness against adversarial attacks. The existing AT methods mainly focused on manipulating the inner maximization for generating quality adversarial variants or manipulating the outer minimization for designing effective learning objectives. However, empirical results of AT always exhibit the robustness at odds with accuracy and the existence of the cross-over mixture problem, which motivates us to study some label randomness for benefiting the AT. First, we thoroughly investigate noisy labels (NLs) injection into AT's inner maximization and outer minimization, respectively and obtain the observations on when NL injection benefits AT. Second, based on the observations, we propose a simple but effective method -- NoiLIn that randomly injects NLs into training data at each training epoch and dynamically increases the NL injection rate once robust overfitting occurs. Empirically, NoiLIn can significantly mitigate the AT's undesirable issue of robust overfitting and even further improve the generalization of the state-of-the-art AT methods. Philosophically, NoiLIn sheds light on a new perspective of learning with NLs: NLs should not always be deemed detrimental, and even in the absence of NLs in the training set, we may consider injecting them deliberately. Codes are available in https://github.com/zjfheart/NoiLIn.



## **10. A Robust graph attention network with dynamic adjusted Graph**

cs.LG

21 pages,13 figures

**SubmitDate**: 2022-08-04    [paper-pdf](http://arxiv.org/pdf/2009.13038v3)

**Authors**: Xianchen Zhou, Yaoyun Zeng, Hongxia Wang

**Abstracts**: Graph Attention Networks(GATs) are useful deep learning models to deal with the graph data. However, recent works show that the classical GAT is vulnerable to adversarial attacks. It degrades dramatically with slight perturbations. Therefore, how to enhance the robustness of GAT is a critical problem. Robust GAT(RoGAT) is proposed in this paper to improve the robustness of GAT based on the revision of the attention mechanism. Different from the original GAT, which uses the attention mechanism for different edges but is still sensitive to the perturbation, RoGAT adds an extra dynamic attention score progressively and improves the robustness. Firstly, RoGAT revises the edges weight based on the smoothness assumption which is quite common for ordinary graphs. Secondly, RoGAT further revises the features to suppress features' noise. Then, an extra attention score is generated by the dynamic edge's weight and can be used to reduce the impact of adversarial attacks. Different experiments against targeted and untargeted attacks on citation data on citation data demonstrate that RoGAT outperforms most of the recent defensive methods.



## **11. Privacy Safe Representation Learning via Frequency Filtering Encoder**

cs.CV

The IJCAI-ECAI-22 Workshop on Artificial Intelligence Safety  (AISafety 2022)

**SubmitDate**: 2022-08-04    [paper-pdf](http://arxiv.org/pdf/2208.02482v1)

**Authors**: Jonghu Jeong, Minyong Cho, Philipp Benz, Jinwoo Hwang, Jeewook Kim, Seungkwan Lee, Tae-hoon Kim

**Abstracts**: Deep learning models are increasingly deployed in real-world applications. These models are often deployed on the server-side and receive user data in an information-rich representation to solve a specific task, such as image classification. Since images can contain sensitive information, which users might not be willing to share, privacy protection becomes increasingly important. Adversarial Representation Learning (ARL) is a common approach to train an encoder that runs on the client-side and obfuscates an image. It is assumed, that the obfuscated image can safely be transmitted and used for the task on the server without privacy concerns. However, in this work, we find that training a reconstruction attacker can successfully recover the original image of existing ARL methods. To this end, we introduce a novel ARL method enhanced through low-pass filtering, limiting the available information amount to be encoded in the frequency domain. Our experimental results reveal that our approach withstands reconstruction attacks while outperforming previous state-of-the-art methods regarding the privacy-utility trade-off. We further conduct a user study to qualitatively assess our defense of the reconstruction attack.



## **12. Node Copying: A Random Graph Model for Effective Graph Sampling**

stat.ML

**SubmitDate**: 2022-08-04    [paper-pdf](http://arxiv.org/pdf/2208.02435v1)

**Authors**: Florence Regol, Soumyasundar Pal, Jianing Sun, Yingxue Zhang, Yanhui Geng, Mark Coates

**Abstracts**: There has been an increased interest in applying machine learning techniques on relational structured-data based on an observed graph. Often, this graph is not fully representative of the true relationship amongst nodes. In these settings, building a generative model conditioned on the observed graph allows to take the graph uncertainty into account. Various existing techniques either rely on restrictive assumptions, fail to preserve topological properties within the samples or are prohibitively expensive for larger graphs. In this work, we introduce the node copying model for constructing a distribution over graphs. Sampling of a random graph is carried out by replacing each node's neighbors by those of a randomly sampled similar node. The sampled graphs preserve key characteristics of the graph structure without explicitly targeting them. Additionally, sampling from this model is extremely simple and scales linearly with the nodes. We show the usefulness of the copying model in three tasks. First, in node classification, a Bayesian formulation based on node copying achieves higher accuracy in sparse data settings. Second, we employ our proposed model to mitigate the effect of adversarial attacks on the graph topology. Last, incorporation of the model in a recommendation system setting improves recall over state-of-the-art methods.



## **13. Is current research on adversarial robustness addressing the right problem?**

cs.CV

**SubmitDate**: 2022-08-04    [paper-pdf](http://arxiv.org/pdf/2208.00539v2)

**Authors**: Ali Borji

**Abstracts**: Short answer: Yes, Long answer: No! Indeed, research on adversarial robustness has led to invaluable insights helping us understand and explore different aspects of the problem. Many attacks and defenses have been proposed over the last couple of years. The problem, however, remains largely unsolved and poorly understood. Here, I argue that the current formulation of the problem serves short term goals, and needs to be revised for us to achieve bigger gains. Specifically, the bound on perturbation has created a somewhat contrived setting and needs to be relaxed. This has misled us to focus on model classes that are not expressive enough to begin with. Instead, inspired by human vision and the fact that we rely more on robust features such as shape, vertices, and foreground objects than non-robust features such as texture, efforts should be steered towards looking for significantly different classes of models. Maybe instead of narrowing down on imperceptible adversarial perturbations, we should attack a more general problem which is finding architectures that are simultaneously robust to perceptible perturbations, geometric transformations (e.g. rotation, scaling), image distortions (lighting, blur), and more (e.g. occlusion, shadow). Only then we may be able to solve the problem of adversarial vulnerability.



## **14. A New Kind of Adversarial Example**

cs.CV

**SubmitDate**: 2022-08-04    [paper-pdf](http://arxiv.org/pdf/2208.02430v1)

**Authors**: Ali Borji

**Abstracts**: Almost all adversarial attacks are formulated to add an imperceptible perturbation to an image in order to fool a model. Here, we consider the opposite which is adversarial examples that can fool a human but not a model. A large enough and perceptible perturbation is added to an image such that a model maintains its original decision, whereas a human will most likely make a mistake if forced to decide (or opt not to decide at all). Existing targeted attacks can be reformulated to synthesize such adversarial examples. Our proposed attack, dubbed NKE, is similar in essence to the fooling images, but is more efficient since it uses gradient descent instead of evolutionary algorithms. It also offers a new and unified perspective into the problem of adversarial vulnerability. Experimental results over MNIST and CIFAR-10 datasets show that our attack is quite efficient in fooling deep neural networks. Code is available at https://github.com/aliborji/NKE.



## **15. MOVE: Effective and Harmless Ownership Verification via Embedded External Features**

cs.CR

15 pages. The journal extension of our conference paper in AAAI 2022  (https://ojs.aaai.org/index.php/AAAI/article/view/20036). arXiv admin note:  substantial text overlap with arXiv:2112.03476

**SubmitDate**: 2022-08-04    [paper-pdf](http://arxiv.org/pdf/2208.02820v1)

**Authors**: Yiming Li, Linghui Zhu, Xiaojun Jia, Yang Bai, Yong Jiang, Shu-Tao Xia, Xiaochun Cao

**Abstracts**: Currently, deep neural networks (DNNs) are widely adopted in different applications. Despite its commercial values, training a well-performed DNN is resource-consuming. Accordingly, the well-trained model is valuable intellectual property for its owner. However, recent studies revealed the threats of model stealing, where the adversaries can obtain a function-similar copy of the victim model, even when they can only query the model. In this paper, we propose an effective and harmless model ownership verification (MOVE) to defend against different types of model stealing simultaneously, without introducing new security risks. In general, we conduct the ownership verification by verifying whether a suspicious model contains the knowledge of defender-specified external features. Specifically, we embed the external features by tempering a few training samples with style transfer. We then train a meta-classifier to determine whether a model is stolen from the victim. This approach is inspired by the understanding that the stolen models should contain the knowledge of features learned by the victim model. In particular, we develop our MOVE method under both white-box and black-box settings to provide comprehensive model protection. Extensive experiments on benchmark datasets verify the effectiveness of our method and its resistance to potential adaptive attacks. The codes for reproducing the main experiments of our method are available at \url{https://github.com/THUYimingLi/MOVE}.



## **16. Deep VULMAN: A Deep Reinforcement Learning-Enabled Cyber Vulnerability Management Framework**

cs.AI

12 pages, 3 figures

**SubmitDate**: 2022-08-03    [paper-pdf](http://arxiv.org/pdf/2208.02369v1)

**Authors**: Soumyadeep Hore, Ankit Shah, Nathaniel D. Bastian

**Abstracts**: Cyber vulnerability management is a critical function of a cybersecurity operations center (CSOC) that helps protect organizations against cyber-attacks on their computer and network systems. Adversaries hold an asymmetric advantage over the CSOC, as the number of deficiencies in these systems is increasing at a significantly higher rate compared to the expansion rate of the security teams to mitigate them in a resource-constrained environment. The current approaches are deterministic and one-time decision-making methods, which do not consider future uncertainties when prioritizing and selecting vulnerabilities for mitigation. These approaches are also constrained by the sub-optimal distribution of resources, providing no flexibility to adjust their response to fluctuations in vulnerability arrivals. We propose a novel framework, Deep VULMAN, consisting of a deep reinforcement learning agent and an integer programming method to fill this gap in the cyber vulnerability management process. Our sequential decision-making framework, first, determines the near-optimal amount of resources to be allocated for mitigation under uncertainty for a given system state and then determines the optimal set of prioritized vulnerability instances for mitigation. Our proposed framework outperforms the current methods in prioritizing the selection of important organization-specific vulnerabilities, on both simulated and real-world vulnerability data, observed over a one-year period.



## **17. Membership Inference Attacks and Defenses in Neural Network Pruning**

cs.CR

This paper has been accepted to USENIX Security Symposium 2022. This  is an extended version with more experimental results

**SubmitDate**: 2022-08-03    [paper-pdf](http://arxiv.org/pdf/2202.03335v2)

**Authors**: Xiaoyong Yuan, Lan Zhang

**Abstracts**: Neural network pruning has been an essential technique to reduce the computation and memory requirements for using deep neural networks for resource-constrained devices. Most existing research focuses primarily on balancing the sparsity and accuracy of a pruned neural network by strategically removing insignificant parameters and retraining the pruned model. Such efforts on reusing training samples pose serious privacy risks due to increased memorization, which, however, has not been investigated yet.   In this paper, we conduct the first analysis of privacy risks in neural network pruning. Specifically, we investigate the impacts of neural network pruning on training data privacy, i.e., membership inference attacks. We first explore the impact of neural network pruning on prediction divergence, where the pruning process disproportionately affects the pruned model's behavior for members and non-members. Meanwhile, the influence of divergence even varies among different classes in a fine-grained manner. Enlighten by such divergence, we proposed a self-attention membership inference attack against the pruned neural networks. Extensive experiments are conducted to rigorously evaluate the privacy impacts of different pruning approaches, sparsity levels, and adversary knowledge. The proposed attack shows the higher attack performance on the pruned models when compared with eight existing membership inference attacks. In addition, we propose a new defense mechanism to protect the pruning process by mitigating the prediction divergence based on KL-divergence distance, whose effectiveness has been experimentally demonstrated to effectively mitigate the privacy risks while maintaining the sparsity and accuracy of the pruned models.



## **18. Design of secure and robust cognitive system for malware detection**

cs.CR

arXiv admin note: substantial text overlap with arXiv:2104.06652

**SubmitDate**: 2022-08-03    [paper-pdf](http://arxiv.org/pdf/2208.02310v1)

**Authors**: Sanket Shukla

**Abstracts**: Machine learning based malware detection techniques rely on grayscale images of malware and tends to classify malware based on the distribution of textures in graycale images. Albeit the advancement and promising results shown by machine learning techniques, attackers can exploit the vulnerabilities by generating adversarial samples. Adversarial samples are generated by intelligently crafting and adding perturbations to the input samples. There exists majority of the software based adversarial attacks and defenses. To defend against the adversaries, the existing malware detection based on machine learning and grayscale images needs a preprocessing for the adversarial data. This can cause an additional overhead and can prolong the real-time malware detection. So, as an alternative to this, we explore RRAM (Resistive Random Access Memory) based defense against adversaries. Therefore, the aim of this thesis is to address the above mentioned critical system security issues. The above mentioned challenges are addressed by demonstrating proposed techniques to design a secure and robust cognitive system. First, a novel technique to detect stealthy malware is proposed. The technique uses malware binary images and then extract different features from the same and then employ different ML-classifiers on the dataset thus obtained. Results demonstrate that this technique is successful in differentiating classes of malware based on the features extracted. Secondly, I demonstrate the effects of adversarial attacks on a reconfigurable RRAM-neuromorphic architecture with different learning algorithms and device characteristics. I also propose an integrated solution for mitigating the effects of the adversarial attack using the reconfigurable RRAM architecture.



## **19. Generating Image Adversarial Examples by Embedding Digital Watermarks**

cs.CV

10 pages, 4 figures

**SubmitDate**: 2022-08-03    [paper-pdf](http://arxiv.org/pdf/2009.05107v2)

**Authors**: Yuexin Xiang, Tiantian Li, Wei Ren, Tianqing Zhu, Kim-Kwang Raymond Choo

**Abstracts**: With the increasing attention to deep neural network (DNN) models, attacks are also upcoming for such models. For example, an attacker may carefully construct images in specific ways (also referred to as adversarial examples) aiming to mislead the DNN models to output incorrect classification results. Similarly, many efforts are proposed to detect and mitigate adversarial examples, usually for certain dedicated attacks. In this paper, we propose a novel digital watermark-based method to generate image adversarial examples to fool DNN models. Specifically, partial main features of the watermark image are embedded into the host image almost invisibly, aiming to tamper with and damage the recognition capabilities of the DNN models. We devise an efficient mechanism to select host images and watermark images and utilize the improved discrete wavelet transform (DWT) based Patchwork watermarking algorithm with a set of valid hyperparameters to embed digital watermarks from the watermark image dataset into original images for generating image adversarial examples. The experimental results illustrate that the attack success rate on common DNN models can reach an average of 95.47% on the CIFAR-10 dataset and the highest at 98.71%. Besides, our scheme is able to generate a large number of adversarial examples efficiently, concretely, an average of 1.17 seconds for completing the attacks on each image on the CIFAR-10 dataset. In addition, we design a baseline experiment using the watermark images generated by Gaussian noise as the watermark image dataset that also displays the effectiveness of our scheme. Similarly, we also propose the modified discrete cosine transform (DCT) based Patchwork watermarking algorithm. To ensure repeatability and reproducibility, the source code is available on GitHub.



## **20. Abusing Commodity DRAMs in IoT Devices to Remotely Spy on Temperature**

cs.CR

Submitted to IEEE TIFS and currently under review

**SubmitDate**: 2022-08-03    [paper-pdf](http://arxiv.org/pdf/2208.02125v1)

**Authors**: Florian Frank, Wenjie Xiong, Nikolaos Athanasios Anagnostopoulos, André Schaller, Tolga Arul, Farinaz Koushanfar, Stefan Katzenbeisser, Ulrich Ruhrmair, Jakub Szefer

**Abstracts**: The ubiquity and pervasiveness of modern Internet of Things (IoT) devices opens up vast possibilities for novel applications, but simultaneously also allows spying on, and collecting data from, unsuspecting users to a previously unseen extent. This paper details a new attack form in this vein, in which the decay properties of widespread, off-the-shelf DRAM modules are exploited to accurately sense the temperature in the vicinity of the DRAM-carrying device. Among others, this enables adversaries to remotely and purely digitally spy on personal behavior in users' private homes, or to collect security-critical data in server farms, cloud storage centers, or commercial production lines. We demonstrate that our attack can be performed by merely compromising the software of an IoT device and does not require hardware modifications or physical access at attack time. It can achieve temperature resolutions of up to 0.5{\deg}C over a range of 0{\deg}C to 70{\deg}C in practice. Perhaps most interestingly, it even works in devices that do not have a dedicated temperature sensor on board. To complete our work, we discuss practical attack scenarios as well as possible countermeasures against our temperature espionage attacks.



## **21. Local Differential Privacy for Federated Learning**

cs.CR

17 pages

**SubmitDate**: 2022-08-03    [paper-pdf](http://arxiv.org/pdf/2202.06053v2)

**Authors**: M. A. P. Chamikara, Dongxi Liu, Seyit Camtepe, Surya Nepal, Marthie Grobler, Peter Bertok, Ibrahim Khalil

**Abstracts**: Advanced adversarial attacks such as membership inference and model memorization can make federated learning (FL) vulnerable and potentially leak sensitive private data. Local differentially private (LDP) approaches are gaining more popularity due to stronger privacy notions and native support for data distribution compared to other differentially private (DP) solutions. However, DP approaches assume that the FL server (that aggregates the models) is honest (run the FL protocol honestly) or semi-honest (run the FL protocol honestly while also trying to learn as much information as possible). These assumptions make such approaches unrealistic and unreliable for real-world settings. Besides, in real-world industrial environments (e.g., healthcare), the distributed entities (e.g., hospitals) are already composed of locally running machine learning models (this setting is also referred to as the cross-silo setting). Existing approaches do not provide a scalable mechanism for privacy-preserving FL to be utilized under such settings, potentially with untrusted parties. This paper proposes a new local differentially private FL (named LDPFL) protocol for industrial settings. LDPFL can run in industrial settings with untrusted entities while enforcing stronger privacy guarantees than existing approaches. LDPFL shows high FL model performance (up to 98%) under small privacy budgets (e.g., epsilon = 0.5) in comparison to existing methods.



## **22. SAC-AP: Soft Actor Critic based Deep Reinforcement Learning for Alert Prioritization**

cs.CR

8 pages, 8 figures, IEEE WORLD CONGRESS ON COMPUTATIONAL INTELLIGENCE  2022

**SubmitDate**: 2022-08-03    [paper-pdf](http://arxiv.org/pdf/2207.13666v3)

**Authors**: Lalitha Chavali, Tanay Gupta, Paresh Saxena

**Abstracts**: Intrusion detection systems (IDS) generate a large number of false alerts which makes it difficult to inspect true positives. Hence, alert prioritization plays a crucial role in deciding which alerts to investigate from an enormous number of alerts that are generated by IDS. Recently, deep reinforcement learning (DRL) based deep deterministic policy gradient (DDPG) off-policy method has shown to achieve better results for alert prioritization as compared to other state-of-the-art methods. However, DDPG is prone to the problem of overfitting. Additionally, it also has a poor exploration capability and hence it is not suitable for problems with a stochastic environment. To address these limitations, we present a soft actor-critic based DRL algorithm for alert prioritization (SAC-AP), an off-policy method, based on the maximum entropy reinforcement learning framework that aims to maximize the expected reward while also maximizing the entropy. Further, the interaction between an adversary and a defender is modeled as a zero-sum game and a double oracle framework is utilized to obtain the approximate mixed strategy Nash equilibrium (MSNE). SAC-AP finds robust alert investigation policies and computes pure strategy best response against opponent's mixed strategy. We present the overall design of SAC-AP and evaluate its performance as compared to other state-of-the art alert prioritization methods. We consider defender's loss, i.e., the defender's inability to investigate the alerts that are triggered due to attacks, as the performance metric. Our results show that SAC-AP achieves up to 30% decrease in defender's loss as compared to the DDPG based alert prioritization method and hence provides better protection against intrusions. Moreover, the benefits are even higher when SAC-AP is compared to other traditional alert prioritization methods including Uniform, GAIN, RIO and Suricata.



## **23. Spectrum Focused Frequency Adversarial Attacks for Automatic Modulation Classification**

cs.CR

6 pages, 9 figures

**SubmitDate**: 2022-08-03    [paper-pdf](http://arxiv.org/pdf/2208.01919v1)

**Authors**: Sicheng Zhang, Jiarun Yu, Zhida Bao, Shiwen Mao, Yun Lin

**Abstracts**: Artificial intelligence (AI) technology has provided a potential solution for automatic modulation recognition (AMC). Unfortunately, AI-based AMC models are vulnerable to adversarial examples, which seriously threatens the efficient, secure and trusted application of AI in AMC. This issue has attracted the attention of researchers. Various studies on adversarial attacks and defenses evolve in a spiral. However, the existing adversarial attack methods are all designed in the time domain. They introduce more high-frequency components in the frequency domain, due to abrupt updates in the time domain. For this issue, from the perspective of frequency domain, we propose a spectrum focused frequency adversarial attacks (SFFAA) for AMC model, and further draw on the idea of meta-learning, propose a Meta-SFFAA algorithm to improve the transferability in the black-box attacks. Extensive experiments, qualitative and quantitative metrics demonstrate that the proposed algorithm can concentrate the adversarial energy on the spectrum where the signal is located, significantly improve the adversarial attack performance while maintaining the concealment in the frequency domain.



## **24. On the Evaluation of User Privacy in Deep Neural Networks using Timing Side Channel**

cs.CR

15 pages, 20 figures

**SubmitDate**: 2022-08-03    [paper-pdf](http://arxiv.org/pdf/2208.01113v2)

**Authors**: Shubhi Shukla, Manaar Alam, Sarani Bhattacharya, Debdeep Mukhopadhyay, Pabitra Mitra

**Abstracts**: Recent Deep Learning (DL) advancements in solving complex real-world tasks have led to its widespread adoption in practical applications. However, this opportunity comes with significant underlying risks, as many of these models rely on privacy-sensitive data for training in a variety of applications, making them an overly-exposed threat surface for privacy violations. Furthermore, the widespread use of cloud-based Machine-Learning-as-a-Service (MLaaS) for its robust infrastructure support has broadened the threat surface to include a variety of remote side-channel attacks. In this paper, we first identify and report a novel data-dependent timing side-channel leakage (termed Class Leakage) in DL implementations originating from non-constant time branching operation in a widely used DL framework PyTorch. We further demonstrate a practical inference-time attack where an adversary with user privilege and hard-label black-box access to an MLaaS can exploit Class Leakage to compromise the privacy of MLaaS users. DL models are vulnerable to Membership Inference Attack (MIA), where an adversary's objective is to deduce whether any particular data has been used while training the model. In this paper, as a separate case study, we demonstrate that a DL model secured with differential privacy (a popular countermeasure against MIA) is still vulnerable to MIA against an adversary exploiting Class Leakage. We develop an easy-to-implement countermeasure by making a constant-time branching operation that alleviates the Class Leakage and also aids in mitigating MIA. We have chosen two standard benchmarking image classification datasets, CIFAR-10 and CIFAR-100 to train five state-of-the-art pre-trained DL models, over two different computing environments having Intel Xeon and Intel i7 processors to validate our approach.



## **25. Adversarial Attacks on ASR Systems: An Overview**

cs.SD

**SubmitDate**: 2022-08-03    [paper-pdf](http://arxiv.org/pdf/2208.02250v1)

**Authors**: Xiao Zhang, Hao Tan, Xuan Huang, Denghui Zhang, Keke Tang, Zhaoquan Gu

**Abstracts**: With the development of hardware and algorithms, ASR(Automatic Speech Recognition) systems evolve a lot. As The models get simpler, the difficulty of development and deployment become easier, ASR systems are getting closer to our life. On the one hand, we often use APPs or APIs of ASR to generate subtitles and record meetings. On the other hand, smart speaker and self-driving car rely on ASR systems to control AIoT devices. In past few years, there are a lot of works on adversarial examples attacks against ASR systems. By adding a small perturbation to the waveforms, the recognition results make a big difference. In this paper, we describe the development of ASR system, different assumptions of attacks, and how to evaluate these attacks. Next, we introduce the current works on adversarial examples attacks from two attack assumptions: white-box attack and black-box attack. Different from other surveys, we pay more attention to which layer they perturb waveforms in ASR system, the relationship between these attacks, and their implementation methods. We focus on the effect of their works.



## **26. Robust Graph Neural Networks using Weighted Graph Laplacian**

cs.LG

Accepted at IEEE International Conference on Signal Processing and  Communications (SPCOM), 2022

**SubmitDate**: 2022-08-03    [paper-pdf](http://arxiv.org/pdf/2208.01853v1)

**Authors**: Bharat Runwal, Vivek, Sandeep Kumar

**Abstracts**: Graph neural network (GNN) is achieving remarkable performances in a variety of application domains. However, GNN is vulnerable to noise and adversarial attacks in input data. Making GNN robust against noises and adversarial attacks is an important problem. The existing defense methods for GNNs are computationally demanding and are not scalable. In this paper, we propose a generic framework for robustifying GNN known as Weighted Laplacian GNN (RWL-GNN). The method combines Weighted Graph Laplacian learning with the GNN implementation. The proposed method benefits from the positive semi-definiteness property of Laplacian matrix, feature smoothness, and latent features via formulating a unified optimization framework, which ensures the adversarial/noisy edges are discarded and connections in the graph are appropriately weighted. For demonstration, the experiments are conducted with Graph convolutional neural network(GCNN) architecture, however, the proposed framework is easily amenable to any existing GNN architecture. The simulation results with benchmark dataset establish the efficacy of the proposed method, both in accuracy and computational efficiency. Code can be accessed at https://github.com/Bharat-Runwal/RWL-GNN.



## **27. Multiclass ASMA vs Targeted PGD Attack in Image Segmentation**

cs.CV

10 pages, 6 figures

**SubmitDate**: 2022-08-03    [paper-pdf](http://arxiv.org/pdf/2208.01844v1)

**Authors**: Johnson Vo, Jiabao Xie, Sahil Patel

**Abstracts**: Deep learning networks have demonstrated high performance in a large variety of applications, such as image classification, speech recognition, and natural language processing. However, there exists a major vulnerability exploited by the use of adversarial attacks. An adversarial attack imputes images by altering the input image very slightly, making it nearly undetectable to the naked eye, but results in a very different classification by the network. This paper explores the projected gradient descent (PGD) attack and the Adaptive Mask Segmentation Attack (ASMA) on the image segmentation DeepLabV3 model using two types of architectures: MobileNetV3 and ResNet50, It was found that PGD was very consistent in changing the segmentation to be its target while the generalization of ASMA to a multiclass target was not as effective. The existence of such attack however puts all of image classification deep learning networks in danger of exploitation.



## **28. Adversarial Camouflage for Node Injection Attack on Graphs**

cs.LG

**SubmitDate**: 2022-08-03    [paper-pdf](http://arxiv.org/pdf/2208.01819v1)

**Authors**: Shuchang Tao, Qi Cao, Huawei Shen, Yunfan Wu, Liang Hou, Xueqi Cheng

**Abstracts**: Node injection attacks against Graph Neural Networks (GNNs) have received emerging attention as a practical attack scenario, where the attacker injects malicious nodes instead of modifying node features or edges to degrade the performance of GNNs. Despite the initial success of node injection attacks, we find that the injected nodes by existing methods are easy to be distinguished from the original normal nodes by defense methods and limiting their attack performance in practice. To solve the above issues, we devote to camouflage node injection attack, i.e., camouflaging injected malicious nodes (structure/attributes) as the normal ones that appear legitimate/imperceptible to defense methods. The non-Euclidean nature of graph data and the lack of human prior brings great challenges to the formalization, implementation, and evaluation of camouflage on graphs. In this paper, we first propose and formulate the camouflage of injected nodes from both the fidelity and diversity of the ego networks centered around injected nodes. Then, we design an adversarial CAmouflage framework for Node injection Attack, namely CANA, to improve the camouflage while ensuring the attack performance. Several novel indicators for graph camouflage are further designed for a comprehensive evaluation. Experimental results demonstrate that when equipping existing node injection attack methods with our proposed CANA framework, the attack performance against defense methods as well as node camouflage is significantly improved.



## **29. Success of Uncertainty-Aware Deep Models Depends on Data Manifold Geometry**

cs.LG

9 pages

**SubmitDate**: 2022-08-02    [paper-pdf](http://arxiv.org/pdf/2208.01705v1)

**Authors**: Mark Penrod, Harrison Termotto, Varshini Reddy, Jiayu Yao, Finale Doshi-Velez, Weiwei Pan

**Abstracts**: For responsible decision making in safety-critical settings, machine learning models must effectively detect and process edge-case data. Although existing works show that predictive uncertainty is useful for these tasks, it is not evident from literature which uncertainty-aware models are best suited for a given dataset. Thus, we compare six uncertainty-aware deep learning models on a set of edge-case tasks: robustness to adversarial attacks as well as out-of-distribution and adversarial detection. We find that the geometry of the data sub-manifold is an important factor in determining the success of various models. Our finding suggests an interesting direction in the study of uncertainty-aware deep learning models.



## **30. CAPD: A Context-Aware, Policy-Driven Framework for Secure and Resilient IoBT Operations**

cs.CR

**SubmitDate**: 2022-08-02    [paper-pdf](http://arxiv.org/pdf/2208.01703v1)

**Authors**: Sai Sree Laya Chukkapalli, Anupam Joshi, Tim Finin, Robert F. Erbacher

**Abstracts**: The Internet of Battlefield Things (IoBT) will advance the operational effectiveness of infantry units. However, this requires autonomous assets such as sensors, drones, combat equipment, and uncrewed vehicles to collaborate, securely share information, and be resilient to adversary attacks in contested multi-domain operations. CAPD addresses this problem by providing a context-aware, policy-driven framework supporting data and knowledge exchange among autonomous entities in a battlespace. We propose an IoBT ontology that facilitates controlled information sharing to enable semantic interoperability between systems. Its key contributions include providing a knowledge graph with a shared semantic schema, integration with background knowledge, efficient mechanisms for enforcing data consistency and drawing inferences, and supporting attribute-based access control. The sensors in the IoBT provide data that create populated knowledge graphs based on the ontology. This paper describes using CAPD to detect and mitigate adversary actions. CAPD enables situational awareness using reasoning over the sensed data and SPARQL queries. For example, adversaries can cause sensor failure or hijacking and disrupt the tactical networks to degrade video surveillance. In such instances, CAPD uses an ontology-based reasoner to see how alternative approaches can still support the mission. Depending on bandwidth availability, the reasoner initiates the creation of a reduced frame rate grayscale video by active transcoding or transmits only still images. This ability to reason over the mission sensed environment and attack context permits the autonomous IoBT system to exhibit resilience in contested conditions.



## **31. Adversarial Detection Avoidance Attacks: Evaluating the robustness of perceptual hashing-based client-side scanning**

cs.CR

This is a revised version of the paper published at USENIX Security  2022. We now use a semi-automated procedure to remove duplicates from the  ImageNet dataset

**SubmitDate**: 2022-08-02    [paper-pdf](http://arxiv.org/pdf/2106.09820v3)

**Authors**: Shubham Jain, Ana-Maria Cretu, Yves-Alexandre de Montjoye

**Abstracts**: End-to-end encryption (E2EE) by messaging platforms enable people to securely and privately communicate with one another. Its widespread adoption however raised concerns that illegal content might now be shared undetected. Following the global pushback against key escrow systems, client-side scanning based on perceptual hashing has been recently proposed by tech companies, governments and researchers to detect illegal content in E2EE communications. We here propose the first framework to evaluate the robustness of perceptual hashing-based client-side scanning to detection avoidance attacks and show current systems to not be robust. More specifically, we propose three adversarial attacks--a general black-box attack and two white-box attacks for discrete cosine transform-based algorithms--against perceptual hashing algorithms. In a large-scale evaluation, we show perceptual hashing-based client-side scanning mechanisms to be highly vulnerable to detection avoidance attacks in a black-box setting, with more than 99.9% of images successfully attacked while preserving the content of the image. We furthermore show our attack to generate diverse perturbations, strongly suggesting that straightforward mitigation strategies would be ineffective. Finally, we show that the larger thresholds necessary to make the attack harder would probably require more than one billion images to be flagged and decrypted daily, raising strong privacy concerns. Taken together, our results shed serious doubts on the robustness of perceptual hashing-based client-side scanning mechanisms currently proposed by governments, organizations, and researchers around the world.



## **32. Quantum Lock: A Provable Quantum Communication Advantage**

quant-ph

Replacement of paper "Hybrid PUF: A Novel Way to Enhance the Security  of Classical PUFs" (arXiv:2110.09469)

**SubmitDate**: 2022-08-02    [paper-pdf](http://arxiv.org/pdf/2110.09469v3)

**Authors**: Kaushik Chakraborty, Mina Doosti, Yao Ma, Chirag Wadhwa, Myrto Arapinis, Elham Kashefi

**Abstracts**: Physical unclonable functions(PUFs) provide a unique fingerprint to a physical entity by exploiting the inherent physical randomness. Gao et al. discussed the vulnerability of most current-day PUFs to sophisticated machine learning-based attacks. We address this problem by integrating classical PUFs and existing quantum communication technology. Specifically, this paper proposes a generic design of provably secure PUFs, called hybrid locked PUFs(HLPUFs), providing a practical solution for securing classical PUFs. An HLPUF uses a classical PUF(CPUF), and encodes the output into non-orthogonal quantum states to hide the outcomes of the underlying CPUF from any adversary. Here we introduce a quantum lock to protect the HLPUFs from any general adversaries. The indistinguishability property of the non-orthogonal quantum states, together with the quantum lockdown technique prevents the adversary from accessing the outcome of the CPUFs. Moreover, we show that by exploiting non-classical properties of quantum states, the HLPUF allows the server to reuse the challenge-response pairs for further client authentication. This result provides an efficient solution for running PUF-based client authentication for an extended period while maintaining a small-sized challenge-response pairs database on the server side. Later, we support our theoretical contributions by instantiating the HLPUFs design using accessible real-world CPUFs. We use the optimal classical machine-learning attacks to forge both the CPUFs and HLPUFs, and we certify the security gap in our numerical simulation for construction which is ready for implementation.



## **33. SCFI: State Machine Control-Flow Hardening Against Fault Attacks**

cs.CR

**SubmitDate**: 2022-08-02    [paper-pdf](http://arxiv.org/pdf/2208.01356v1)

**Authors**: Pascal Nasahl, Martin Unterguggenberger, Rishub Nagpal, Robert Schilling, David Schrammel, Stefan Mangard

**Abstracts**: Fault injection (FI) is a powerful attack methodology allowing an adversary to entirely break the security of a target device. As finite-state machines (FSMs) are fundamental hardware building blocks responsible for controlling systems, inducing faults into these controllers enables an adversary to hijack the execution of the integrated circuit. A common defense strategy mitigating these attacks is to manually instantiate FSMs multiple times and detect faults using a majority voting logic. However, as each additional FSM instance only provides security against one additional induced fault, this approach scales poorly in a multi-fault attack scenario.   In this paper, we present SCFI: a strong, probabilistic FSM protection mechanism ensuring that control-flow deviations from the intended control-flow are detected even in the presence of multiple faults. At its core, SCFI consists of a hardened next-state function absorbing the execution history as well as the FSM's control signals to derive the next state. When either the absorbed inputs, the state registers, or the function itself are affected by faults, SCFI triggers an error with no detection latency. We integrate SCFI into a synthesis tool capable of automatically hardening arbitrary unprotected FSMs without user interaction and open-source the tool. Our evaluation shows that SCFI provides strong protection guarantees with a better area-time product than FSMs protected using classical redundancy-based approaches. Finally, we formally verify the resilience of the protected state machines using a pre-silicon fault analysis tool.



## **34. Understanding Adversarial Robustness of Vision Transformers via Cauchy Problem**

cs.CV

Accepted by ECML-PKDD 2022

**SubmitDate**: 2022-08-01    [paper-pdf](http://arxiv.org/pdf/2208.00906v1)

**Authors**: Zheng Wang, Wenjie Ruan

**Abstracts**: Recent research on the robustness of deep learning has shown that Vision Transformers (ViTs) surpass the Convolutional Neural Networks (CNNs) under some perturbations, e.g., natural corruption, adversarial attacks, etc. Some papers argue that the superior robustness of ViT comes from the segmentation of its input images; others say that the Multi-head Self-Attention (MSA) is the key to preserving the robustness. In this paper, we aim to introduce a principled and unified theoretical framework to investigate such an argument on ViT's robustness. We first theoretically prove that, unlike Transformers in Natural Language Processing, ViTs are Lipschitz continuous. Then we theoretically analyze the adversarial robustness of ViTs from the perspective of the Cauchy Problem, via which we can quantify how the robustness propagates through layers. We demonstrate that the first and last layers are the critical factors to affect the robustness of ViTs. Furthermore, based on our theory, we empirically show that unlike the claims from existing research, MSA only contributes to the adversarial robustness of ViTs under weak adversarial attacks, e.g., FGSM, and surprisingly, MSA actually comprises the model's adversarial robustness under stronger attacks, e.g., PGD attacks.



## **35. On the Detection of Adaptive Adversarial Attacks in Speaker Verification Systems**

cs.CR

**SubmitDate**: 2022-08-01    [paper-pdf](http://arxiv.org/pdf/2202.05725v2)

**Authors**: Zesheng Chen

**Abstracts**: Speaker verification systems have been widely used in smart phones and Internet of things devices to identify legitimate users. In recent work, it has been shown that adversarial attacks, such as FAKEBOB, can work effectively against speaker verification systems. The goal of this paper is to design a detector that can distinguish an original audio from an audio contaminated by adversarial attacks. Specifically, our designed detector, called MEH-FEST, calculates the minimum energy in high frequencies from the short-time Fourier transform of an audio and uses it as a detection metric. Through both analysis and experiments, we show that our proposed detector is easy to implement, fast to process an input audio, and effective in determining whether an audio is corrupted by FAKEBOB attacks. The experimental results indicate that the detector is extremely effective: with near zero false positive and false negative rates for detecting FAKEBOB attacks in Gaussian mixture model (GMM) and i-vector speaker verification systems. Moreover, adaptive adversarial attacks against our proposed detector and their countermeasures are discussed and studied, showing the game between attackers and defenders.



## **36. The Geometry of Adversarial Training in Binary Classification**

cs.LG

**SubmitDate**: 2022-08-01    [paper-pdf](http://arxiv.org/pdf/2111.13613v2)

**Authors**: Leon Bungert, Nicolás García Trillos, Ryan Murray

**Abstracts**: We establish an equivalence between a family of adversarial training problems for non-parametric binary classification and a family of regularized risk minimization problems where the regularizer is a nonlocal perimeter functional. The resulting regularized risk minimization problems admit exact convex relaxations of the type $L^1+$ (nonlocal) $\operatorname{TV}$, a form frequently studied in image analysis and graph-based learning. A rich geometric structure is revealed by this reformulation which in turn allows us to establish a series of properties of optimal solutions of the original problem, including the existence of minimal and maximal solutions (interpreted in a suitable sense), and the existence of regular solutions (also interpreted in a suitable sense). In addition, we highlight how the connection between adversarial training and perimeter minimization problems provides a novel, directly interpretable, statistical motivation for a family of regularized risk minimization problems involving perimeter/total variation. The majority of our theoretical results are independent of the distance used to define adversarial attacks.



## **37. DNNShield: Dynamic Randomized Model Sparsification, A Defense Against Adversarial Machine Learning**

cs.CR

**SubmitDate**: 2022-07-31    [paper-pdf](http://arxiv.org/pdf/2208.00498v1)

**Authors**: Mohammad Hossein Samavatian, Saikat Majumdar, Kristin Barber, Radu Teodorescu

**Abstracts**: DNNs are known to be vulnerable to so-called adversarial attacks that manipulate inputs to cause incorrect results that can be beneficial to an attacker or damaging to the victim. Recent works have proposed approximate computation as a defense mechanism against machine learning attacks. We show that these approaches, while successful for a range of inputs, are insufficient to address stronger, high-confidence adversarial attacks. To address this, we propose DNNSHIELD, a hardware-accelerated defense that adapts the strength of the response to the confidence of the adversarial input. Our approach relies on dynamic and random sparsification of the DNN model to achieve inference approximation efficiently and with fine-grain control over the approximation error. DNNSHIELD uses the output distribution characteristics of sparsified inference compared to a dense reference to detect adversarial inputs. We show an adversarial detection rate of 86% when applied to VGG16 and 88% when applied to ResNet50, which exceeds the detection rate of the state of the art approaches, with a much lower overhead. We demonstrate a software/hardware-accelerated FPGA prototype, which reduces the performance impact of DNNSHIELD relative to software-only CPU and GPU implementations.



## **38. Adversarial Robustness Verification and Attack Synthesis in Stochastic Systems**

cs.CR

To Appear, 35th IEEE Computer Security Foundations Symposium (2022)

**SubmitDate**: 2022-07-31    [paper-pdf](http://arxiv.org/pdf/2110.02125v2)

**Authors**: Lisa Oakley, Alina Oprea, Stavros Tripakis

**Abstracts**: Probabilistic model checking is a useful technique for specifying and verifying properties of stochastic systems including randomized protocols and reinforcement learning models. Existing methods rely on the assumed structure and probabilities of certain system transitions. These assumptions may be incorrect, and may even be violated by an adversary who gains control of system components.   In this paper, we develop a formal framework for adversarial robustness in systems modeled as discrete time Markov chains (DTMCs). We base our framework on existing methods for verifying probabilistic temporal logic properties and extend it to include deterministic, memoryless policies acting in Markov decision processes (MDPs). Our framework includes a flexible approach for specifying structure-preserving and non structure-preserving adversarial models. We outline a class of threat models under which adversaries can perturb system transitions, constrained by an $\varepsilon$ ball around the original transition probabilities.   We define three main DTMC adversarial robustness problems: adversarial robustness verification, maximal $\delta$ synthesis, and worst case attack synthesis. We present two optimization-based solutions to these three problems, leveraging traditional and parametric probabilistic model checking techniques. We then evaluate our solutions on two stochastic protocols and a collection of Grid World case studies, which model an agent acting in an environment described as an MDP. We find that the parametric solution results in fast computation for small parameter spaces. In the case of less restrictive (stronger) adversaries, the number of parameters increases, and directly computing property satisfaction probabilities is more scalable. We demonstrate the usefulness of our definitions and solutions by comparing system outcomes over various properties, threat models, and case studies.



## **39. Robust Real-World Image Super-Resolution against Adversarial Attacks**

cs.CV

ACM-MM 2021, Code:  https://github.com/lhaof/Robust-SR-against-Adversarial-Attacks

**SubmitDate**: 2022-07-31    [paper-pdf](http://arxiv.org/pdf/2208.00428v1)

**Authors**: Jiutao Yue, Haofeng Li, Pengxu Wei, Guanbin Li, Liang Lin

**Abstracts**: Recently deep neural networks (DNNs) have achieved significant success in real-world image super-resolution (SR). However, adversarial image samples with quasi-imperceptible noises could threaten deep learning SR models. In this paper, we propose a robust deep learning framework for real-world SR that randomly erases potential adversarial noises in the frequency domain of input images or features. The rationale is that on the SR task clean images or features have a different pattern from the attacked ones in the frequency domain. Observing that existing adversarial attacks usually add high-frequency noises to input images, we introduce a novel random frequency mask module that blocks out high-frequency components possibly containing the harmful perturbations in a stochastic manner. Since the frequency masking may not only destroys the adversarial perturbations but also affects the sharp details in a clean image, we further develop an adversarial sample classifier based on the frequency domain of images to determine if applying the proposed mask module. Based on the above ideas, we devise a novel real-world image SR framework that combines the proposed frequency mask modules and the proposed adversarial classifier with an existing super-resolution backbone network. Experiments show that our proposed method is more insensitive to adversarial attacks and presents more stable SR results than existing models and defenses.



## **40. Electromagnetic Signal Injection Attacks on Differential Signaling**

cs.CR

14 pages, 15 figures

**SubmitDate**: 2022-07-31    [paper-pdf](http://arxiv.org/pdf/2208.00343v1)

**Authors**: Youqian Zhang, Kasper Rasmussen

**Abstracts**: Differential signaling is a method of data transmission that uses two complementary electrical signals to encode information. This allows a receiver to reject any noise by looking at the difference between the two signals, assuming the noise affects both signals in the same way. Many protocols such as USB, Ethernet, and HDMI use differential signaling to achieve a robust communication channel in a noisy environment. This generally works well and has led many to believe that it is infeasible to remotely inject attacking signals into such a differential pair. In this paper we challenge this assumption and show that an adversary can in fact inject malicious signals from a distance, purely using common-mode injection, i.e., injecting into both wires at the same time. We show how this allows an attacker to inject bits or even arbitrary messages into a communication line. Such an attack is a significant threat to many applications, from home security and privacy to automotive systems, critical infrastructure, or implantable medical devices; in which incorrect data or unauthorized control could cause significant damage, or even fatal accidents.   We show in detail the principles of how an electromagnetic signal can bypass the noise rejection of differential signaling, and eventually result in incorrect bits in the receiver. We show how an attacker can exploit this to achieve a successful injection of an arbitrary bit, and we analyze the success rate of injecting longer arbitrary messages. We demonstrate the attack on a real system and show that the success rate can reach as high as $90\%$. Finally, we present a case study where we wirelessly inject a message into a Controller Area Network (CAN) bus, which is a differential signaling bus protocol used in many critical applications, including the automotive and aviation sector.



## **41. Backdoor Attack is a Devil in Federated GAN-based Medical Image Synthesis**

cs.CV

13 pages, 4 figures, Accepted by MICCAI 2022 SASHIMI Workshop

**SubmitDate**: 2022-07-30    [paper-pdf](http://arxiv.org/pdf/2207.00762v2)

**Authors**: Ruinan Jin, Xiaoxiao Li

**Abstracts**: Deep Learning-based image synthesis techniques have been applied in healthcare research for generating medical images to support open research. Training generative adversarial neural networks (GAN) usually requires large amounts of training data. Federated learning (FL) provides a way of training a central model using distributed data from different medical institutions while keeping raw data locally. However, FL is vulnerable to backdoor attack, an adversarial by poisoning training data, given the central server cannot access the original data directly. Most backdoor attack strategies focus on classification models and centralized domains. In this study, we propose a way of attacking federated GAN (FedGAN) by treating the discriminator with a commonly used data poisoning strategy in backdoor attack classification models. We demonstrate that adding a small trigger with size less than 0.5 percent of the original image size can corrupt the FL-GAN model. Based on the proposed attack, we provide two effective defense strategies: global malicious detection and local training regularization. We show that combining the two defense strategies yields a robust medical image generation.



## **42. Towards Privacy-Preserving, Real-Time and Lossless Feature Matching**

cs.CV

**SubmitDate**: 2022-07-30    [paper-pdf](http://arxiv.org/pdf/2208.00214v1)

**Authors**: Qiang Meng, Feng Zhou

**Abstracts**: Most visual retrieval applications store feature vectors for downstream matching tasks. These vectors, from where user information can be spied out, will cause privacy leakage if not carefully protected. To mitigate privacy risks, current works primarily utilize non-invertible transformations or fully cryptographic algorithms. However, transformation-based methods usually fail to achieve satisfying matching performances while cryptosystems suffer from heavy computational overheads. In addition, secure levels of current methods should be improved to confront potential adversary attacks. To address these issues, this paper proposes a plug-in module called SecureVector that protects features by random permutations, 4L-DEC converting and existing homomorphic encryption techniques. For the first time, SecureVector achieves real-time and lossless feature matching among sanitized features, along with much higher security levels than current state-of-the-arts. Extensive experiments on face recognition, person re-identification, image retrieval, and privacy analyses demonstrate the effectiveness of our method. Given limited public projects in this field, codes of our method and implemented baselines are made open-source in https://github.com/IrvingMeng/SecureVector.



## **43. Towards Bridging the gap between Empirical and Certified Robustness against Adversarial Examples**

cs.LG

An abridged version of this work has been presented at ICLR 2021  Workshop on Security and Safety in Machine Learning Systems:  https://aisecure-workshop.github.io/aml-iclr2021/papers/2.pdf

**SubmitDate**: 2022-07-30    [paper-pdf](http://arxiv.org/pdf/2102.05096v3)

**Authors**: Jay Nandy, Sudipan Saha, Wynne Hsu, Mong Li Lee, Xiao Xiang Zhu

**Abstracts**: The current state-of-the-art defense methods against adversarial examples typically focus on improving either empirical or certified robustness. Among them, adversarially trained (AT) models produce empirical state-of-the-art defense against adversarial examples without providing any robustness guarantees for large classifiers or higher-dimensional inputs. In contrast, existing randomized smoothing based models achieve state-of-the-art certified robustness while significantly degrading the empirical robustness against adversarial examples. In this paper, we propose a novel method, called \emph{Certification through Adaptation}, that transforms an AT model into a randomized smoothing classifier during inference to provide certified robustness for $\ell_2$ norm without affecting their empirical robustness against adversarial attacks. We also propose \emph{Auto-Noise} technique that efficiently approximates the appropriate noise levels to flexibly certify the test examples using randomized smoothing technique. Our proposed \emph{Certification through Adaptation} with \emph{Auto-Noise} technique achieves an \textit{average certified radius (ACR) scores} up to $1.102$ and $1.148$ respectively for CIFAR-10 and ImageNet datasets using AT models without affecting their empirical robustness or benign accuracy. Therefore, our paper is a step towards bridging the gap between the empirical and certified robustness against adversarial examples by achieving both using the same classifier.



## **44. Robust Trajectory Prediction against Adversarial Attacks**

cs.LG

**SubmitDate**: 2022-07-29    [paper-pdf](http://arxiv.org/pdf/2208.00094v1)

**Authors**: Yulong Cao, Danfei Xu, Xinshuo Weng, Zhuoqing Mao, Anima Anandkumar, Chaowei Xiao, Marco Pavone

**Abstracts**: Trajectory prediction using deep neural networks (DNNs) is an essential component of autonomous driving (AD) systems. However, these methods are vulnerable to adversarial attacks, leading to serious consequences such as collisions. In this work, we identify two key ingredients to defend trajectory prediction models against adversarial attacks including (1) designing effective adversarial training methods and (2) adding domain-specific data augmentation to mitigate the performance degradation on clean data. We demonstrate that our method is able to improve the performance by 46% on adversarial data and at the cost of only 3% performance degradation on clean data, compared to the model trained with clean data. Additionally, compared to existing robust methods, our method can improve performance by 21% on adversarial examples and 9% on clean data. Our robust model is evaluated with a planner to study its downstream impacts. We demonstrate that our model can significantly reduce the severe accident rates (e.g., collisions and off-road driving).



## **45. Sampling Attacks on Meta Reinforcement Learning: A Minimax Formulation and Complexity Analysis**

cs.LG

**SubmitDate**: 2022-07-29    [paper-pdf](http://arxiv.org/pdf/2208.00081v1)

**Authors**: Tao Li, Haozhe Lei, Quanyan Zhu

**Abstracts**: Meta reinforcement learning (meta RL), as a combination of meta-learning ideas and reinforcement learning (RL), enables the agent to adapt to different tasks using a few samples. However, this sampling-based adaptation also makes meta RL vulnerable to adversarial attacks. By manipulating the reward feedback from sampling processes in meta RL, an attacker can mislead the agent into building wrong knowledge from training experience, which deteriorates the agent's performance when dealing with different tasks after adaptation. This paper provides a game-theoretical underpinning for understanding this type of security risk. In particular, we formally define the sampling attack model as a Stackelberg game between the attacker and the agent, which yields a minimax formulation. It leads to two online attack schemes: Intermittent Attack and Persistent Attack, which enable the attacker to learn an optimal sampling attack, defined by an $\epsilon$-first-order stationary point, within $\mathcal{O}(\epsilon^{-2})$ iterations. These attack schemes freeride the learning progress concurrently without extra interactions with the environment. By corroborating the convergence results with numerical experiments, we observe that a minor effort of the attacker can significantly deteriorate the learning performance, and the minimax approach can also help robustify the meta RL algorithms.



## **46. Can We Mitigate Backdoor Attack Using Adversarial Detection Methods?**

cs.LG

Accepted by IEEE TDSC

**SubmitDate**: 2022-07-28    [paper-pdf](http://arxiv.org/pdf/2006.14871v2)

**Authors**: Kaidi Jin, Tianwei Zhang, Chao Shen, Yufei Chen, Ming Fan, Chenhao Lin, Ting Liu

**Abstracts**: Deep Neural Networks are well known to be vulnerable to adversarial attacks and backdoor attacks, where minor modifications on the input are able to mislead the models to give wrong results. Although defenses against adversarial attacks have been widely studied, investigation on mitigating backdoor attacks is still at an early stage. It is unknown whether there are any connections and common characteristics between the defenses against these two attacks. We conduct comprehensive studies on the connections between adversarial examples and backdoor examples of Deep Neural Networks to seek to answer the question: can we detect backdoor using adversarial detection methods. Our insights are based on the observation that both adversarial examples and backdoor examples have anomalies during the inference process, highly distinguishable from benign samples. As a result, we revise four existing adversarial defense methods for detecting backdoor examples. Extensive evaluations indicate that these approaches provide reliable protection against backdoor attacks, with a higher accuracy than detecting adversarial examples. These solutions also reveal the relations of adversarial examples, backdoor examples and normal samples in model sensitivity, activation space and feature space. This is able to enhance our understanding about the inherent features of these two attacks and the defense opportunities.



## **47. Look Closer to Your Enemy: Learning to Attack via Teacher-student Mimicking**

cs.CV

13 pages, 8 figures, NDSS

**SubmitDate**: 2022-07-28    [paper-pdf](http://arxiv.org/pdf/2207.13381v2)

**Authors**: Mingejie Wang, Zhiqing Tang, Sirui Li, Dingwen Xiao

**Abstracts**: This paper aims to generate realistic attack samples of person re-identification, ReID, by reading the enemy's mind (VM). In this paper, we propose a novel inconspicuous and controllable ReID attack baseline, LCYE, to generate adversarial query images. Concretely, LCYE first distills VM's knowledge via teacher-student memory mimicking in the proxy task. Then this knowledge prior acts as an explicit cipher conveying what is essential and realistic, believed by VM, for accurate adversarial misleading. Besides, benefiting from the multiple opposing task framework of LCYE, we further investigate the interpretability and generalization of ReID models from the view of the adversarial attack, including cross-domain adaption, cross-model consensus, and online learning process. Extensive experiments on four ReID benchmarks show that our method outperforms other state-of-the-art attackers with a large margin in white-box, black-box, and target attacks. Our code is now available at https://gitfront.io/r/user-3704489/mKXusqDT4ffr/LCYE/.



## **48. Privacy-Preserving Federated Recurrent Neural Networks**

cs.CR

**SubmitDate**: 2022-07-28    [paper-pdf](http://arxiv.org/pdf/2207.13947v1)

**Authors**: Sinem Sav, Abdulrahman Diaa, Apostolos Pyrgelis, Jean-Philippe Bossuat, Jean-Pierre Hubaux

**Abstracts**: We present RHODE, a novel system that enables privacy-preserving training of and prediction on Recurrent Neural Networks (RNNs) in a federated learning setting by relying on multiparty homomorphic encryption (MHE). RHODE preserves the confidentiality of the training data, the model, and the prediction data; and it mitigates the federated learning attacks that target the gradients under a passive-adversary threat model. We propose a novel packing scheme, multi-dimensional packing, for a better utilization of Single Instruction, Multiple Data (SIMD) operations under encryption. With multi-dimensional packing, RHODE enables the efficient processing, in parallel, of a batch of samples. To avoid the exploding gradients problem, we also provide several clip-by-value approximations for enabling gradient clipping under encryption. We experimentally show that the model performance with RHODE remains similar to non-secure solutions both for homogeneous and heterogeneous data distribution among the data holders. Our experimental evaluation shows that RHODE scales linearly with the number of data holders and the number of timesteps, sub-linearly and sub-quadratically with the number of features and the number of hidden units of RNNs, respectively. To the best of our knowledge, RHODE is the first system that provides the building blocks for the training of RNNs and its variants, under encryption in a federated learning setting.



## **49. Label-Only Membership Inference Attack against Node-Level Graph Neural Networks**

cs.CR

**SubmitDate**: 2022-07-27    [paper-pdf](http://arxiv.org/pdf/2207.13766v1)

**Authors**: Mauro Conti, Jiaxin Li, Stjepan Picek, Jing Xu

**Abstracts**: Graph Neural Networks (GNNs), inspired by Convolutional Neural Networks (CNNs), aggregate the message of nodes' neighbors and structure information to acquire expressive representations of nodes for node classification, graph classification, and link prediction. Previous studies have indicated that GNNs are vulnerable to Membership Inference Attacks (MIAs), which infer whether a node is in the training data of GNNs and leak the node's private information, like the patient's disease history. The implementation of previous MIAs takes advantage of the models' probability output, which is infeasible if GNNs only provide the prediction label (label-only) for the input.   In this paper, we propose a label-only MIA against GNNs for node classification with the help of GNNs' flexible prediction mechanism, e.g., obtaining the prediction label of one node even when neighbors' information is unavailable. Our attacking method achieves around 60\% accuracy, precision, and Area Under the Curve (AUC) for most datasets and GNN models, some of which are competitive or even better than state-of-the-art probability-based MIAs implemented under our environment and settings. Additionally, we analyze the influence of the sampling method, model selection approach, and overfitting level on the attack performance of our label-only MIA. Both of those factors have an impact on the attack performance. Then, we consider scenarios where assumptions about the adversary's additional dataset (shadow dataset) and extra information about the target model are relaxed. Even in those scenarios, our label-only MIA achieves a better attack performance in most cases. Finally, we explore the effectiveness of possible defenses, including Dropout, Regularization, Normalization, and Jumping knowledge. None of those four defenses prevent our attack completely.



## **50. Membership Inference Attacks via Adversarial Examples**

cs.LG

**SubmitDate**: 2022-07-27    [paper-pdf](http://arxiv.org/pdf/2207.13572v1)

**Authors**: Hamid Jalalzai, Elie Kadoche, Rémi Leluc, Vincent Plassier

**Abstracts**: The raise of machine learning and deep learning led to significant improvement in several domains. This change is supported by both the dramatic rise in computation power and the collection of large datasets. Such massive datasets often include personal data which can represent a threat to privacy. Membership inference attacks are a novel direction of research which aims at recovering training data used by a learning algorithm. In this paper, we develop a mean to measure the leakage of training data leveraging a quantity appearing as a proxy of the total variation of a trained model near its training samples. We extend our work by providing a novel defense mechanism. Our contributions are supported by empirical evidence through convincing numerical experiments.



