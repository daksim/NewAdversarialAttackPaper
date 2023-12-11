# Latest Adversarial Attack Papers
**update at 2023-12-11 16:13:30**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. Topology-Based Reconstruction Prevention for Decentralised Learning**

cs.CR

15 pages, 8 figures, submitted to IEEE S&P 2024, for associated  experiment source code see doi:10.4121/21572601

**SubmitDate**: 2023-12-08    [abs](http://arxiv.org/abs/2312.05248v1) [paper-pdf](http://arxiv.org/pdf/2312.05248v1)

**Authors**: Florine W. Dekker, Zekeriya Erkin, Mauro Conti

**Abstract**: Decentralised learning has recently gained traction as an alternative to federated learning in which both data and coordination are distributed over its users. To preserve the confidentiality of users' data, decentralised learning relies on differential privacy, multi-party computation, or a combination thereof. However, running multiple privacy-preserving summations in sequence may allow adversaries to perform reconstruction attacks. Unfortunately, current reconstruction countermeasures either cannot trivially be adapted to the distributed setting, or add excessive amounts of noise.   In this work, we first show that passive honest-but-curious adversaries can reconstruct other users' private data after several privacy-preserving summations. For example, in subgraphs with 18 users, we show that only three passive honest-but-curious adversaries succeed at reconstructing private data 11.0% of the time, requiring an average of 8.8 summations per adversary. The success rate is independent of the size of the full network. We consider weak adversaries, who do not control the graph topology and can exploit neither the workings of the summation protocol nor the specifics of users' data.   We develop a mathematical understanding of how reconstruction relates to topology and propose the first topology-based decentralised defence against reconstruction attacks. Specifically, we show that reconstruction requires a number of adversaries linear in the length of the network's shortest cycle. Consequently, reconstructing private data from privacy-preserving summations is impossible in acyclic networks.   Our work is a stepping stone for a formal theory of decentralised reconstruction defences based on topology. Such a theory would generalise our countermeasure beyond summation, define confidentiality in terms of entropy, and describe the effects of (topology-aware) differential privacy.



## **2. On the Robustness of Large Multimodal Models Against Image Adversarial Attacks**

cs.CV

**SubmitDate**: 2023-12-08    [abs](http://arxiv.org/abs/2312.03777v2) [paper-pdf](http://arxiv.org/pdf/2312.03777v2)

**Authors**: Xuanming Cui, Alejandro Aparcedo, Young Kyun Jang, Ser-Nam Lim

**Abstract**: Recent advances in instruction tuning have led to the development of State-of-the-Art Large Multimodal Models (LMMs). Given the novelty of these models, the impact of visual adversarial attacks on LMMs has not been thoroughly examined. We conduct a comprehensive study of the robustness of various LMMs against different adversarial attacks, evaluated across tasks including image classification, image captioning, and Visual Question Answer (VQA). We find that in general LMMs are not robust to visual adversarial inputs. However, our findings suggest that context provided to the model via prompts, such as questions in a QA pair helps to mitigate the effects of visual adversarial inputs. Notably, the LMMs evaluated demonstrated remarkable resilience to such attacks on the ScienceQA task with only an 8.10% drop in performance compared to their visual counterparts which dropped 99.73%. We also propose a new approach to real-world image classification which we term query decomposition. By incorporating existence queries into our input prompt we observe diminished attack effectiveness and improvements in image classification accuracy. This research highlights a previously under-explored facet of LMM robustness and sets the stage for future work aimed at strengthening the resilience of multimodal systems in adversarial environments.



## **3. Partial-Information, Longitudinal Cyber Attacks on LiDAR in Autonomous Vehicles**

cs.CR

**SubmitDate**: 2023-12-08    [abs](http://arxiv.org/abs/2303.03470v3) [paper-pdf](http://arxiv.org/pdf/2303.03470v3)

**Authors**: R. Spencer Hallyburton, Qingzhao Zhang, Z. Morley Mao, Miroslav Pajic

**Abstract**: What happens to an autonomous vehicle (AV) if its data are adversarially compromised? Prior security studies have addressed this question through mostly unrealistic threat models, with limited practical relevance, such as white-box adversarial learning or nanometer-scale laser aiming and spoofing. With growing evidence that cyber threats pose real, imminent danger to AVs and cyber-physical systems (CPS) in general, we present and evaluate a novel AV threat model: a cyber-level attacker capable of disrupting sensor data but lacking any situational awareness. We demonstrate that even though the attacker has minimal knowledge and only access to raw data from a single sensor (i.e., LiDAR), she can design several attacks that critically compromise perception and tracking in multi-sensor AVs. To mitigate vulnerabilities and advance secure architectures in AVs, we introduce two improvements for security-aware fusion: a probabilistic data-asymmetry monitor and a scalable track-to-track fusion of 3D LiDAR and monocular detections (T2T-3DLM); we demonstrate that the approaches significantly reduce attack effectiveness. To support objective safety and security evaluations in AVs, we release our security evaluation platform, AVsec, which is built on security-relevant metrics to benchmark AVs on gold-standard longitudinal AV datasets and AV simulators.



## **4. MIMIR: Masked Image Modeling for Mutual Information-based Adversarial Robustness**

cs.CV

**SubmitDate**: 2023-12-08    [abs](http://arxiv.org/abs/2312.04960v1) [paper-pdf](http://arxiv.org/pdf/2312.04960v1)

**Authors**: Xiaoyun Xu, Shujian Yu, Jingzheng Wu, Stjepan Picek

**Abstract**: Vision Transformers (ViTs) achieve superior performance on various tasks compared to convolutional neural networks (CNNs), but ViTs are also vulnerable to adversarial attacks. Adversarial training is one of the most successful methods to build robust CNN models. Thus, recent works explored new methodologies for adversarial training of ViTs based on the differences between ViTs and CNNs, such as better training strategies, preventing attention from focusing on a single block, or discarding low-attention embeddings. However, these methods still follow the design of traditional supervised adversarial training, limiting the potential of adversarial training on ViTs. This paper proposes a novel defense method, MIMIR, which aims to build a different adversarial training methodology by utilizing Masked Image Modeling at pre-training. We create an autoencoder that accepts adversarial examples as input but takes the clean examples as the modeling target. Then, we create a mutual information (MI) penalty following the idea of the Information Bottleneck. Among the two information source inputs and corresponding adversarial perturbation, the perturbation information is eliminated due to the constraint of the modeling target. Next, we provide a theoretical analysis of MIMIR using the bounds of the MI penalty. We also design two adaptive attacks when the adversary is aware of the MIMIR defense and show that MIMIR still performs well. The experimental results show that MIMIR improves (natural and adversarial) accuracy on average by 4.19\% on CIFAR-10 and 5.52\% on ImageNet-1K, compared to baselines. On Tiny-ImageNet, we obtained improved natural accuracy of 2.99\% on average and comparable adversarial accuracy. Our code and trained models are publicly available\footnote{\url{https://anonymous.4open.science/r/MIMIR-5444/README.md}}.



## **5. AHSecAgg and TSKG: Lightweight Secure Aggregation for Federated Learning Without Compromise**

cs.CR

**SubmitDate**: 2023-12-08    [abs](http://arxiv.org/abs/2312.04937v1) [paper-pdf](http://arxiv.org/pdf/2312.04937v1)

**Authors**: Siqing Zhang, Yong Liao, Pengyuan Zhou

**Abstract**: Leveraging federated learning (FL) to enable cross-domain privacy-sensitive data mining represents a vital breakthrough to accomplish privacy-preserving learning. However, attackers can infer the original user data by analyzing the uploaded intermediate parameters during the aggregation process. Therefore, secure aggregation has become a critical issue in the field of FL. Many secure aggregation protocols face the problem of high computation costs, which severely limits their applicability. To this end, we propose AHSecAgg, a lightweight secure aggregation protocol using additive homomorphic masks. AHSecAgg significantly reduces computation overhead without compromising the dropout handling capability or model accuracy. We prove the security of AHSecAgg in semi-honest and active adversary settings. In addition, in cross-silo scenarios where the group of participants is relatively fixed during each round, we propose TSKG, a lightweight Threshold Signature based masking key generation method. TSKG can generate different temporary secrets and shares for different aggregation rounds using the initial key and thus effectively eliminates the cost of secret sharing and key agreement. We prove TSKG does not sacrifice security. Extensive experiments show that AHSecAgg significantly outperforms state-of-the-art mask-based secure aggregation protocols in terms of computational efficiency, and TSKG effectively reduces the computation and communication costs for existing secure aggregation protocols.



## **6. SA-Attack: Improving Adversarial Transferability of Vision-Language Pre-training Models via Self-Augmentation**

cs.CV

**SubmitDate**: 2023-12-08    [abs](http://arxiv.org/abs/2312.04913v1) [paper-pdf](http://arxiv.org/pdf/2312.04913v1)

**Authors**: Bangyan He, Xiaojun Jia, Siyuan Liang, Tianrui Lou, Yang Liu, Xiaochun Cao

**Abstract**: Current Visual-Language Pre-training (VLP) models are vulnerable to adversarial examples. These adversarial examples present substantial security risks to VLP models, as they can leverage inherent weaknesses in the models, resulting in incorrect predictions. In contrast to white-box adversarial attacks, transfer attacks (where the adversary crafts adversarial examples on a white-box model to fool another black-box model) are more reflective of real-world scenarios, thus making them more meaningful for research. By summarizing and analyzing existing research, we identified two factors that can influence the efficacy of transfer attacks on VLP models: inter-modal interaction and data diversity. Based on these insights, we propose a self-augment-based transfer attack method, termed SA-Attack. Specifically, during the generation of adversarial images and adversarial texts, we apply different data augmentation methods to the image modality and text modality, respectively, with the aim of improving the adversarial transferability of the generated adversarial images and texts. Experiments conducted on the FLickr30K and COCO datasets have validated the effectiveness of our method. Our code will be available after this paper is accepted.



## **7. HC-Ref: Hierarchical Constrained Refinement for Robust Adversarial Training of GNNs**

cs.LG

**SubmitDate**: 2023-12-08    [abs](http://arxiv.org/abs/2312.04879v1) [paper-pdf](http://arxiv.org/pdf/2312.04879v1)

**Authors**: Xiaobing Pei, Haoran Yang, Gang Shen

**Abstract**: Recent studies have shown that attackers can catastrophically reduce the performance of GNNs by maliciously modifying the graph structure or node features on the graph. Adversarial training, which has been shown to be one of the most effective defense mechanisms against adversarial attacks in computer vision, holds great promise for enhancing the robustness of GNNs. There is limited research on defending against attacks by performing adversarial training on graphs, and it is crucial to delve deeper into this approach to optimize its effectiveness. Therefore, based on robust adversarial training on graphs, we propose a hierarchical constraint refinement framework (HC-Ref) that enhances the anti-perturbation capabilities of GNNs and downstream classifiers separately, ultimately leading to improved robustness. We propose corresponding adversarial regularization terms that are conducive to adaptively narrowing the domain gap between the normal part and the perturbation part according to the characteristics of different layers, promoting the smoothness of the predicted distribution of both parts. Moreover, existing research on graph robust adversarial training primarily concentrates on training from the standpoint of node feature perturbations and seldom takes into account alterations in the graph structure. This limitation makes it challenging to prevent attacks based on topological changes in the graph. This paper generates adversarial examples by utilizing graph structure perturbations, offering an effective approach to defend against attack methods that are based on topological changes. Extensive experiments on two real-world graph benchmarks show that HC-Ref successfully resists various attacks and has better node classification performance compared to several baseline methods.



## **8. Data-Driven Identification of Attack-free Sensors in Networked Control Systems**

eess.SY

Conference submission

**SubmitDate**: 2023-12-08    [abs](http://arxiv.org/abs/2312.04845v1) [paper-pdf](http://arxiv.org/pdf/2312.04845v1)

**Authors**: Sribalaji C. Anand, Michelle S. Chong, André M. H. Teixeira

**Abstract**: This paper proposes a data-driven framework to identify the attack-free sensors in a networked control system when some of the sensors are corrupted by an adversary. An operator with access to offline input-output attack-free trajectories of the plant is considered. Then, a data-driven algorithm is proposed to identify the attack-free sensors when the plant is controlled online. We also provide necessary conditions, based on the properties of the plant, under which the algorithm is feasible. An extension of the algorithm is presented to identify the sensors completely online against certain classes of attacks. The efficacy of our algorithm is depicted through numerical examples.



## **9. MimicDiffusion: Purifying Adversarial Perturbation via Mimicking Clean Diffusion Model**

cs.CV

**SubmitDate**: 2023-12-08    [abs](http://arxiv.org/abs/2312.04802v1) [paper-pdf](http://arxiv.org/pdf/2312.04802v1)

**Authors**: Kaiyu Song, Hanjiang Lai

**Abstract**: Deep neural networks (DNNs) are vulnerable to adversarial perturbation, where an imperceptible perturbation is added to the image that can fool the DNNs. Diffusion-based adversarial purification focuses on using the diffusion model to generate a clean image against such adversarial attacks. Unfortunately, the generative process of the diffusion model is also inevitably affected by adversarial perturbation since the diffusion model is also a deep network where its input has adversarial perturbation. In this work, we propose MimicDiffusion, a new diffusion-based adversarial purification technique, that directly approximates the generative process of the diffusion model with the clean image as input. Concretely, we analyze the differences between the guided terms using the clean image and the adversarial sample. After that, we first implement MimicDiffusion based on Manhattan distance. Then, we propose two guidance to purify the adversarial perturbation and approximate the clean diffusion model. Extensive experiments on three image datasets including CIFAR-10, CIFAR-100, and ImageNet with three classifier backbones including WideResNet-70-16, WideResNet-28-10, and ResNet50 demonstrate that MimicDiffusion significantly performs better than the state-of-the-art baselines. On CIFAR-10, CIFAR-100, and ImageNet, it achieves 92.67\%, 61.35\%, and 61.53\% average robust accuracy, which are 18.49\%, 13.23\%, and 17.64\% higher, respectively. The code is available in the supplementary material.



## **10. DeceptPrompt: Exploiting LLM-driven Code Generation via Adversarial Natural Language Instructions**

cs.CR

**SubmitDate**: 2023-12-07    [abs](http://arxiv.org/abs/2312.04730v1) [paper-pdf](http://arxiv.org/pdf/2312.04730v1)

**Authors**: Fangzhou Wu, Xiaogeng Liu, Chaowei Xiao

**Abstract**: With the advancement of Large Language Models (LLMs), significant progress has been made in code generation, enabling LLMs to transform natural language into programming code. These Code LLMs have been widely accepted by massive users and organizations. However, a dangerous nature is hidden in the code, which is the existence of fatal vulnerabilities. While some LLM providers have attempted to address these issues by aligning with human guidance, these efforts fall short of making Code LLMs practical and robust. Without a deep understanding of the performance of the LLMs under the practical worst cases, it would be concerning to apply them to various real-world applications. In this paper, we answer the critical issue: Are existing Code LLMs immune to generating vulnerable code? If not, what is the possible maximum severity of this issue in practical deployment scenarios? In this paper, we introduce DeceptPrompt, a novel algorithm that can generate adversarial natural language instructions that drive the Code LLMs to generate functionality correct code with vulnerabilities. DeceptPrompt is achieved through a systematic evolution-based algorithm with a fine grain loss design. The unique advantage of DeceptPrompt enables us to find natural prefix/suffix with totally benign and non-directional semantic meaning, meanwhile, having great power in inducing the Code LLMs to generate vulnerable code. This feature can enable us to conduct the almost-worstcase red-teaming on these LLMs in a real scenario, where users are using natural language. Our extensive experiments and analyses on DeceptPrompt not only validate the effectiveness of our approach but also shed light on the huge weakness of LLMs in the code generation task. When applying the optimized prefix/suffix, the attack success rate (ASR) will improve by average 50% compared with no prefix/suffix applying.



## **11. gcDLSeg: Integrating Graph-cut into Deep Learning for Binary Semantic Segmentation**

cs.CV

12 pages

**SubmitDate**: 2023-12-07    [abs](http://arxiv.org/abs/2312.04713v1) [paper-pdf](http://arxiv.org/pdf/2312.04713v1)

**Authors**: Hui Xie, Weiyu Xu, Ya Xing Wang, John Buatti, Xiaodong Wu

**Abstract**: Binary semantic segmentation in computer vision is a fundamental problem. As a model-based segmentation method, the graph-cut approach was one of the most successful binary segmentation methods thanks to its global optimality guarantee of the solutions and its practical polynomial-time complexity. Recently, many deep learning (DL) based methods have been developed for this task and yielded remarkable performance, resulting in a paradigm shift in this field. To combine the strengths of both approaches, we propose in this study to integrate the graph-cut approach into a deep learning network for end-to-end learning. Unfortunately, backward propagation through the graph-cut module in the DL network is challenging due to the combinatorial nature of the graph-cut algorithm. To tackle this challenge, we propose a novel residual graph-cut loss and a quasi-residual connection, enabling the backward propagation of the gradients of the residual graph-cut loss for effective feature learning guided by the graph-cut segmentation model. In the inference phase, globally optimal segmentation is achieved with respect to the graph-cut energy defined on the optimized image features learned from DL networks. Experiments on the public AZH chronic wound data set and the pancreas cancer data set from the medical segmentation decathlon (MSD) demonstrated promising segmentation accuracy, and improved robustness against adversarial attacks.



## **12. Diffence: Fencing Membership Privacy With Diffusion Models**

cs.CR

**SubmitDate**: 2023-12-07    [abs](http://arxiv.org/abs/2312.04692v1) [paper-pdf](http://arxiv.org/pdf/2312.04692v1)

**Authors**: Yuefeng Peng, Ali Naseh, Amir Houmansadr

**Abstract**: Deep learning models, while achieving remarkable performance across various tasks, are vulnerable to member inference attacks, wherein adversaries identify if a specific data point was part of a model's training set. This susceptibility raises substantial privacy concerns, especially when models are trained on sensitive datasets. Current defense methods often struggle to provide robust protection without hurting model utility, and they often require retraining the model or using extra data. In this work, we introduce a novel defense framework against membership attacks by leveraging generative models. The key intuition of our defense is to remove the differences between member and non-member inputs which can be used to perform membership attacks, by re-generating input samples before feeding them to the target model. Therefore, our defense works \emph{pre-inference}, which is unlike prior defenses that are either training-time (modify the model) or post-inference time (modify the model's output).   A unique feature of our defense is that it works on input samples only, without modifying the training or inference phase of the target model. Therefore, it can be cascaded with other defense mechanisms as we demonstrate through experiments. Through extensive experimentation, we show that our approach can serve as a robust plug-n-play defense mechanism, enhancing membership privacy without compromising model utility in both baseline and defended settings. For example, our method enhanced the effectiveness of recent state-of-the-art defenses, reducing attack accuracy by an average of 5.7\% to 12.4\% across three datasets, without any impact on the model's accuracy. By integrating our method with prior defenses, we achieve new state-of-the-art performance in the privacy-utility trade-off.



## **13. Defending Against Alignment-Breaking Attacks via Robustly Aligned LLM**

cs.CL

16 Pages, 5 Figures, 6 Tables

**SubmitDate**: 2023-12-07    [abs](http://arxiv.org/abs/2309.14348v2) [paper-pdf](http://arxiv.org/pdf/2309.14348v2)

**Authors**: Bochuan Cao, Yuanpu Cao, Lu Lin, Jinghui Chen

**Abstract**: Recently, Large Language Models (LLMs) have made significant advancements and are now widely used across various domains. Unfortunately, there has been a rising concern that LLMs can be misused to generate harmful or malicious content. Though a line of research has focused on aligning LLMs with human values and preventing them from producing inappropriate content, such alignments are usually vulnerable and can be bypassed by alignment-breaking attacks via adversarially optimized or handcrafted jailbreaking prompts. In this work, we introduce a Robustly Aligned LLM (RA-LLM) to defend against potential alignment-breaking attacks. RA-LLM can be directly constructed upon an existing aligned LLM with a robust alignment checking function, without requiring any expensive retraining or fine-tuning process of the original LLM. Furthermore, we also provide a theoretical analysis for RA-LLM to verify its effectiveness in defending against alignment-breaking attacks. Through real-world experiments on open-source large language models, we demonstrate that RA-LLM can successfully defend against both state-of-the-art adversarial prompts and popular handcrafted jailbreaking prompts by reducing their attack success rates from nearly 100% to around 10% or less.



## **14. FreqFed: A Frequency Analysis-Based Approach for Mitigating Poisoning Attacks in Federated Learning**

cs.CR

To appear in the Network and Distributed System Security (NDSS)  Symposium 2024. 16 pages, 8 figures, 12 tables, 1 algorithm, 3 equations

**SubmitDate**: 2023-12-07    [abs](http://arxiv.org/abs/2312.04432v1) [paper-pdf](http://arxiv.org/pdf/2312.04432v1)

**Authors**: Hossein Fereidooni, Alessandro Pegoraro, Phillip Rieger, Alexandra Dmitrienko, Ahmad-Reza Sadeghi

**Abstract**: Federated learning (FL) is a collaborative learning paradigm allowing multiple clients to jointly train a model without sharing their training data. However, FL is susceptible to poisoning attacks, in which the adversary injects manipulated model updates into the federated model aggregation process to corrupt or destroy predictions (untargeted poisoning) or implant hidden functionalities (targeted poisoning or backdoors). Existing defenses against poisoning attacks in FL have several limitations, such as relying on specific assumptions about attack types and strategies or data distributions or not sufficiently robust against advanced injection techniques and strategies and simultaneously maintaining the utility of the aggregated model. To address the deficiencies of existing defenses, we take a generic and completely different approach to detect poisoning (targeted and untargeted) attacks. We present FreqFed, a novel aggregation mechanism that transforms the model updates (i.e., weights) into the frequency domain, where we can identify the core frequency components that inherit sufficient information about weights. This allows us to effectively filter out malicious updates during local training on the clients, regardless of attack types, strategies, and clients' data distributions. We extensively evaluate the efficiency and effectiveness of FreqFed in different application domains, including image classification, word prediction, IoT intrusion detection, and speech recognition. We demonstrate that FreqFed can mitigate poisoning attacks effectively with a negligible impact on the utility of the aggregated model.



## **15. OT-Attack: Enhancing Adversarial Transferability of Vision-Language Models via Optimal Transport Optimization**

cs.CV

**SubmitDate**: 2023-12-07    [abs](http://arxiv.org/abs/2312.04403v1) [paper-pdf](http://arxiv.org/pdf/2312.04403v1)

**Authors**: Dongchen Han, Xiaojun Jia, Yang Bai, Jindong Gu, Yang Liu, Xiaochun Cao

**Abstract**: Vision-language pre-training (VLP) models demonstrate impressive abilities in processing both images and text. However, they are vulnerable to multi-modal adversarial examples (AEs). Investigating the generation of high-transferability adversarial examples is crucial for uncovering VLP models' vulnerabilities in practical scenarios. Recent works have indicated that leveraging data augmentation and image-text modal interactions can enhance the transferability of adversarial examples for VLP models significantly. However, they do not consider the optimal alignment problem between dataaugmented image-text pairs. This oversight leads to adversarial examples that are overly tailored to the source model, thus limiting improvements in transferability. In our research, we first explore the interplay between image sets produced through data augmentation and their corresponding text sets. We find that augmented image samples can align optimally with certain texts while exhibiting less relevance to others. Motivated by this, we propose an Optimal Transport-based Adversarial Attack, dubbed OT-Attack. The proposed method formulates the features of image and text sets as two distinct distributions and employs optimal transport theory to determine the most efficient mapping between them. This optimal mapping informs our generation of adversarial examples to effectively counteract the overfitting issues. Extensive experiments across various network architectures and datasets in image-text matching tasks reveal that our OT-Attack outperforms existing state-of-the-art methods in terms of adversarial transferability.



## **16. Similarity of Neural Architectures using Adversarial Attack Transferability**

cs.LG

20pages, 13 figures, 2.3MB

**SubmitDate**: 2023-12-07    [abs](http://arxiv.org/abs/2210.11407v3) [paper-pdf](http://arxiv.org/pdf/2210.11407v3)

**Authors**: Jaehui Hwang, Dongyoon Han, Byeongho Heo, Song Park, Sanghyuk Chun, Jong-Seok Lee

**Abstract**: In recent years, many deep neural architectures have been developed for image classification. Whether they are similar or dissimilar and what factors contribute to their (dis)similarities remains curious. To address this question, we aim to design a quantitative and scalable similarity measure between neural architectures. We propose Similarity by Attack Transferability (SAT) from the observation that adversarial attack transferability contains information related to input gradients and decision boundaries widely used to understand model behaviors. We conduct a large-scale analysis on 69 state-of-the-art ImageNet classifiers using our proposed similarity function to answer the question. Moreover, we observe neural architecture-related phenomena using model similarity that model diversity can lead to better performance on model ensembles and knowledge distillation under specific conditions. Our results provide insights into why developing diverse neural architectures with distinct components is necessary.



## **17. Temporal Shuffling for Defending Deep Action Recognition Models against Adversarial Attacks**

cs.CV

12 pages, accepted to Neural Networks

**SubmitDate**: 2023-12-07    [abs](http://arxiv.org/abs/2112.07921v2) [paper-pdf](http://arxiv.org/pdf/2112.07921v2)

**Authors**: Jaehui Hwang, Huan Zhang, Jun-Ho Choi, Cho-Jui Hsieh, Jong-Seok Lee

**Abstract**: Recently, video-based action recognition methods using convolutional neural networks (CNNs) achieve remarkable recognition performance. However, there is still lack of understanding about the generalization mechanism of action recognition models. In this paper, we suggest that action recognition models rely on the motion information less than expected, and thus they are robust to randomization of frame orders. Furthermore, we find that motion monotonicity remaining after randomization also contributes to such robustness. Based on this observation, we develop a novel defense method using temporal shuffling of input videos against adversarial attacks for action recognition models. Another observation enabling our defense method is that adversarial perturbations on videos are sensitive to temporal destruction. To the best of our knowledge, this is the first attempt to design a defense method without additional training for 3D CNN-based video action recognition models.



## **18. Experimentally Certified Transmission of a Quantum Message through an Untrusted and Lossy Quantum Channel via Bell's Theorem**

quant-ph

35 pages, 14 figures

**SubmitDate**: 2023-12-07    [abs](http://arxiv.org/abs/2304.09605v2) [paper-pdf](http://arxiv.org/pdf/2304.09605v2)

**Authors**: Simon Neves, Laura dos Santos Martins, Verena Yacoub, Pascal Lefebvre, Ivan Supic, Damian Markham, Eleni Diamanti

**Abstract**: Quantum transmission links are central elements in essentially all protocols involving the exchange of quantum messages. Emerging progress in quantum technologies involving such links needs to be accompanied by appropriate certification tools. In adversarial scenarios, a certification method can be vulnerable to attacks if too much trust is placed on the underlying system. Here, we propose a protocol in a device independent framework, which allows for the certification of practical quantum transmission links in scenarios where minimal assumptions are made about the functioning of the certification setup. In particular, we take unavoidable transmission losses into account by modeling the link as a completely-positive trace-decreasing map. We also, crucially, remove the assumption of independent and identically distributed samples, which is known to be incompatible with adversarial settings. Finally, in view of the use of the certified transmitted states for follow-up applications, our protocol moves beyond certification of the channel to allow us to estimate the quality of the transmitted quantum message itself. To illustrate the practical relevance and the feasibility of our protocol with currently available technology we provide an experimental implementation based on a state-of-the-art polarization entangled photon pair source in a Sagnac configuration and analyze its robustness for realistic losses and errors.



## **19. Adv-4-Adv: Thwarting Changing Adversarial Perturbations via Adversarial Domain Adaptation**

cs.CV

22 pages

**SubmitDate**: 2023-12-07    [abs](http://arxiv.org/abs/2112.00428v3) [paper-pdf](http://arxiv.org/pdf/2112.00428v3)

**Authors**: Tianyue Zheng, Zhe Chen, Shuya Ding, Chao Cai, Jun Luo

**Abstract**: Whereas adversarial training can be useful against specific adversarial perturbations, they have also proven ineffective in generalizing towards attacks deviating from those used for training. However, we observe that this ineffectiveness is intrinsically connected to domain adaptability, another crucial issue in deep learning for which adversarial domain adaptation appears to be a promising solution. Consequently, we proposed Adv-4-Adv as a novel adversarial training method that aims to retain robustness against unseen adversarial perturbations. Essentially, Adv-4-Adv treats attacks incurring different perturbations as distinct domains, and by leveraging the power of adversarial domain adaptation, it aims to remove the domain/attack-specific features. This forces a trained model to learn a robust domain-invariant representation, which in turn enhances its generalization ability. Extensive evaluations on Fashion-MNIST, SVHN, CIFAR-10, and CIFAR-100 demonstrate that a model trained by Adv-4-Adv based on samples crafted by simple attacks (e.g., FGSM) can be generalized to more advanced attacks (e.g., PGD), and the performance exceeds state-of-the-art proposals on these datasets.



## **20. Point Cloud Attacks in Graph Spectral Domain: When 3D Geometry Meets Graph Signal Processing**

cs.CV

Accepted to IEEE Transactions on Pattern Analysis and Machine  Intelligence (TPAMI). arXiv admin note: substantial text overlap with  arXiv:2202.07261

**SubmitDate**: 2023-12-07    [abs](http://arxiv.org/abs/2207.13326v2) [paper-pdf](http://arxiv.org/pdf/2207.13326v2)

**Authors**: Daizong Liu, Wei Hu, Xin Li

**Abstract**: With the increasing attention in various 3D safety-critical applications, point cloud learning models have been shown to be vulnerable to adversarial attacks. Although existing 3D attack methods achieve high success rates, they delve into the data space with point-wise perturbation, which may neglect the geometric characteristics. Instead, we propose point cloud attacks from a new perspective -- the graph spectral domain attack, aiming to perturb graph transform coefficients in the spectral domain that corresponds to varying certain geometric structure. Specifically, leveraging on graph signal processing, we first adaptively transform the coordinates of points onto the spectral domain via graph Fourier transform (GFT) for compact representation. Then, we analyze the influence of different spectral bands on the geometric structure, based on which we propose to perturb the GFT coefficients via a learnable graph spectral filter. Considering the low-frequency components mainly contribute to the rough shape of the 3D object, we further introduce a low-frequency constraint to limit perturbations within imperceptible high-frequency components. Finally, the adversarial point cloud is generated by transforming the perturbed spectral representation back to the data domain via the inverse GFT. Experimental results demonstrate the effectiveness of the proposed attack in terms of both the imperceptibility and attack success rates.



## **21. Defense against ML-based Power Side-channel Attacks on DNN Accelerators with Adversarial Attacks**

cs.CR

**SubmitDate**: 2023-12-07    [abs](http://arxiv.org/abs/2312.04035v1) [paper-pdf](http://arxiv.org/pdf/2312.04035v1)

**Authors**: Xiaobei Yan, Chip Hong Chang, Tianwei Zhang

**Abstract**: Artificial Intelligence (AI) hardware accelerators have been widely adopted to enhance the efficiency of deep learning applications. However, they also raise security concerns regarding their vulnerability to power side-channel attacks (SCA). In these attacks, the adversary exploits unintended communication channels to infer sensitive information processed by the accelerator, posing significant privacy and copyright risks to the models. Advanced machine learning algorithms are further employed to facilitate the side-channel analysis and exacerbate the privacy issue of AI accelerators. Traditional defense strategies naively inject execution noise to the runtime of AI models, which inevitably introduce large overheads.   In this paper, we present AIAShield, a novel defense methodology to safeguard FPGA-based AI accelerators and mitigate model extraction threats via power-based SCAs. The key insight of AIAShield is to leverage the prominent adversarial attack technique from the machine learning community to craft delicate noise, which can significantly obfuscate the adversary's side-channel observation while incurring minimal overhead to the execution of the protected model. At the hardware level, we design a new module based on ring oscillators to achieve fine-grained noise generation. At the algorithm level, we repurpose Neural Architecture Search to worsen the adversary's extraction results. Extensive experiments on the Nvidia Deep Learning Accelerator (NVDLA) demonstrate that AIAShield outperforms existing solutions with excellent transferability.



## **22. Characterizing the Optimal 0-1 Loss for Multi-class Classification with a Test-time Attacker**

cs.LG

NeurIPS 2023 Spotlight

**SubmitDate**: 2023-12-06    [abs](http://arxiv.org/abs/2302.10722v2) [paper-pdf](http://arxiv.org/pdf/2302.10722v2)

**Authors**: Sihui Dai, Wenxin Ding, Arjun Nitin Bhagoji, Daniel Cullina, Ben Y. Zhao, Haitao Zheng, Prateek Mittal

**Abstract**: Finding classifiers robust to adversarial examples is critical for their safe deployment. Determining the robustness of the best possible classifier under a given threat model for a given data distribution and comparing it to that achieved by state-of-the-art training methods is thus an important diagnostic tool. In this paper, we find achievable information-theoretic lower bounds on loss in the presence of a test-time attacker for multi-class classifiers on any discrete dataset. We provide a general framework for finding the optimal 0-1 loss that revolves around the construction of a conflict hypergraph from the data and adversarial constraints. We further define other variants of the attacker-classifier game that determine the range of the optimal loss more efficiently than the full-fledged hypergraph construction. Our evaluation shows, for the first time, an analysis of the gap to optimal robustness for classifiers in the multi-class setting on benchmark datasets.



## **23. Dr. Jekyll and Mr. Hyde: Two Faces of LLMs**

cs.CR

**SubmitDate**: 2023-12-06    [abs](http://arxiv.org/abs/2312.03853v1) [paper-pdf](http://arxiv.org/pdf/2312.03853v1)

**Authors**: Matteo Gioele Collu, Tom Janssen-Groesbeek, Stefanos Koffas, Mauro Conti, Stjepan Picek

**Abstract**: This year, we witnessed a rise in the use of Large Language Models, especially when combined with applications like chatbot assistants. Safety mechanisms and specialized training procedures are put in place to prevent improper responses from these assistants. In this work, we bypass these measures for ChatGPT and Bard (and, to some extent, Bing chat) by making them impersonate complex personas with opposite characteristics as those of the truthful assistants they are supposed to be. We start by creating elaborate biographies of these personas, which we then use in a new session with the same chatbots. Our conversation followed a role-play style to get the response the assistant was not allowed to provide. By making use of personas, we show that the response that is prohibited is actually provided, making it possible to obtain unauthorized, illegal, or harmful information. This work shows that by using adversarial personas, one can overcome safety mechanisms set out by ChatGPT and Bard. It also introduces several ways of activating such adversarial personas, altogether showing that both chatbots are vulnerable to this kind of attack.



## **24. Memory Triggers: Unveiling Memorization in Text-To-Image Generative Models through Word-Level Duplication**

cs.CR

**SubmitDate**: 2023-12-06    [abs](http://arxiv.org/abs/2312.03692v1) [paper-pdf](http://arxiv.org/pdf/2312.03692v1)

**Authors**: Ali Naseh, Jaechul Roh, Amir Houmansadr

**Abstract**: Diffusion-based models, such as the Stable Diffusion model, have revolutionized text-to-image synthesis with their ability to produce high-quality, high-resolution images. These advancements have prompted significant progress in image generation and editing tasks. However, these models also raise concerns due to their tendency to memorize and potentially replicate exact training samples, posing privacy risks and enabling adversarial attacks. Duplication in training datasets is recognized as a major factor contributing to memorization, and various forms of memorization have been studied so far. This paper focuses on two distinct and underexplored types of duplication that lead to replication during inference in diffusion-based models, particularly in the Stable Diffusion model. We delve into these lesser-studied duplication phenomena and their implications through two case studies, aiming to contribute to the safer and more responsible use of generative models in various applications.



## **25. Temporal Robustness against Data Poisoning**

cs.LG

37th Conference on Neural Information Processing Systems (NeurIPS  2023)

**SubmitDate**: 2023-12-06    [abs](http://arxiv.org/abs/2302.03684v3) [paper-pdf](http://arxiv.org/pdf/2302.03684v3)

**Authors**: Wenxiao Wang, Soheil Feizi

**Abstract**: Data poisoning considers cases when an adversary manipulates the behavior of machine learning algorithms through malicious training data. Existing threat models of data poisoning center around a single metric, the number of poisoned samples. In consequence, if attackers can poison more samples than expected with affordable overhead, as in many practical scenarios, they may be able to render existing defenses ineffective in a short time. To address this issue, we leverage timestamps denoting the birth dates of data, which are often available but neglected in the past. Benefiting from these timestamps, we propose a temporal threat model of data poisoning with two novel metrics, earliness and duration, which respectively measure how long an attack started in advance and how long an attack lasted. Using these metrics, we define the notions of temporal robustness against data poisoning, providing a meaningful sense of protection even with unbounded amounts of poisoned samples when the attacks are temporally bounded. We present a benchmark with an evaluation protocol simulating continuous data collection and periodic deployments of updated models, thus enabling empirical evaluation of temporal robustness. Lastly, we develop and also empirically verify a baseline defense, namely temporal aggregation, offering provable temporal robustness and highlighting the potential of our temporal threat model for data poisoning.



## **26. PyraTrans: Attention-Enriched Pyramid Transformer for Malicious URL Detection**

cs.CR

12 pages, 7 figures

**SubmitDate**: 2023-12-06    [abs](http://arxiv.org/abs/2312.00508v2) [paper-pdf](http://arxiv.org/pdf/2312.00508v2)

**Authors**: Ruitong Liu, Yanbin Wang, Zhenhao Guo, Haitao Xu, Zhan Qin, Wenrui Ma, Fan Zhang

**Abstract**: Although advancements in machine learning have driven the development of malicious URL detection technology, current techniques still face significant challenges in their capacity to generalize and their resilience against evolving threats. In this paper, we propose PyraTrans, a novel method that integrates pretrained Transformers with pyramid feature learning to detect malicious URL. PyraTrans utilizes a pretrained CharBERT as its foundation and is augmented with three interconnected feature modules: 1) Encoder Feature Extraction, extracting multi-order feature matrices from each CharBERT encoder layer; 2) Multi-Scale Feature Learning, capturing local contextual insights at various scales and aggregating information across encoder layers; and 3) Spatial Pyramid Attention, focusing on regional-level attention to emphasize areas rich in expressive information. The proposed approach addresses the limitations of the Transformer in local feature learning and regional relational awareness, which are vital for capturing URL-specific word patterns, character combinations, or structural anomalies. In several challenging experimental scenarios, the proposed method has shown significant improvements in accuracy, generalization, and robustness in malicious URL detection. For instance, it achieved a peak F1-score improvement of 40% in class-imbalanced scenarios, and exceeded the best baseline result by 14.13% in accuracy in adversarial attack scenarios. Additionally, we conduct a case study where our method accurately identifies all 30 active malicious web pages, whereas two pior SOTA methods miss 4 and 7 malicious web pages respectively. Codes and data are available at:https://github.com/Alixyvtte/PyraTrans.



## **27. Defense Against Adversarial Attacks using Convolutional Auto-Encoders**

cs.CV

9 pages, 6 figures, 3 tables

**SubmitDate**: 2023-12-06    [abs](http://arxiv.org/abs/2312.03520v1) [paper-pdf](http://arxiv.org/pdf/2312.03520v1)

**Authors**: Shreyasi Mandal

**Abstract**: Deep learning models, while achieving state-of-the-art performance on many tasks, are susceptible to adversarial attacks that exploit inherent vulnerabilities in their architectures. Adversarial attacks manipulate the input data with imperceptible perturbations, causing the model to misclassify the data or produce erroneous outputs. This work is based on enhancing the robustness of targeted classifier models against adversarial attacks. To achieve this, an convolutional autoencoder-based approach is employed that effectively counters adversarial perturbations introduced to the input images. By generating images closely resembling the input images, the proposed methodology aims to restore the model's accuracy.



## **28. Quantum-secured single-pixel imaging under general spoofing attack**

quant-ph

9 pages, 6 figures

**SubmitDate**: 2023-12-06    [abs](http://arxiv.org/abs/2312.03465v1) [paper-pdf](http://arxiv.org/pdf/2312.03465v1)

**Authors**: Jaesung Heo, Taek Jeong, Nam Hun Park, Yonggi Jo

**Abstract**: In this paper, we introduce a quantum-secured single-pixel imaging (QS-SPI) technique designed to withstand spoofing attacks, wherein adversaries attempt to deceive imaging systems with fake signals. Unlike previous quantum-secured protocols that impose a threshold error rate limiting their operation, even with the existence of true signals, our approach not only identifies spoofing attacks but also facilitates the reconstruction of a true image. Our method involves the analysis of a specific mode correlation of a photon-pair, which is independent of the mode used for image construction, to check security. Through this analysis, we can identify both the targeted image region by the attack and the type of spoofing attack, enabling reconstruction of the true image. A proof-of-principle demonstration employing polarization-correlation of a photon-pair is provided, showcasing successful image reconstruction even under the condition of spoofing signals 2000 times stronger than the true signals. We expect our approach to be applied to quantum-secured signal processing such as quantum target detection or ranging.



## **29. Synthesizing Physical Backdoor Datasets: An Automated Framework Leveraging Deep Generative Models**

cs.CR

**SubmitDate**: 2023-12-06    [abs](http://arxiv.org/abs/2312.03419v1) [paper-pdf](http://arxiv.org/pdf/2312.03419v1)

**Authors**: Sze Jue Yang, Chinh D. La, Quang H. Nguyen, Eugene Bagdasaryan, Kok-Seng Wong, Anh Tuan Tran, Chee Seng Chan, Khoa D. Doan

**Abstract**: Backdoor attacks, representing an emerging threat to the integrity of deep neural networks, have garnered significant attention due to their ability to compromise deep learning systems clandestinely. While numerous backdoor attacks occur within the digital realm, their practical implementation in real-world prediction systems remains limited and vulnerable to disturbances in the physical world. Consequently, this limitation has given rise to the development of physical backdoor attacks, where trigger objects manifest as physical entities within the real world. However, creating the requisite dataset to train or evaluate a physical backdoor model is a daunting task, limiting the backdoor researchers and practitioners from studying such physical attack scenarios. This paper unleashes a recipe that empowers backdoor researchers to effortlessly create a malicious, physical backdoor dataset based on advances in generative modeling. Particularly, this recipe involves 3 automatic modules: suggesting the suitable physical triggers, generating the poisoned candidate samples (either by synthesizing new samples or editing existing clean samples), and finally refining for the most plausible ones. As such, it effectively mitigates the perceived complexity associated with creating a physical backdoor dataset, transforming it from a daunting task into an attainable objective. Extensive experiment results show that datasets created by our "recipe" enable adversaries to achieve an impressive attack success rate on real physical world data and exhibit similar properties compared to previous physical backdoor attack studies. This paper offers researchers a valuable toolkit for studies of physical backdoors, all within the confines of their laboratories.



## **30. SAIF: Sparse Adversarial and Imperceptible Attack Framework**

cs.CV

**SubmitDate**: 2023-12-06    [abs](http://arxiv.org/abs/2212.07495v2) [paper-pdf](http://arxiv.org/pdf/2212.07495v2)

**Authors**: Tooba Imtiaz, Morgan Kohler, Jared Miller, Zifeng Wang, Mario Sznaier, Octavia Camps, Jennifer Dy

**Abstract**: Adversarial attacks hamper the decision-making ability of neural networks by perturbing the input signal. The addition of calculated small distortion to images, for instance, can deceive a well-trained image classification network. In this work, we propose a novel attack technique called Sparse Adversarial and Interpretable Attack Framework (SAIF). Specifically, we design imperceptible attacks that contain low-magnitude perturbations at a small number of pixels and leverage these sparse attacks to reveal the vulnerability of classifiers. We use the Frank-Wolfe (conditional gradient) algorithm to simultaneously optimize the attack perturbations for bounded magnitude and sparsity with $O(1/\sqrt{T})$ convergence. Empirical results show that SAIF computes highly imperceptible and interpretable adversarial examples, and outperforms state-of-the-art sparse attack methods on the ImageNet dataset.



## **31. Privacy-Preserving Task-Oriented Semantic Communications Against Model Inversion Attacks**

cs.IT

**SubmitDate**: 2023-12-06    [abs](http://arxiv.org/abs/2312.03252v1) [paper-pdf](http://arxiv.org/pdf/2312.03252v1)

**Authors**: Yanhu Wang, Shuaishuai Guo, Yiqin Deng, Haixia Zhang, Yuguang Fang

**Abstract**: Semantic communication has been identified as a core technology for the sixth generation (6G) of wireless networks. Recently, task-oriented semantic communications have been proposed for low-latency inference with limited bandwidth. Although transmitting only task-related information does protect a certain level of user privacy, adversaries could apply model inversion techniques to reconstruct the raw data or extract useful information, thereby infringing on users' privacy. To mitigate privacy infringement, this paper proposes an information bottleneck and adversarial learning (IBAL) approach to protect users' privacy against model inversion attacks. Specifically, we extract task-relevant features from the input based on the information bottleneck (IB) theory. To overcome the difficulty in calculating the mutual information in high-dimensional space, we derive a variational upper bound to estimate the true mutual information. To prevent data reconstruction from task-related features by adversaries, we leverage adversarial learning to train encoder to fool adversaries by maximizing reconstruction distortion. Furthermore, considering the impact of channel variations on privacy-utility trade-off and the difficulty in manually tuning the weights of each loss, we propose an adaptive weight adjustment method. Numerical results demonstrate that the proposed approaches can effectively protect privacy without significantly affecting task performance and achieve better privacy-utility trade-offs than baseline methods.



## **32. A Simple Framework to Enhance the Adversarial Robustness of Deep Learning-based Intrusion Detection System**

cs.CR

Accepted by Computers & Security

**SubmitDate**: 2023-12-06    [abs](http://arxiv.org/abs/2312.03245v1) [paper-pdf](http://arxiv.org/pdf/2312.03245v1)

**Authors**: Xinwei Yuan, Shu Han, Wei Huang, Hongliang Ye, Xianglong Kong, Fan Zhang

**Abstract**: Deep learning based intrusion detection systems (DL-based IDS) have emerged as one of the best choices for providing security solutions against various network intrusion attacks. However, due to the emergence and development of adversarial deep learning technologies, it becomes challenging for the adoption of DL models into IDS. In this paper, we propose a novel IDS architecture that can enhance the robustness of IDS against adversarial attacks by combining conventional machine learning (ML) models and Deep Learning models. The proposed DLL-IDS consists of three components: DL-based IDS, adversarial example (AE) detector, and ML-based IDS. We first develop a novel AE detector based on the local intrinsic dimensionality (LID). Then, we exploit the low attack transferability between DL models and ML models to find a robust ML model that can assist us in determining the maliciousness of AEs. If the input traffic is detected as an AE, the ML-based IDS will predict the maliciousness of input traffic, otherwise the DL-based IDS will work for the prediction. The fusion mechanism can leverage the high prediction accuracy of DL models and low attack transferability between DL models and ML models to improve the robustness of the whole system. In our experiments, we observe a significant improvement in the prediction performance of the IDS when subjected to adversarial attack, achieving high accuracy with low resource consumption.



## **33. Model-tuning Via Prompts Makes NLP Models Adversarially Robust**

cs.CL

Accepted to the EMNLP 2023 Conference

**SubmitDate**: 2023-12-06    [abs](http://arxiv.org/abs/2303.07320v2) [paper-pdf](http://arxiv.org/pdf/2303.07320v2)

**Authors**: Mrigank Raman, Pratyush Maini, J. Zico Kolter, Zachary C. Lipton, Danish Pruthi

**Abstract**: In recent years, NLP practitioners have converged on the following practice: (i) import an off-the-shelf pretrained (masked) language model; (ii) append a multilayer perceptron atop the CLS token's hidden representation (with randomly initialized weights); and (iii) fine-tune the entire model on a downstream task (MLP-FT). This procedure has produced massive gains on standard NLP benchmarks, but these models remain brittle, even to mild adversarial perturbations. In this work, we demonstrate surprising gains in adversarial robustness enjoyed by Model-tuning Via Prompts (MVP), an alternative method of adapting to downstream tasks. Rather than appending an MLP head to make output prediction, MVP appends a prompt template to the input, and makes prediction via text infilling/completion. Across 5 NLP datasets, 4 adversarial attacks, and 3 different models, MVP improves performance against adversarial substitutions by an average of 8% over standard methods and even outperforms adversarial training-based state-of-art defenses by 3.5%. By combining MVP with adversarial training, we achieve further improvements in adversarial robustness while maintaining performance on unperturbed examples. Finally, we conduct ablations to investigate the mechanism underlying these gains. Notably, we find that the main causes of vulnerability of MLP-FT can be attributed to the misalignment between pre-training and fine-tuning tasks, and the randomly initialized MLP parameters.



## **34. Effective Backdoor Mitigation Depends on the Pre-training Objective**

cs.LG

Accepted for oral presentation at BUGS workshop @ NeurIPS 2023  (https://neurips2023-bugs.github.io/)

**SubmitDate**: 2023-12-05    [abs](http://arxiv.org/abs/2311.14948v3) [paper-pdf](http://arxiv.org/pdf/2311.14948v3)

**Authors**: Sahil Verma, Gantavya Bhatt, Avi Schwarzschild, Soumye Singhal, Arnav Mohanty Das, Chirag Shah, John P Dickerson, Jeff Bilmes

**Abstract**: Despite the advanced capabilities of contemporary machine learning (ML) models, they remain vulnerable to adversarial and backdoor attacks. This vulnerability is particularly concerning in real-world deployments, where compromised models may exhibit unpredictable behavior in critical scenarios. Such risks are heightened by the prevalent practice of collecting massive, internet-sourced datasets for pre-training multimodal models, as these datasets may harbor backdoors. Various techniques have been proposed to mitigate the effects of backdooring in these models such as CleanCLIP which is the current state-of-the-art approach. In this work, we demonstrate that the efficacy of CleanCLIP in mitigating backdoors is highly dependent on the particular objective used during model pre-training. We observe that stronger pre-training objectives correlate with harder to remove backdoors behaviors. We show this by training multimodal models on two large datasets consisting of 3 million (CC3M) and 6 million (CC6M) datapoints, under various pre-training objectives, followed by poison removal using CleanCLIP. We find that CleanCLIP is ineffective when stronger pre-training objectives are used, even with extensive hyperparameter tuning. Our findings underscore critical considerations for ML practitioners who pre-train models using large-scale web-curated data and are concerned about potential backdoor threats. Notably, our results suggest that simpler pre-training objectives are more amenable to effective backdoor removal. This insight is pivotal for practitioners seeking to balance the trade-offs between using stronger pre-training objectives and security against backdoor attacks.



## **35. Beyond Detection: Unveiling Fairness Vulnerabilities in Abusive Language Models**

cs.CL

Under review

**SubmitDate**: 2023-12-05    [abs](http://arxiv.org/abs/2311.09428v2) [paper-pdf](http://arxiv.org/pdf/2311.09428v2)

**Authors**: Yueqing Liang, Lu Cheng, Ali Payani, Kai Shu

**Abstract**: This work investigates the potential of undermining both fairness and detection performance in abusive language detection. In a dynamic and complex digital world, it is crucial to investigate the vulnerabilities of these detection models to adversarial fairness attacks to improve their fairness robustness. We propose a simple yet effective framework FABLE that leverages backdoor attacks as they allow targeted control over the fairness and detection performance. FABLE explores three types of trigger designs (i.e., rare, artificial, and natural triggers) and novel sampling strategies. Specifically, the adversary can inject triggers into samples in the minority group with the favored outcome (i.e., "non-abusive") and flip their labels to the unfavored outcome, i.e., "abusive". Experiments on benchmark datasets demonstrate the effectiveness of FABLE attacking fairness and utility in abusive language detection.



## **36. ScAR: Scaling Adversarial Robustness for LiDAR Object Detection**

cs.CV

**SubmitDate**: 2023-12-05    [abs](http://arxiv.org/abs/2312.03085v1) [paper-pdf](http://arxiv.org/pdf/2312.03085v1)

**Authors**: Xiaohu Lu, Hayder Radha

**Abstract**: The adversarial robustness of a model is its ability to resist adversarial attacks in the form of small perturbations to input data. Universal adversarial attack methods such as Fast Sign Gradient Method (FSGM) and Projected Gradient Descend (PGD) are popular for LiDAR object detection, but they are often deficient compared to task-specific adversarial attacks. Additionally, these universal methods typically require unrestricted access to the model's information, which is difficult to obtain in real-world applications. To address these limitations, we present a black-box Scaling Adversarial Robustness (ScAR) method for LiDAR object detection. By analyzing the statistical characteristics of 3D object detection datasets such as KITTI, Waymo, and nuScenes, we have found that the model's prediction is sensitive to scaling of 3D instances. We propose three black-box scaling adversarial attack methods based on the available information: model-aware attack, distribution-aware attack, and blind attack. We also introduce a strategy for generating scaling adversarial examples to improve the model's robustness against these three scaling adversarial attacks. Comparison with other methods on public datasets under different 3D object detection architectures demonstrates the effectiveness of our proposed method.



## **37. Realistic Scatterer Based Adversarial Attacks on SAR Image Classifiers**

cs.CV

**SubmitDate**: 2023-12-05    [abs](http://arxiv.org/abs/2312.02912v1) [paper-pdf](http://arxiv.org/pdf/2312.02912v1)

**Authors**: Tian Ye, Rajgopal Kannan, Viktor Prasanna, Carl Busart, Lance Kaplan

**Abstract**: Adversarial attacks have highlighted the vulnerability of classifiers based on machine learning for Synthetic Aperture Radar (SAR) Automatic Target Recognition (ATR) tasks. An adversarial attack perturbs SAR images of on-ground targets such that the classifiers are misled into making incorrect predictions. However, many existing attacking techniques rely on arbitrary manipulation of SAR images while overlooking the feasibility of executing the attacks on real-world SAR imagery. Instead, adversarial attacks should be able to be implemented by physical actions, for example, placing additional false objects as scatterers around the on-ground target to perturb the SAR image and fool the SAR ATR.   In this paper, we propose the On-Target Scatterer Attack (OTSA), a scatterer-based physical adversarial attack. To ensure the feasibility of its physical execution, we enforce a constraint on the positioning of the scatterers. Specifically, we restrict the scatterers to be placed only on the target instead of in the shadow regions or the background. To achieve this, we introduce a positioning score based on Gaussian kernels and formulate an optimization problem for our OTSA attack. Using a gradient ascent method to solve the optimization problem, the OTSA can generate a vector of parameters describing the positions, shapes, sizes and amplitudes of the scatterers to guide the physical execution of the attack that will mislead SAR image classifiers. The experimental results show that our attack obtains significantly higher success rates under the positioning constraint compared with the existing method.



## **38. Scaling Laws for Adversarial Attacks on Language Model Activations**

cs.LG

15 pages, 9 figures

**SubmitDate**: 2023-12-05    [abs](http://arxiv.org/abs/2312.02780v1) [paper-pdf](http://arxiv.org/pdf/2312.02780v1)

**Authors**: Stanislav Fort

**Abstract**: We explore a class of adversarial attacks targeting the activations of language models. By manipulating a relatively small subset of model activations, $a$, we demonstrate the ability to control the exact prediction of a significant number (in some cases up to 1000) of subsequent tokens $t$. We empirically verify a scaling law where the maximum number of target tokens $t_\mathrm{max}$ predicted depends linearly on the number of tokens $a$ whose activations the attacker controls as $t_\mathrm{max} = \kappa a$. We find that the number of bits of control in the input space needed to control a single bit in the output space (what we call attack resistance $\chi$) is remarkably constant between $\approx 16$ and $\approx 25$ over 2 orders of magnitude of model sizes for different language models. Compared to attacks on tokens, attacks on activations are predictably much stronger, however, we identify a surprising regularity where one bit of input steered either via activations or via tokens is able to exert control over a similar amount of output bits. This gives support for the hypothesis that adversarial attacks are a consequence of dimensionality mismatch between the input and output spaces. A practical implication of the ease of attacking language model activations instead of tokens is for multi-modal and selected retrieval models, where additional data sources are added as activations directly, sidestepping the tokenized input. This opens up a new, broad attack surface. By using language models as a controllable test-bed to study adversarial attacks, we were able to experiment with input-output dimensions that are inaccessible in computer vision, especially where the output dimension dominates.



## **39. Generating Visually Realistic Adversarial Patch**

cs.CV

14 pages

**SubmitDate**: 2023-12-05    [abs](http://arxiv.org/abs/2312.03030v1) [paper-pdf](http://arxiv.org/pdf/2312.03030v1)

**Authors**: Xiaosen Wang, Kunyu Wang

**Abstract**: Deep neural networks (DNNs) are vulnerable to various types of adversarial examples, bringing huge threats to security-critical applications. Among these, adversarial patches have drawn increasing attention due to their good applicability to fool DNNs in the physical world. However, existing works often generate patches with meaningless noise or patterns, making it conspicuous to humans. To address this issue, we explore how to generate visually realistic adversarial patches to fool DNNs. Firstly, we analyze that a high-quality adversarial patch should be realistic, position irrelevant, and printable to be deployed in the physical world. Based on this analysis, we propose an effective attack called VRAP, to generate visually realistic adversarial patches. Specifically, VRAP constrains the patch in the neighborhood of a real image to ensure the visual reality, optimizes the patch at the poorest position for position irrelevance, and adopts Total Variance loss as well as gamma transformation to make the generated patch printable without losing information. Empirical evaluations on the ImageNet dataset demonstrate that the proposed VRAP exhibits outstanding attack performance in the digital world. Moreover, the generated adversarial patches can be disguised as the scrawl or logo in the physical world to fool the deep models without being detected, bringing significant threats to DNNs-enabled applications.



## **40. Byzantine-Robust Distributed Online Learning: Taming Adversarial Participants in An Adversarial Environment**

cs.LG

**SubmitDate**: 2023-12-05    [abs](http://arxiv.org/abs/2307.07980v3) [paper-pdf](http://arxiv.org/pdf/2307.07980v3)

**Authors**: Xingrong Dong, Zhaoxian Wu, Qing Ling, Zhi Tian

**Abstract**: This paper studies distributed online learning under Byzantine attacks. The performance of an online learning algorithm is often characterized by (adversarial) regret, which evaluates the quality of one-step-ahead decision-making when an environment provides adversarial losses, and a sublinear bound is preferred. But we prove that, even with a class of state-of-the-art robust aggregation rules, in an adversarial environment and in the presence of Byzantine participants, distributed online gradient descent can only achieve a linear adversarial regret bound, which is tight. This is the inevitable consequence of Byzantine attacks, even though we can control the constant of the linear adversarial regret to a reasonable level. Interestingly, when the environment is not fully adversarial so that the losses of the honest participants are i.i.d. (independent and identically distributed), we show that sublinear stochastic regret, in contrast to the aforementioned adversarial regret, is possible. We develop a Byzantine-robust distributed online momentum algorithm to attain such a sublinear stochastic regret bound. Extensive numerical experiments corroborate our theoretical analysis.



## **41. FedBayes: A Zero-Trust Federated Learning Aggregation to Defend Against Adversarial Attacks**

cs.CR

Accepted to IEEE CCWC 2024

**SubmitDate**: 2023-12-04    [abs](http://arxiv.org/abs/2312.04587v1) [paper-pdf](http://arxiv.org/pdf/2312.04587v1)

**Authors**: Marc Vucovich, Devin Quinn, Kevin Choi, Christopher Redino, Abdul Rahman, Edward Bowen

**Abstract**: Federated learning has created a decentralized method to train a machine learning model without needing direct access to client data. The main goal of a federated learning architecture is to protect the privacy of each client while still contributing to the training of the global model. However, the main advantage of privacy in federated learning is also the easiest aspect to exploit. Without being able to see the clients' data, it is difficult to determine the quality of the data. By utilizing data poisoning methods, such as backdoor or label-flipping attacks, or by sending manipulated information about their data back to the server, malicious clients are able to corrupt the global model and degrade performance across all clients within a federation. Our novel aggregation method, FedBayes, mitigates the effect of a malicious client by calculating the probabilities of a client's model weights given to the prior model's weights using Bayesian statistics. Our results show that this approach negates the effects of malicious clients and protects the overall federation.



## **42. InstructTA: Instruction-Tuned Targeted Attack for Large Vision-Language Models**

cs.CV

**SubmitDate**: 2023-12-04    [abs](http://arxiv.org/abs/2312.01886v1) [paper-pdf](http://arxiv.org/pdf/2312.01886v1)

**Authors**: Xunguang Wang, Zhenlan Ji, Pingchuan Ma, Zongjie Li, Shuai Wang

**Abstract**: Large vision-language models (LVLMs) have demonstrated their incredible capability in image understanding and response generation. However, this rich visual interaction also makes LVLMs vulnerable to adversarial examples. In this paper, we formulate a novel and practical gray-box attack scenario that the adversary can only access the visual encoder of the victim LVLM, without the knowledge of its prompts (which are often proprietary for service providers and not publicly available) and its underlying large language model (LLM). This practical setting poses challenges to the cross-prompt and cross-model transferability of targeted adversarial attack, which aims to confuse the LVLM to output a response that is semantically similar to the attacker's chosen target text. To this end, we propose an instruction-tuned targeted attack (dubbed InstructTA) to deliver the targeted adversarial attack on LVLMs with high transferability. Initially, we utilize a public text-to-image generative model to "reverse" the target response into a target image, and employ GPT-4 to infer a reasonable instruction $\boldsymbol{p}^\prime$ from the target response. We then form a local surrogate model (sharing the same visual encoder with the victim LVLM) to extract instruction-aware features of an adversarial image example and the target image, and minimize the distance between these two features to optimize the adversarial example. To further improve the transferability, we augment the instruction $\boldsymbol{p}^\prime$ with instructions paraphrased from an LLM. Extensive experiments demonstrate the superiority of our proposed method in targeted attack performance and transferability.



## **43. Two-stage optimized unified adversarial patch for attacking visible-infrared cross-modal detectors in the physical world**

cs.CV

**SubmitDate**: 2023-12-04    [abs](http://arxiv.org/abs/2312.01789v1) [paper-pdf](http://arxiv.org/pdf/2312.01789v1)

**Authors**: Chengyin Hu, Weiwen Shi

**Abstract**: Currently, many studies have addressed security concerns related to visible and infrared detectors independently. In practical scenarios, utilizing cross-modal detectors for tasks proves more reliable than relying on single-modal detectors. Despite this, there is a lack of comprehensive security evaluations for cross-modal detectors. While existing research has explored the feasibility of attacks against cross-modal detectors, the implementation of a robust attack remains unaddressed. This work introduces the Two-stage Optimized Unified Adversarial Patch (TOUAP) designed for performing attacks against visible-infrared cross-modal detectors in real-world, black-box settings. The TOUAP employs a two-stage optimization process: firstly, PSO optimizes an irregular polygonal infrared patch to attack the infrared detector; secondly, the color QR code is optimized, and the shape information of the infrared patch from the first stage is used as a mask. The resulting irregular polygon visible modal patch executes an attack on the visible detector. Through extensive experiments conducted in both digital and physical environments, we validate the effectiveness and robustness of the proposed method. As the TOUAP surpasses baseline performance, we advocate for its widespread attention.



## **44. Singular Regularization with Information Bottleneck Improves Model's Adversarial Robustness**

cs.CV

**SubmitDate**: 2023-12-04    [abs](http://arxiv.org/abs/2312.02237v1) [paper-pdf](http://arxiv.org/pdf/2312.02237v1)

**Authors**: Guanlin Li, Naishan Zheng, Man Zhou, Jie Zhang, Tianwei Zhang

**Abstract**: Adversarial examples are one of the most severe threats to deep learning models. Numerous works have been proposed to study and defend adversarial examples. However, these works lack analysis of adversarial information or perturbation, which cannot reveal the mystery of adversarial examples and lose proper interpretation. In this paper, we aim to fill this gap by studying adversarial information as unstructured noise, which does not have a clear pattern. Specifically, we provide some empirical studies with singular value decomposition, by decomposing images into several matrices, to analyze adversarial information for different attacks. Based on the analysis, we propose a new module to regularize adversarial information and combine information bottleneck theory, which is proposed to theoretically restrict intermediate representations. Therefore, our method is interpretable. Moreover, the fashion of our design is a novel principle that is general and unified. Equipped with our new module, we evaluate two popular model structures on two mainstream datasets with various adversarial attacks. The results indicate that the improvement in robust accuracy is significant. On the other hand, we prove that our method is efficient with only a few additional parameters and able to be explained under regional faithfulness analysis.



## **45. Warfare:Breaking the Watermark Protection of AI-Generated Content**

cs.CV

**SubmitDate**: 2023-12-04    [abs](http://arxiv.org/abs/2310.07726v2) [paper-pdf](http://arxiv.org/pdf/2310.07726v2)

**Authors**: Guanlin Li, Yifei Chen, Jie Zhang, Jiwei Li, Shangwei Guo, Tianwei Zhang

**Abstract**: AI-Generated Content (AIGC) is gaining great popularity, with many emerging commercial services and applications. These services leverage advanced generative models, such as latent diffusion models and large language models, to generate creative content (e.g., realistic images and fluent sentences) for users. The usage of such generated content needs to be highly regulated, as the service providers need to ensure the users do not violate the usage policies (e.g., abuse for commercialization, generating and distributing unsafe content). A promising solution to achieve this goal is watermarking, which adds unique and imperceptible watermarks on the content for service verification and attribution. Numerous watermarking approaches have been proposed recently. However, in this paper, we show that an adversary can easily break these watermarking mechanisms. Specifically, we consider two possible attacks. (1) Watermark removal: the adversary can easily erase the embedded watermark from the generated content and then use it freely bypassing the regulation of the service provider. (2) Watermark forging: the adversary can create illegal content with forged watermarks from another user, causing the service provider to make wrong attributions. We propose Warfare, a unified methodology to achieve both attacks in a holistic way. The key idea is to leverage a pre-trained diffusion model for content processing and a generative adversarial network for watermark removal or forging. We evaluate Warfare on different datasets and embedding setups. The results prove that it can achieve high success rates while maintaining the quality of the generated content. Compared to existing diffusion model-based attacks, Warfare is 5,050~11,000x faster.



## **46. Malicious Lateral Movement in 5G Core With Network Slicing And Its Detection**

cs.CR

Accepted for publication in the Proceedings of IEEE ITNAC-2023

**SubmitDate**: 2023-12-04    [abs](http://arxiv.org/abs/2312.01681v1) [paper-pdf](http://arxiv.org/pdf/2312.01681v1)

**Authors**: Ayush Kumar, Vrizlynn L. L. Thing

**Abstract**: 5G networks are susceptible to cyber attacks due to reasons such as implementation issues and vulnerabilities in 3GPP standard specifications. In this work, we propose lateral movement strategies in a 5G Core (5GC) with network slicing enabled, as part of a larger attack campaign by well-resourced adversaries such as APT groups. Further, we present 5GLatte, a system to detect such malicious lateral movement. 5GLatte operates on a host-container access graph built using host/NF container logs collected from the 5GC. Paths inferred from the access graph are scored based on selected filtering criteria and subsequently presented as input to a threshold-based anomaly detection algorithm to reveal malicious lateral movement paths. We evaluate 5GLatte on a dataset containing attack campaigns (based on MITRE ATT&CK and FiGHT frameworks) launched in a 5G test environment which shows that compared to other lateral movement detectors based on state-of-the-art, it can achieve higher true positive rates with similar false positive rates.



## **47. Adversarial Medical Image with Hierarchical Feature Hiding**

eess.IV

Our code is available at  \url{https://github.com/qsyao/Hierarchical_Feature_Constraint}. arXiv admin  note: text overlap with arXiv:2012.09501

**SubmitDate**: 2023-12-04    [abs](http://arxiv.org/abs/2312.01679v1) [paper-pdf](http://arxiv.org/pdf/2312.01679v1)

**Authors**: Qingsong Yao, Zecheng He, Yuexiang Li, Yi Lin, Kai Ma, Yefeng Zheng, S. Kevin Zhou

**Abstract**: Deep learning based methods for medical images can be easily compromised by adversarial examples (AEs), posing a great security flaw in clinical decision-making. It has been discovered that conventional adversarial attacks like PGD which optimize the classification logits, are easy to distinguish in the feature space, resulting in accurate reactive defenses. To better understand this phenomenon and reassess the reliability of the reactive defenses for medical AEs, we thoroughly investigate the characteristic of conventional medical AEs. Specifically, we first theoretically prove that conventional adversarial attacks change the outputs by continuously optimizing vulnerable features in a fixed direction, thereby leading to outlier representations in the feature space. Then, a stress test is conducted to reveal the vulnerability of medical images, by comparing with natural images. Interestingly, this vulnerability is a double-edged sword, which can be exploited to hide AEs. We then propose a simple-yet-effective hierarchical feature constraint (HFC), a novel add-on to conventional white-box attacks, which assists to hide the adversarial feature in the target feature distribution. The proposed method is evaluated on three medical datasets, both 2D and 3D, with different modalities. The experimental results demonstrate the superiority of HFC, \emph{i.e.,} it bypasses an array of state-of-the-art adversarial medical AE detectors more efficiently than competing adaptive attacks, which reveals the deficiencies of medical reactive defense and allows to develop more robust defenses in future.



## **48. The Queen's Guard: A Secure Enforcement of Fine-grained Access Control In Distributed Data Analytics Platforms**

cs.CR

**SubmitDate**: 2023-12-04    [abs](http://arxiv.org/abs/2106.13123v4) [paper-pdf](http://arxiv.org/pdf/2106.13123v4)

**Authors**: Fahad Shaon, Sazzadur Rahaman, Murat Kantarcioglu

**Abstract**: Distributed data analytics platforms (i.e., Apache Spark, Hadoop) provide high-level APIs to programmatically write analytics tasks that are run distributedly in multiple computing nodes. The design of these frameworks was primarily motivated by performance and usability. Thus, the security takes a back seat. Consequently, they do not inherently support fine-grained access control or offer any plugin mechanism to enable it, making them risky to be used in multi-tier organizational settings.   There have been attempts to build "add-on" solutions to enable fine-grained access control for distributed data analytics platforms. In this paper, first, we show that straightforward enforcement of ``add-on'' access control is insecure under adversarial code execution. Specifically, we show that an attacker can abuse platform-provided APIs to evade access controls without leaving any traces. Second, we designed a two-layered (i.e., proactive and reactive) defense system to protect against API abuses. On submission of a user code, our proactive security layer statically screens it to find potential attack signatures prior to its execution. The reactive security layer employs code instrumentation-based runtime checks and sandboxed execution to throttle any exploits at runtime. Next, we propose a new fine-grained access control framework with an enhanced policy language that supports map and filter primitives. Finally, we build a system named SecureDL with our new access control framework and defense system on top of Apache Spark, which ensures secure access control policy enforcement under adversaries capable of executing code.   To the best of our knowledge, this is the first fine-grained attribute-based access control framework for distributed data analytics platforms that is secure against platform API abuse attacks. Performance evaluation showed that the overhead due to added security is low.



## **49. Robust Evaluation of Diffusion-Based Adversarial Purification**

cs.CV

Accepted by ICCV 2023, oral presentation. Code is available at  https://github.com/ml-postech/robust-evaluation-of-diffusion-based-purification

**SubmitDate**: 2023-12-03    [abs](http://arxiv.org/abs/2303.09051v3) [paper-pdf](http://arxiv.org/pdf/2303.09051v3)

**Authors**: Minjong Lee, Dongwoo Kim

**Abstract**: We question the current evaluation practice on diffusion-based purification methods. Diffusion-based purification methods aim to remove adversarial effects from an input data point at test time. The approach gains increasing attention as an alternative to adversarial training due to the disentangling between training and testing. Well-known white-box attacks are often employed to measure the robustness of the purification. However, it is unknown whether these attacks are the most effective for the diffusion-based purification since the attacks are often tailored for adversarial training. We analyze the current practices and provide a new guideline for measuring the robustness of purification methods against adversarial attacks. Based on our analysis, we further propose a new purification strategy improving robustness compared to the current diffusion-based purification methods.



## **50. QuantAttack: Exploiting Dynamic Quantization to Attack Vision Transformers**

cs.CV

**SubmitDate**: 2023-12-03    [abs](http://arxiv.org/abs/2312.02220v1) [paper-pdf](http://arxiv.org/pdf/2312.02220v1)

**Authors**: Amit Baras, Alon Zolfi, Yuval Elovici, Asaf Shabtai

**Abstract**: In recent years, there has been a significant trend in deep neural networks (DNNs), particularly transformer-based models, of developing ever-larger and more capable models. While they demonstrate state-of-the-art performance, their growing scale requires increased computational resources (e.g., GPUs with greater memory capacity). To address this problem, quantization techniques (i.e., low-bit-precision representation and matrix multiplication) have been proposed. Most quantization techniques employ a static strategy in which the model parameters are quantized, either during training or inference, without considering the test-time sample. In contrast, dynamic quantization techniques, which have become increasingly popular, adapt during inference based on the input provided, while maintaining full-precision performance. However, their dynamic behavior and average-case performance assumption makes them vulnerable to a novel threat vector -- adversarial attacks that target the model's efficiency and availability. In this paper, we present QuantAttack, a novel attack that targets the availability of quantized models, slowing down the inference, and increasing memory usage and energy consumption. We show that carefully crafted adversarial examples, which are designed to exhaust the resources of the operating system, can trigger worst-case performance. In our experiments, we demonstrate the effectiveness of our attack on vision transformers on a wide range of tasks, both uni-modal and multi-modal. We also examine the effect of different attack variants (e.g., a universal perturbation) and the transferability between different models.



