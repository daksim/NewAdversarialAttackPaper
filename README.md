# Latest Adversarial Attack Papers
**update at 2024-07-09 10:23:40**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. Improving Alignment and Robustness with Circuit Breakers**

cs.LG

**SubmitDate**: 2024-07-08    [abs](http://arxiv.org/abs/2406.04313v3) [paper-pdf](http://arxiv.org/pdf/2406.04313v3)

**Authors**: Andy Zou, Long Phan, Justin Wang, Derek Duenas, Maxwell Lin, Maksym Andriushchenko, Rowan Wang, Zico Kolter, Matt Fredrikson, Dan Hendrycks

**Abstract**: AI systems can take harmful actions and are highly vulnerable to adversarial attacks. We present an approach, inspired by recent advances in representation engineering, that interrupts the models as they respond with harmful outputs with "circuit breakers." Existing techniques aimed at improving alignment, such as refusal training, are often bypassed. Techniques such as adversarial training try to plug these holes by countering specific attacks. As an alternative to refusal training and adversarial training, circuit-breaking directly controls the representations that are responsible for harmful outputs in the first place. Our technique can be applied to both text-only and multimodal language models to prevent the generation of harmful outputs without sacrificing utility -- even in the presence of powerful unseen attacks. Notably, while adversarial robustness in standalone image recognition remains an open challenge, circuit breakers allow the larger multimodal system to reliably withstand image "hijacks" that aim to produce harmful content. Finally, we extend our approach to AI agents, demonstrating considerable reductions in the rate of harmful actions when they are under attack. Our approach represents a significant step forward in the development of reliable safeguards to harmful behavior and adversarial attacks.



## **2. Adaptive and robust watermark against model extraction attack**

cs.CR

**SubmitDate**: 2024-07-08    [abs](http://arxiv.org/abs/2405.02365v2) [paper-pdf](http://arxiv.org/pdf/2405.02365v2)

**Authors**: Kaiyi Pang

**Abstract**: Large language models (LLMs) demonstrate general intelligence across a variety of machine learning tasks, thereby enhancing the commercial value of their intellectual property (IP). To protect this IP, model owners typically allow user access only in a black-box manner, however, adversaries can still utilize model extraction attacks to steal the model intelligence encoded in model generation. Watermarking technology offers a promising solution for defending against such attacks by embedding unique identifiers into the model-generated content. However, existing watermarking methods often compromise the quality of generated content due to heuristic alterations and lack robust mechanisms to counteract adversarial strategies, thus limiting their practicality in real-world scenarios. In this paper, we introduce an adaptive and robust watermarking method (named ModelShield) to protect the IP of LLMs. Our method incorporates a self-watermarking mechanism that allows LLMs to autonomously insert watermarks into their generated content to avoid the degradation of model content. We also propose a robust watermark detection mechanism capable of effectively identifying watermark signals under the interference of varying adversarial strategies. Besides, ModelShield is a plug-and-play method that does not require additional model training, enhancing its applicability in LLM deployments. Extensive evaluations on two real-world datasets and three LLMs demonstrate that our method surpasses existing methods in terms of defense effectiveness and robustness while significantly reducing the degradation of watermarking on the model-generated content.



## **3. Multi-View Black-Box Physical Attacks on Infrared Pedestrian Detectors Using Adversarial Infrared Grid**

cs.CV

**SubmitDate**: 2024-07-08    [abs](http://arxiv.org/abs/2407.01168v2) [paper-pdf](http://arxiv.org/pdf/2407.01168v2)

**Authors**: Kalibinuer Tiliwalidi, Chengyin Hu, Weiwen Shi

**Abstract**: While extensive research exists on physical adversarial attacks within the visible spectrum, studies on such techniques in the infrared spectrum are limited. Infrared object detectors are vital in modern technological applications but are susceptible to adversarial attacks, posing significant security threats. Previous studies using physical perturbations like light bulb arrays and aerogels for white-box attacks, or hot and cold patches for black-box attacks, have proven impractical or limited in multi-view support. To address these issues, we propose the Adversarial Infrared Grid (AdvGrid), which models perturbations in a grid format and uses a genetic algorithm for black-box optimization. These perturbations are cyclically applied to various parts of a pedestrian's clothing to facilitate multi-view black-box physical attacks on infrared pedestrian detectors. Extensive experiments validate AdvGrid's effectiveness, stealthiness, and robustness. The method achieves attack success rates of 80.00\% in digital environments and 91.86\% in physical environments, outperforming baseline methods. Additionally, the average attack success rate exceeds 50\% against mainstream detectors, demonstrating AdvGrid's robustness. Our analyses include ablation studies, transfer attacks, and adversarial defenses, confirming the method's superiority.



## **4. Malicious Agent Detection for Robust Multi-Agent Collaborative Perception**

cs.CR

Accepted by IROS 2024

**SubmitDate**: 2024-07-08    [abs](http://arxiv.org/abs/2310.11901v2) [paper-pdf](http://arxiv.org/pdf/2310.11901v2)

**Authors**: Yangheng Zhao, Zhen Xiang, Sheng Yin, Xianghe Pang, Siheng Chen, Yanfeng Wang

**Abstract**: Recently, multi-agent collaborative (MAC) perception has been proposed and outperformed the traditional single-agent perception in many applications, such as autonomous driving. However, MAC perception is more vulnerable to adversarial attacks than single-agent perception due to the information exchange. The attacker can easily degrade the performance of a victim agent by sending harmful information from a malicious agent nearby. In this paper, we extend adversarial attacks to an important perception task -- MAC object detection, where generic defenses such as adversarial training are no longer effective against these attacks. More importantly, we propose Malicious Agent Detection (MADE), a reactive defense specific to MAC perception that can be deployed by each agent to accurately detect and then remove any potential malicious agent in its local collaboration network. In particular, MADE inspects each agent in the network independently using a semi-supervised anomaly detector based on a double-hypothesis test with the Benjamini-Hochberg procedure to control the false positive rate of the inference. For the two hypothesis tests, we propose a match loss statistic and a collaborative reconstruction loss statistic, respectively, both based on the consistency between the agent to be inspected and the ego agent where our detector is deployed. We conduct comprehensive evaluations on a benchmark 3D dataset V2X-sim and a real-road dataset DAIR-V2X and show that with the protection of MADE, the drops in the average precision compared with the best-case "oracle" defender against our attack are merely 1.28% and 0.34%, respectively, much lower than 8.92% and 10.00% for adversarial training, respectively.



## **5. Formal Verification of Object Detection**

cs.CV

**SubmitDate**: 2024-07-08    [abs](http://arxiv.org/abs/2407.01295v2) [paper-pdf](http://arxiv.org/pdf/2407.01295v2)

**Authors**: Avraham Raviv, Yizhak Y. Elboher, Michelle Aluf-Medina, Yael Leibovich Weiss, Omer Cohen, Roy Assa, Guy Katz, Hillel Kugler

**Abstract**: Deep Neural Networks (DNNs) are ubiquitous in real-world applications, yet they remain vulnerable to errors and adversarial attacks. This work tackles the challenge of applying formal verification to ensure the safety of computer vision models, extending verification beyond image classification to object detection. We propose a general formulation for certifying the robustness of object detection models using formal verification and outline implementation strategies compatible with state-of-the-art verification tools. Our approach enables the application of these tools, originally designed for verifying classification models, to object detection. We define various attacks for object detection, illustrating the diverse ways adversarial inputs can compromise neural network outputs. Our experiments, conducted on several common datasets and networks, reveal potential errors in object detection models, highlighting system vulnerabilities and emphasizing the need for expanding formal verification to these new domains. This work paves the way for further research in integrating formal verification across a broader range of computer vision applications.



## **6. Exploring the Adversarial Capabilities of Large Language Models**

cs.AI

**SubmitDate**: 2024-07-08    [abs](http://arxiv.org/abs/2402.09132v4) [paper-pdf](http://arxiv.org/pdf/2402.09132v4)

**Authors**: Lukas Struppek, Minh Hieu Le, Dominik Hintersdorf, Kristian Kersting

**Abstract**: The proliferation of large language models (LLMs) has sparked widespread and general interest due to their strong language generation capabilities, offering great potential for both industry and research. While previous research delved into the security and privacy issues of LLMs, the extent to which these models can exhibit adversarial behavior remains largely unexplored. Addressing this gap, we investigate whether common publicly available LLMs have inherent capabilities to perturb text samples to fool safety measures, so-called adversarial examples resp.~attacks. More specifically, we investigate whether LLMs are inherently able to craft adversarial examples out of benign samples to fool existing safe rails. Our experiments, which focus on hate speech detection, reveal that LLMs succeed in finding adversarial perturbations, effectively undermining hate speech detection systems. Our findings carry significant implications for (semi-)autonomous systems relying on LLMs, highlighting potential challenges in their interaction with existing systems and safety measures.



## **7. Improving Adversarial Transferability of Vision-Language Pre-training Models through Collaborative Multimodal Interaction**

cs.CV

This work won first place in CVPR 2024 Workshop Challenge: Black-box  Adversarial Attacks on Vision Foundation Models

**SubmitDate**: 2024-07-08    [abs](http://arxiv.org/abs/2403.10883v2) [paper-pdf](http://arxiv.org/pdf/2403.10883v2)

**Authors**: Jiyuan Fu, Zhaoyu Chen, Kaixun Jiang, Haijing Guo, Jiafeng Wang, Shuyong Gao, Wenqiang Zhang

**Abstract**: Despite the substantial advancements in Vision-Language Pre-training (VLP) models, their susceptibility to adversarial attacks poses a significant challenge. Existing work rarely studies the transferability of attacks on VLP models, resulting in a substantial performance gap from white-box attacks. We observe that prior work overlooks the interaction mechanisms between modalities, which plays a crucial role in understanding the intricacies of VLP models. In response, we propose a novel attack, called Collaborative Multimodal Interaction Attack (CMI-Attack), leveraging modality interaction through embedding guidance and interaction enhancement. Specifically, attacking text at the embedding level while preserving semantics, as well as utilizing interaction image gradients to enhance constraints on perturbations of texts and images. Significantly, in the image-text retrieval task on Flickr30K dataset, CMI-Attack raises the transfer success rates from ALBEF to TCL, $\text{CLIP}_{\text{ViT}}$ and $\text{CLIP}_{\text{CNN}}$ by 8.11%-16.75% over state-of-the-art methods. Moreover, CMI-Attack also demonstrates superior performance in cross-task generalization scenarios. Our work addresses the underexplored realm of transfer attacks on VLP models, shedding light on the importance of modality interaction for enhanced adversarial robustness.



## **8. A Survey of Fragile Model Watermarking**

cs.CR

Submitted Signal Processing

**SubmitDate**: 2024-07-08    [abs](http://arxiv.org/abs/2406.04809v4) [paper-pdf](http://arxiv.org/pdf/2406.04809v4)

**Authors**: Zhenzhe Gao, Yu Cheng, Zhaoxia Yin

**Abstract**: Model fragile watermarking, inspired by both the field of adversarial attacks on neural networks and traditional multimedia fragile watermarking, has gradually emerged as a potent tool for detecting tampering, and has witnessed rapid development in recent years. Unlike robust watermarks, which are widely used for identifying model copyrights, fragile watermarks for models are designed to identify whether models have been subjected to unexpected alterations such as backdoors, poisoning, compression, among others. These alterations can pose unknown risks to model users, such as misidentifying stop signs as speed limit signs in classic autonomous driving scenarios. This paper provides an overview of the relevant work in the field of model fragile watermarking since its inception, categorizing them and revealing the developmental trajectory of the field, thus offering a comprehensive survey for future endeavors in model fragile watermarking.



## **9. To Generate or Not? Safety-Driven Unlearned Diffusion Models Are Still Easy To Generate Unsafe Images ... For Now**

cs.CV

Accepted by ECCV'24. Codes are available at  https://github.com/OPTML-Group/Diffusion-MU-Attack

**SubmitDate**: 2024-07-07    [abs](http://arxiv.org/abs/2310.11868v4) [paper-pdf](http://arxiv.org/pdf/2310.11868v4)

**Authors**: Yimeng Zhang, Jinghan Jia, Xin Chen, Aochuan Chen, Yihua Zhang, Jiancheng Liu, Ke Ding, Sijia Liu

**Abstract**: The recent advances in diffusion models (DMs) have revolutionized the generation of realistic and complex images. However, these models also introduce potential safety hazards, such as producing harmful content and infringing data copyrights. Despite the development of safety-driven unlearning techniques to counteract these challenges, doubts about their efficacy persist. To tackle this issue, we introduce an evaluation framework that leverages adversarial prompts to discern the trustworthiness of these safety-driven DMs after they have undergone the process of unlearning harmful concepts. Specifically, we investigated the adversarial robustness of DMs, assessed by adversarial prompts, when eliminating unwanted concepts, styles, and objects. We develop an effective and efficient adversarial prompt generation approach for DMs, termed UnlearnDiffAtk. This method capitalizes on the intrinsic classification abilities of DMs to simplify the creation of adversarial prompts, thereby eliminating the need for auxiliary classification or diffusion models. Through extensive benchmarking, we evaluate the robustness of widely-used safety-driven unlearned DMs (i.e., DMs after unlearning undesirable concepts, styles, or objects) across a variety of tasks. Our results demonstrate the effectiveness and efficiency merits of UnlearnDiffAtk over the state-of-the-art adversarial prompt generation method and reveal the lack of robustness of current safetydriven unlearning techniques when applied to DMs. Codes are available at https://github.com/OPTML-Group/Diffusion-MU-Attack. WARNING: There exist AI generations that may be offensive in nature.



## **10. Rethinking Targeted Adversarial Attacks For Neural Machine Translation**

cs.CL

5 pages, 2 figures, accepted by ICASSP 2024

**SubmitDate**: 2024-07-07    [abs](http://arxiv.org/abs/2407.05319v1) [paper-pdf](http://arxiv.org/pdf/2407.05319v1)

**Authors**: Junjie Wu, Lemao Liu, Wei Bi, Dit-Yan Yeung

**Abstract**: Targeted adversarial attacks are widely used to evaluate the robustness of neural machine translation systems. Unfortunately, this paper first identifies a critical issue in the existing settings of NMT targeted adversarial attacks, where their attacking results are largely overestimated. To this end, this paper presents a new setting for NMT targeted adversarial attacks that could lead to reliable attacking results. Under the new setting, it then proposes a Targeted Word Gradient adversarial Attack (TWGA) method to craft adversarial examples. Experimental results demonstrate that our proposed setting could provide faithful attacking results for targeted adversarial attacks on NMT systems, and the proposed TWGA method can effectively attack such victim NMT systems. In-depth analyses on a large-scale dataset further illustrate some valuable findings. 1 Our code and data are available at https://github.com/wujunjie1998/TWGA.



## **11. TrojanRAG: Retrieval-Augmented Generation Can Be Backdoor Driver in Large Language Models**

cs.CR

19 pages, 14 figures, 4 tables

**SubmitDate**: 2024-07-07    [abs](http://arxiv.org/abs/2405.13401v4) [paper-pdf](http://arxiv.org/pdf/2405.13401v4)

**Authors**: Pengzhou Cheng, Yidong Ding, Tianjie Ju, Zongru Wu, Wei Du, Ping Yi, Zhuosheng Zhang, Gongshen Liu

**Abstract**: Large language models (LLMs) have raised concerns about potential security threats despite performing significantly in Natural Language Processing (NLP). Backdoor attacks initially verified that LLM is doing substantial harm at all stages, but the cost and robustness have been criticized. Attacking LLMs is inherently risky in security review, while prohibitively expensive. Besides, the continuous iteration of LLMs will degrade the robustness of backdoors. In this paper, we propose TrojanRAG, which employs a joint backdoor attack in the Retrieval-Augmented Generation, thereby manipulating LLMs in universal attack scenarios. Specifically, the adversary constructs elaborate target contexts and trigger sets. Multiple pairs of backdoor shortcuts are orthogonally optimized by contrastive learning, thus constraining the triggering conditions to a parameter subspace to improve the matching. To improve the recall of the RAG for the target contexts, we introduce a knowledge graph to construct structured data to achieve hard matching at a fine-grained level. Moreover, we normalize the backdoor scenarios in LLMs to analyze the real harm caused by backdoors from both attackers' and users' perspectives and further verify whether the context is a favorable tool for jailbreaking models. Extensive experimental results on truthfulness, language understanding, and harmfulness show that TrojanRAG exhibits versatility threats while maintaining retrieval capabilities on normal queries.



## **12. FedCG: Leverage Conditional GAN for Protecting Privacy and Maintaining Competitive Performance in Federated Learning**

cs.LG

**SubmitDate**: 2024-07-07    [abs](http://arxiv.org/abs/2111.08211v3) [paper-pdf](http://arxiv.org/pdf/2111.08211v3)

**Authors**: Yuezhou Wu, Yan Kang, Jiahuan Luo, Yuanqin He, Qiang Yang

**Abstract**: Federated learning (FL) aims to protect data privacy by enabling clients to build machine learning models collaboratively without sharing their private data. Recent works demonstrate that information exchanged during FL is subject to gradient-based privacy attacks, and consequently, a variety of privacy-preserving methods have been adopted to thwart such attacks. However, these defensive methods either introduce orders of magnitude more computational and communication overheads (e.g., with homomorphic encryption) or incur substantial model performance losses in terms of prediction accuracy (e.g., with differential privacy). In this work, we propose $\textsc{FedCG}$, a novel federated learning method that leverages conditional generative adversarial networks to achieve high-level privacy protection while still maintaining competitive model performance. $\textsc{FedCG}$ decomposes each client's local network into a private extractor and a public classifier and keeps the extractor local to protect privacy. Instead of exposing extractors, $\textsc{FedCG}$ shares clients' generators with the server for aggregating clients' shared knowledge, aiming to enhance the performance of each client's local networks. Extensive experiments demonstrate that $\textsc{FedCG}$ can achieve competitive model performance compared with FL baselines, and privacy analysis shows that $\textsc{FedCG}$ has a high-level privacy-preserving capability. Code is available at https://github.com/yankang18/FedCG



## **13. A Novel Bifurcation Method for Observation Perturbation Attacks on Reinforcement Learning Agents: Load Altering Attacks on a Cyber Physical Power System**

cs.LG

12 pages, 5 figures

**SubmitDate**: 2024-07-06    [abs](http://arxiv.org/abs/2407.05182v1) [paper-pdf](http://arxiv.org/pdf/2407.05182v1)

**Authors**: Kiernan Broda-Milian, Ranwa Al-Mallah, Hanane Dagdougui

**Abstract**: Components of cyber physical systems, which affect real-world processes, are often exposed to the internet. Replacing conventional control methods with Deep Reinforcement Learning (DRL) in energy systems is an active area of research, as these systems become increasingly complex with the advent of renewable energy sources and the desire to improve their efficiency. Artificial Neural Networks (ANN) are vulnerable to specific perturbations of their inputs or features, called adversarial examples. These perturbations are difficult to detect when properly regularized, but have significant effects on the ANN's output. Because DRL uses ANN to map optimal actions to observations, they are similarly vulnerable to adversarial examples. This work proposes a novel attack technique for continuous control using Group Difference Logits loss with a bifurcation layer. By combining aspects of targeted and untargeted attacks, the attack significantly increases the impact compared to an untargeted attack, with drastically smaller distortions than an optimally targeted attack. We demonstrate the impacts of powerful gradient-based attacks in a realistic smart energy environment, show how the impacts change with different DRL agents and training procedures, and use statistical and time-series analysis to evaluate attacks' stealth. The results show that adversarial attacks can have significant impacts on DRL controllers, and constraining an attack's perturbations makes it difficult to detect. However, certain DRL architectures are far more robust, and robust training methods can further reduce the impact.



## **14. Robust Skin Color Driven Privacy Preserving Face Recognition via Function Secret Sharing**

cs.CV

Accepted at ICIP2024

**SubmitDate**: 2024-07-06    [abs](http://arxiv.org/abs/2407.05045v1) [paper-pdf](http://arxiv.org/pdf/2407.05045v1)

**Authors**: Dong Han, Yufan Jiang, Yong Li, Ricardo Mendes, Joachim Denzler

**Abstract**: In this work, we leverage the pure skin color patch from the face image as the additional information to train an auxiliary skin color feature extractor and face recognition model in parallel to improve performance of state-of-the-art (SOTA) privacy-preserving face recognition (PPFR) systems. Our solution is robust against black-box attacking and well-established generative adversarial network (GAN) based image restoration. We analyze the potential risk in previous work, where the proposed cosine similarity computation might directly leak the protected precomputed embedding stored on the server side. We propose a Function Secret Sharing (FSS) based face embedding comparison protocol without any intermediate result leakage. In addition, we show in experiments that the proposed protocol is more efficient compared to the Secret Sharing (SS) based protocol.



## **15. Certified Zeroth-order Black-Box Defense with Robust UNet Denoiser**

cs.CV

**SubmitDate**: 2024-07-06    [abs](http://arxiv.org/abs/2304.06430v2) [paper-pdf](http://arxiv.org/pdf/2304.06430v2)

**Authors**: Astha Verma, A V Subramanyam, Siddhesh Bangar, Naman Lal, Rajiv Ratn Shah, Shin'ichi Satoh

**Abstract**: Certified defense methods against adversarial perturbations have been recently investigated in the black-box setting with a zeroth-order (ZO) perspective. However, these methods suffer from high model variance with low performance on high-dimensional datasets due to the ineffective design of the denoiser and are limited in their utilization of ZO techniques. To this end, we propose a certified ZO preprocessing technique for removing adversarial perturbations from the attacked image in the black-box setting using only model queries. We propose a robust UNet denoiser (RDUNet) that ensures the robustness of black-box models trained on high-dimensional datasets. We propose a novel black-box denoised smoothing (DS) defense mechanism, ZO-RUDS, by prepending our RDUNet to the black-box model, ensuring black-box defense. We further propose ZO-AE-RUDS in which RDUNet followed by autoencoder (AE) is prepended to the black-box model. We perform extensive experiments on four classification datasets, CIFAR-10, CIFAR-10, Tiny Imagenet, STL-10, and the MNIST dataset for image reconstruction tasks. Our proposed defense methods ZO-RUDS and ZO-AE-RUDS beat SOTA with a huge margin of $35\%$ and $9\%$, for low dimensional (CIFAR-10) and with a margin of $20.61\%$ and $23.51\%$ for high-dimensional (STL-10) datasets, respectively.



## **16. PAC-Bayesian Adversarially Robust Generalization Bounds for Graph Neural Network**

stat.ML

38pages

**SubmitDate**: 2024-07-06    [abs](http://arxiv.org/abs/2402.04038v2) [paper-pdf](http://arxiv.org/pdf/2402.04038v2)

**Authors**: Tan Sun, Junhong Lin

**Abstract**: Graph neural networks (GNNs) have gained popularity for various graph-related tasks. However, similar to deep neural networks, GNNs are also vulnerable to adversarial attacks. Empirical studies have shown that adversarially robust generalization has a pivotal role in establishing effective defense algorithms against adversarial attacks. In this paper, we contribute by providing adversarially robust generalization bounds for two kinds of popular GNNs, graph convolutional network (GCN) and message passing graph neural network, using the PAC-Bayesian framework. Our result reveals that spectral norm of the diffusion matrix on the graph and spectral norm of the weights as well as the perturbation factor govern the robust generalization bounds of both models. Our bounds are nontrivial generalizations of the results developed in (Liao et al., 2020) from the standard setting to adversarial setting while avoiding exponential dependence of the maximum node degree. As corollaries, we derive better PAC-Bayesian robust generalization bounds for GCN in the standard setting, which improve the bounds in (Liao et al., 2020) by avoiding exponential dependence on the maximum node degree.



## **17. Defensive Reconfigurable Intelligent Surface (D-RIS) Based on Non-Reciprocal Channel Links**

eess.SP

14 pages, journal paper

**SubmitDate**: 2024-07-06    [abs](http://arxiv.org/abs/2407.04905v1) [paper-pdf](http://arxiv.org/pdf/2407.04905v1)

**Authors**: Kun Chen-Hu, Petar Popovski

**Abstract**: A reconfigurable intelligent surface (RIS) is commonly made of low-cost passive and reflective meta-materials with excellent beam steering capabilities. It is applied to enhance wireless communication systems as a customizable signal reflector. However, RIS can also be adversely employed to disrupt the existing communication systems by introducing new types of vulnerability to the physical layer. We consider the \emph{RIS-In-The-Middle (RITM) attack}, in which an adversary uses RIS to jeopardize the direct channel between two transceivers by providing an alternative one with higher signal quality. This adversary can eavesdrop on all exchanged data by the legitimate users, but also perform a false data injection to the receiver. This work devises anti-attack techniques based on a non-reciprocal channel produced by a defensive RIS (D-RIS). The proposed precoding and combining methods and the channel estimation procedure for a non-reciprocal link are effective against potential adversaries while keeping the existing advantages of the RIS. We analyse the robustness of the system against attacks in terms of achievable secrecy rate and probability of detecting fake data. We believe that this defensive role of RIS can be a basis for new protocols and algorithms in the area.



## **18. Late Breaking Results: Fortifying Neural Networks: Safeguarding Against Adversarial Attacks with Stochastic Computing**

cs.CR

3 pages, 1 figure, 2 tables

**SubmitDate**: 2024-07-05    [abs](http://arxiv.org/abs/2407.04861v1) [paper-pdf](http://arxiv.org/pdf/2407.04861v1)

**Authors**: Faeze S. Banitaba, Sercan Aygun, M. Hassan Najafi

**Abstract**: In neural network (NN) security, safeguarding model integrity and resilience against adversarial attacks has become paramount. This study investigates the application of stochastic computing (SC) as a novel mechanism to fortify NN models. The primary objective is to assess the efficacy of SC to mitigate the deleterious impact of attacks on NN results. Through a series of rigorous experiments and evaluations, we explore the resilience of NNs employing SC when subjected to adversarial attacks. Our findings reveal that SC introduces a robust layer of defense, significantly reducing the susceptibility of networks to attack-induced alterations in their outcomes. This research contributes novel insights into the development of more secure and reliable NN systems, essential for applications in sensitive domains where data integrity is of utmost concern.



## **19. Where have you been? A Study of Privacy Risk for Point-of-Interest Recommendation**

cs.LG

18 pages

**SubmitDate**: 2024-07-05    [abs](http://arxiv.org/abs/2310.18606v2) [paper-pdf](http://arxiv.org/pdf/2310.18606v2)

**Authors**: Kunlin Cai, Jinghuai Zhang, Zhiqing Hong, Will Shand, Guang Wang, Desheng Zhang, Jianfeng Chi, Yuan Tian

**Abstract**: As location-based services (LBS) have grown in popularity, more human mobility data has been collected. The collected data can be used to build machine learning (ML) models for LBS to enhance their performance and improve overall experience for users. However, the convenience comes with the risk of privacy leakage since this type of data might contain sensitive information related to user identities, such as home/work locations. Prior work focuses on protecting mobility data privacy during transmission or prior to release, lacking the privacy risk evaluation of mobility data-based ML models. To better understand and quantify the privacy leakage in mobility data-based ML models, we design a privacy attack suite containing data extraction and membership inference attacks tailored for point-of-interest (POI) recommendation models, one of the most widely used mobility data-based ML models. These attacks in our attack suite assume different adversary knowledge and aim to extract different types of sensitive information from mobility data, providing a holistic privacy risk assessment for POI recommendation models. Our experimental evaluation using two real-world mobility datasets demonstrates that current POI recommendation models are vulnerable to our attacks. We also present unique findings to understand what types of mobility data are more susceptible to privacy attacks. Finally, we evaluate defenses against these attacks and highlight future directions and challenges. Our attack suite is released at https://github.com/KunlinChoi/POIPrivacy.



## **20. CosPGD: an efficient white-box adversarial attack for pixel-wise prediction tasks**

cs.CV

Accepted at 41st International Conference on Machine Learning (ICML),  2024

**SubmitDate**: 2024-07-05    [abs](http://arxiv.org/abs/2302.02213v3) [paper-pdf](http://arxiv.org/pdf/2302.02213v3)

**Authors**: Shashank Agnihotri, Steffen Jung, Margret Keuper

**Abstract**: While neural networks allow highly accurate predictions in many tasks, their lack of robustness towards even slight input perturbations often hampers their deployment. Adversarial attacks such as the seminal projected gradient descent (PGD) offer an effective means to evaluate a model's robustness and dedicated solutions have been proposed for attacks on semantic segmentation or optical flow estimation. While they attempt to increase the attack's efficiency, a further objective is to balance its effect, so that it acts on the entire image domain instead of isolated point-wise predictions. This often comes at the cost of optimization stability and thus efficiency. Here, we propose CosPGD, an attack that encourages more balanced errors over the entire image domain while increasing the attack's overall efficiency. To this end, CosPGD leverages a simple alignment score computed from any pixel-wise prediction and its target to scale the loss in a smooth and fully differentiable way. It leads to efficient evaluations of a model's robustness for semantic segmentation as well as regression models (such as optical flow, disparity estimation, or image restoration), and it allows it to outperform the previous SotA attack on semantic segmentation. We provide code for the CosPGD algorithm and example usage at https://github.com/shashankskagnihotri/cospgd.



## **21. On Evaluating The Performance of Watermarked Machine-Generated Texts Under Adversarial Attacks**

cs.CR

**SubmitDate**: 2024-07-05    [abs](http://arxiv.org/abs/2407.04794v1) [paper-pdf](http://arxiv.org/pdf/2407.04794v1)

**Authors**: Zesen Liu, Tianshuo Cong, Xinlei He, Qi Li

**Abstract**: Large Language Models (LLMs) excel in various applications, including text generation and complex tasks. However, the misuse of LLMs raises concerns about the authenticity and ethical implications of the content they produce, such as deepfake news, academic fraud, and copyright infringement. Watermarking techniques, which embed identifiable markers in machine-generated text, offer a promising solution to these issues by allowing for content verification and origin tracing. Unfortunately, the robustness of current LLM watermarking schemes under potential watermark removal attacks has not been comprehensively explored.   In this paper, to fill this gap, we first systematically comb the mainstream watermarking schemes and removal attacks on machine-generated texts, and then we categorize them into pre-text (before text generation) and post-text (after text generation) classes so that we can conduct diversified analyses. In our experiments, we evaluate eight watermarks (five pre-text, three post-text) and twelve attacks (two pre-text, ten post-text) across 87 scenarios. Evaluation results indicate that (1) KGW and Exponential watermarks offer high text quality and watermark retention but remain vulnerable to most attacks; (2) Post-text attacks are found to be more efficient and practical than pre-text attacks; (3) Pre-text watermarks are generally more imperceptible, as they do not alter text fluency, unlike post-text watermarks; (4) Additionally, combined attack methods can significantly increase effectiveness, highlighting the need for more robust watermarking solutions. Our study underscores the vulnerabilities of current techniques and the necessity for developing more resilient schemes.



## **22. Remembering Everything Makes You Vulnerable: A Limelight on Machine Unlearning for Personalized Healthcare Sector**

cs.LG

15 Pages, Exploring unlearning techniques on ECG Classifier

**SubmitDate**: 2024-07-05    [abs](http://arxiv.org/abs/2407.04589v1) [paper-pdf](http://arxiv.org/pdf/2407.04589v1)

**Authors**: Ahan Chatterjee, Sai Anirudh Aryasomayajula, Rajat Chaudhari, Subhajit Paul, Vishwa Mohan Singh

**Abstract**: As the prevalence of data-driven technologies in healthcare continues to rise, concerns regarding data privacy and security become increasingly paramount. This thesis aims to address the vulnerability of personalized healthcare models, particularly in the context of ECG monitoring, to adversarial attacks that compromise patient privacy. We propose an approach termed "Machine Unlearning" to mitigate the impact of exposed data points on machine learning models, thereby enhancing model robustness against adversarial attacks while preserving individual privacy. Specifically, we investigate the efficacy of Machine Unlearning in the context of personalized ECG monitoring, utilizing a dataset of clinical ECG recordings. Our methodology involves training a deep neural classifier on ECG data and fine-tuning the model for individual patients. We demonstrate the susceptibility of fine-tuned models to adversarial attacks, such as the Fast Gradient Sign Method (FGSM), which can exploit additional data points in personalized models. To address this vulnerability, we propose a Machine Unlearning algorithm that selectively removes sensitive data points from fine-tuned models, effectively enhancing model resilience against adversarial manipulation. Experimental results demonstrate the effectiveness of our approach in mitigating the impact of adversarial attacks while maintaining the pre-trained model accuracy.



## **23. Certifiable Black-Box Attacks with Randomized Adversarial Examples: Breaking Defenses with Provable Confidence**

cs.LG

**SubmitDate**: 2024-07-05    [abs](http://arxiv.org/abs/2304.04343v2) [paper-pdf](http://arxiv.org/pdf/2304.04343v2)

**Authors**: Hanbin Hong, Xinyu Zhang, Binghui Wang, Zhongjie Ba, Yuan Hong

**Abstract**: Black-box adversarial attacks have shown strong potential to subvert machine learning models. Existing black-box attacks craft adversarial examples by iteratively querying the target model and/or leveraging the transferability of a local surrogate model. Recently, such attacks can be effectively mitigated by state-of-the-art (SOTA) defenses, e.g., detection via the pattern of sequential queries, or injecting noise into the model. To our best knowledge, we take the first step to study a new paradigm of black-box attacks with provable guarantees -- certifiable black-box attacks that can guarantee the attack success probability (ASP) of adversarial examples before querying over the target model. This new black-box attack unveils significant vulnerabilities of machine learning models, compared to traditional empirical black-box attacks, e.g., breaking strong SOTA defenses with provable confidence, constructing a space of (infinite) adversarial examples with high ASP, and the ASP of the generated adversarial examples is theoretically guaranteed without verification/queries over the target model. Specifically, we establish a novel theoretical foundation for ensuring the ASP of the black-box attack with randomized adversarial examples (AEs). Then, we propose several novel techniques to craft the randomized AEs while reducing the perturbation size for better imperceptibility. Finally, we have comprehensively evaluated the certifiable black-box attacks on the CIFAR10/100, ImageNet, and LibriSpeech datasets, while benchmarking with 16 SOTA empirical black-box attacks, against various SOTA defenses in the domains of computer vision and speech recognition. Both theoretical and experimental results have validated the significance of the proposed attack.



## **24. Artwork Protection Against Neural Style Transfer Using Locally Adaptive Adversarial Color Attack**

cs.CV

9 pages, 5 figures, 4 tables

**SubmitDate**: 2024-07-05    [abs](http://arxiv.org/abs/2401.09673v3) [paper-pdf](http://arxiv.org/pdf/2401.09673v3)

**Authors**: Zhongliang Guo, Junhao Dong, Yifei Qian, Kaixuan Wang, Weiye Li, Ziheng Guo, Yuheng Wang, Yanli Li, Ognjen Arandjelović, Lei Fang

**Abstract**: Neural style transfer (NST) generates new images by combining the style of one image with the content of another. However, unauthorized NST can exploit artwork, raising concerns about artists' rights and motivating the development of proactive protection methods. We propose Locally Adaptive Adversarial Color Attack (LAACA), empowering artists to protect their artwork from unauthorized style transfer by processing before public release. By delving into the intricacies of human visual perception and the role of different frequency components, our method strategically introduces frequency-adaptive perturbations in the image. These perturbations significantly degrade the generation quality of NST while maintaining an acceptable level of visual change in the original image, ensuring that potential infringers are discouraged from using the protected artworks, because of its bad NST generation quality. Additionally, existing metrics often overlook the importance of color fidelity in evaluating color-mattered tasks, such as the quality of NST-generated images, which is crucial in the context of artistic works. To comprehensively assess the color-mattered tasks, we propose the Adversarial Color Distance Metric (ACDM), designed to quantify the color difference of images pre- and post-manipulations. Experimental results confirm that attacking NST using LAACA results in visually inferior style transfer, and the ACDM can efficiently measure color-mattered tasks. By providing artists with a tool to safeguard their intellectual property, our work relieves the socio-technical challenges posed by the misuse of NST in the art community.



## **25. Distributed Black-box Attack: Do Not Overestimate Black-box Attacks**

cs.LG

8 pages, 10 figures

**SubmitDate**: 2024-07-05    [abs](http://arxiv.org/abs/2210.16371v4) [paper-pdf](http://arxiv.org/pdf/2210.16371v4)

**Authors**: Han Wu, Sareh Rowlands, Johan Wahlstrom

**Abstract**: Black-box adversarial attacks can fool image classifiers into misclassifying images without requiring access to model structure and weights. Recent studies have reported attack success rates of over 95% with less than 1,000 queries. The question then arises of whether black-box attacks have become a real threat against IoT devices that rely on cloud APIs to achieve image classification. To shed some light on this, note that prior research has primarily focused on increasing the success rate and reducing the number of queries. However, another crucial factor for black-box attacks against cloud APIs is the time required to perform the attack. This paper applies black-box attacks directly to cloud APIs rather than to local models, thereby avoiding mistakes made in prior research that applied the perturbation before image encoding and pre-processing. Further, we exploit load balancing to enable distributed black-box attacks that can reduce the attack time by a factor of about five for both local search and gradient estimation methods.



## **26. Controlling Whisper: Universal Acoustic Adversarial Attacks to Control Speech Foundation Models**

cs.SD

**SubmitDate**: 2024-07-05    [abs](http://arxiv.org/abs/2407.04482v1) [paper-pdf](http://arxiv.org/pdf/2407.04482v1)

**Authors**: Vyas Raina, Mark Gales

**Abstract**: Speech enabled foundation models, either in the form of flexible speech recognition based systems or audio-prompted large language models (LLMs), are becoming increasingly popular. One of the interesting aspects of these models is their ability to perform tasks other than automatic speech recognition (ASR) using an appropriate prompt. For example, the OpenAI Whisper model can perform both speech transcription and speech translation. With the development of audio-prompted LLMs there is the potential for even greater control options. In this work we demonstrate that with this greater flexibility the systems can be susceptible to model-control adversarial attacks. Without any access to the model prompt it is possible to modify the behaviour of the system by appropriately changing the audio input. To illustrate this risk, we demonstrate that it is possible to prepend a short universal adversarial acoustic segment to any input speech signal to override the prompt setting of an ASR foundation model. Specifically, we successfully use a universal adversarial acoustic segment to control Whisper to always perform speech translation, despite being set to perform speech transcription. Overall, this work demonstrates a new form of adversarial attack on multi-tasking speech enabled foundation models that needs to be considered prior to the deployment of this form of model.



## **27. Self-Supervised Representation Learning for Adversarial Attack Detection**

cs.CV

ECCV 2024

**SubmitDate**: 2024-07-05    [abs](http://arxiv.org/abs/2407.04382v1) [paper-pdf](http://arxiv.org/pdf/2407.04382v1)

**Authors**: Yi Li, Plamen Angelov, Neeraj Suri

**Abstract**: Supervised learning-based adversarial attack detection methods rely on a large number of labeled data and suffer significant performance degradation when applying the trained model to new domains. In this paper, we propose a self-supervised representation learning framework for the adversarial attack detection task to address this drawback. Firstly, we map the pixels of augmented input images into an embedding space. Then, we employ the prototype-wise contrastive estimation loss to cluster prototypes as latent variables. Additionally, drawing inspiration from the concept of memory banks, we introduce a discrimination bank to distinguish and learn representations for each individual instance that shares the same or a similar prototype, establishing a connection between instances and their associated prototypes. We propose a parallel axial-attention (PAA)-based encoder to facilitate the training process by parallel training over height- and width-axis of attention maps. Experimental results show that, compared to various benchmark self-supervised vision learning models and supervised adversarial attack detection methods, the proposed model achieves state-of-the-art performance on the adversarial attack detection task across a wide range of images.



## **28. Jailbreak Attacks and Defenses Against Large Language Models: A Survey**

cs.CR

**SubmitDate**: 2024-07-05    [abs](http://arxiv.org/abs/2407.04295v1) [paper-pdf](http://arxiv.org/pdf/2407.04295v1)

**Authors**: Sibo Yi, Yule Liu, Zhen Sun, Tianshuo Cong, Xinlei He, Jiaxing Song, Ke Xu, Qi Li

**Abstract**: Large Language Models (LLMs) have performed exceptionally in various text-generative tasks, including question answering, translation, code completion, etc. However, the over-assistance of LLMs has raised the challenge of "jailbreaking", which induces the model to generate malicious responses against the usage policy and society by designing adversarial prompts. With the emergence of jailbreak attack methods exploiting different vulnerabilities in LLMs, the corresponding safety alignment measures are also evolving. In this paper, we propose a comprehensive and detailed taxonomy of jailbreak attack and defense methods. For instance, the attack methods are divided into black-box and white-box attacks based on the transparency of the target model. Meanwhile, we classify defense methods into prompt-level and model-level defenses. Additionally, we further subdivide these attack and defense methods into distinct sub-classes and present a coherent diagram illustrating their relationships. We also conduct an investigation into the current evaluation methods and compare them from different perspectives. Our findings aim to inspire future research and practical implementations in safeguarding LLMs against adversarial attacks. Above all, although jailbreak remains a significant concern within the community, we believe that our work enhances the understanding of this domain and provides a foundation for developing more secure LLMs.



## **29. FlowMur: A Stealthy and Practical Audio Backdoor Attack with Limited Knowledge**

cs.CR

To appear at lEEE Symposium on Security & Privacy (Oakland) 2024

**SubmitDate**: 2024-07-05    [abs](http://arxiv.org/abs/2312.09665v2) [paper-pdf](http://arxiv.org/pdf/2312.09665v2)

**Authors**: Jiahe Lan, Jie Wang, Baochen Yan, Zheng Yan, Elisa Bertino

**Abstract**: Speech recognition systems driven by DNNs have revolutionized human-computer interaction through voice interfaces, which significantly facilitate our daily lives. However, the growing popularity of these systems also raises special concerns on their security, particularly regarding backdoor attacks. A backdoor attack inserts one or more hidden backdoors into a DNN model during its training process, such that it does not affect the model's performance on benign inputs, but forces the model to produce an adversary-desired output if a specific trigger is present in the model input. Despite the initial success of current audio backdoor attacks, they suffer from the following limitations: (i) Most of them require sufficient knowledge, which limits their widespread adoption. (ii) They are not stealthy enough, thus easy to be detected by humans. (iii) Most of them cannot attack live speech, reducing their practicality. To address these problems, in this paper, we propose FlowMur, a stealthy and practical audio backdoor attack that can be launched with limited knowledge. FlowMur constructs an auxiliary dataset and a surrogate model to augment adversary knowledge. To achieve dynamicity, it formulates trigger generation as an optimization problem and optimizes the trigger over different attachment positions. To enhance stealthiness, we propose an adaptive data poisoning method according to Signal-to-Noise Ratio (SNR). Furthermore, ambient noise is incorporated into the process of trigger generation and data poisoning to make FlowMur robust to ambient noise and improve its practicality. Extensive experiments conducted on two datasets demonstrate that FlowMur achieves high attack performance in both digital and physical settings while remaining resilient to state-of-the-art defenses. In particular, a human study confirms that triggers generated by FlowMur are not easily detected by participants.



## **30. Defending Jailbreak Prompts via In-Context Adversarial Game**

cs.LG

**SubmitDate**: 2024-07-05    [abs](http://arxiv.org/abs/2402.13148v2) [paper-pdf](http://arxiv.org/pdf/2402.13148v2)

**Authors**: Yujun Zhou, Yufei Han, Haomin Zhuang, Kehan Guo, Zhenwen Liang, Hongyan Bao, Xiangliang Zhang

**Abstract**: Large Language Models (LLMs) demonstrate remarkable capabilities across diverse applications. However, concerns regarding their security, particularly the vulnerability to jailbreak attacks, persist. Drawing inspiration from adversarial training in deep learning and LLM agent learning processes, we introduce the In-Context Adversarial Game (ICAG) for defending against jailbreaks without the need for fine-tuning. ICAG leverages agent learning to conduct an adversarial game, aiming to dynamically extend knowledge to defend against jailbreaks. Unlike traditional methods that rely on static datasets, ICAG employs an iterative process to enhance both the defense and attack agents. This continuous improvement process strengthens defenses against newly generated jailbreak prompts. Our empirical studies affirm ICAG's efficacy, where LLMs safeguarded by ICAG exhibit significantly reduced jailbreak success rates across various attack scenarios. Moreover, ICAG demonstrates remarkable transferability to other LLMs, indicating its potential as a versatile defense mechanism.



## **31. Securing Multi-turn Conversational Language Models Against Distributed Backdoor Triggers**

cs.CL

Submitted to EMNLP 2024

**SubmitDate**: 2024-07-04    [abs](http://arxiv.org/abs/2407.04151v1) [paper-pdf](http://arxiv.org/pdf/2407.04151v1)

**Authors**: Terry Tong, Jiashu Xu, Qin Liu, Muhao Chen

**Abstract**: The security of multi-turn conversational large language models (LLMs) is understudied despite it being one of the most popular LLM utilization. Specifically, LLMs are vulnerable to data poisoning backdoor attacks, where an adversary manipulates the training data to cause the model to output malicious responses to predefined triggers. Specific to the multi-turn dialogue setting, LLMs are at the risk of even more harmful and stealthy backdoor attacks where the backdoor triggers may span across multiple utterances, giving lee-way to context-driven attacks. In this paper, we explore a novel distributed backdoor trigger attack that serves to be an extra tool in an adversary's toolbox that can interface with other single-turn attack strategies in a plug and play manner. Results on two representative defense mechanisms indicate that distributed backdoor triggers are robust against existing defense strategies which are designed for single-turn user-model interactions, motivating us to propose a new defense strategy for the multi-turn dialogue setting that is more challenging. To this end, we also explore a novel contrastive decoding based defense that is able to mitigate the backdoor with a low computational tradeoff.



## **32. A Survey on Adversarial Contention Resolution**

cs.DC

**SubmitDate**: 2024-07-04    [abs](http://arxiv.org/abs/2403.03876v2) [paper-pdf](http://arxiv.org/pdf/2403.03876v2)

**Authors**: Ioana Banicescu, Trisha Chakraborty, Seth Gilbert, Maxwell Young

**Abstract**: Contention resolution addresses the challenge of coordinating access by multiple processes to a shared resource such as memory, disk storage, or a communication channel. Originally spurred by challenges in database systems and bus networks, contention resolution has endured as an important abstraction for resource sharing, despite decades of technological change. Here, we survey the literature on resolving worst-case contention, where the number of processes and the time at which each process may start seeking access to the resource is dictated by an adversary. We highlight the evolution of contention resolution, where new concerns -- such as security, quality of service, and energy efficiency -- are motivated by modern systems. These efforts have yielded insights into the limits of randomized and deterministic approaches, as well as the impact of different model assumptions such as global clock synchronization, knowledge of the number of processors, feedback from access attempts, and attacks on the availability of the shared resource.



## **33. Optimizing a-DCF for Spoofing-Robust Speaker Verification**

eess.AS

**SubmitDate**: 2024-07-04    [abs](http://arxiv.org/abs/2407.04034v1) [paper-pdf](http://arxiv.org/pdf/2407.04034v1)

**Authors**: Oğuzhan Kurnaz, Jagabandhu Mishra, Tomi H. Kinnunen, Cemal Hanilçi

**Abstract**: Automatic speaker verification (ASV) systems are vulnerable to spoofing attacks such as text-to-speech. In this study, we propose a novel spoofing-robust ASV back-end classifier, optimized directly for the recently introduced, architecture-agnostic detection cost function (a-DCF). We combine a-DCF and binary cross-entropy (BCE) losses to optimize the network weights, combined by a novel, straightforward detection threshold optimization technique. Experiments on the ASVspoof2019 database demonstrate considerable improvement over the baseline optimized using BCE only (from minimum a-DCF of 0.1445 to 0.1254), representing 13% relative improvement. These initial promising results demonstrate that it is possible to adjust an ASV system to find appropriate balance across the contradicting aims of user convenience and security against adversaries.



## **34. Mitigating Low-Frequency Bias: Feature Recalibration and Frequency Attention Regularization for Adversarial Robustness**

cs.CV

**SubmitDate**: 2024-07-04    [abs](http://arxiv.org/abs/2407.04016v1) [paper-pdf](http://arxiv.org/pdf/2407.04016v1)

**Authors**: Kejia Zhang, Juanjuan Weng, Yuanzheng Cai, Zhiming Luo, Shaozi Li

**Abstract**: Ensuring the robustness of computer vision models against adversarial attacks is a significant and long-lasting objective. Motivated by adversarial attacks, researchers have devoted considerable efforts to enhancing model robustness by adversarial training (AT). However, we observe that while AT improves the models' robustness against adversarial perturbations, it fails to improve their ability to effectively extract features across all frequency components. Each frequency component contains distinct types of crucial information: low-frequency features provide fundamental structural insights, while high-frequency features capture intricate details and textures. In particular, AT tends to neglect the reliance on susceptible high-frequency features. This low-frequency bias impedes the model's ability to effectively leverage the potentially meaningful semantic information present in high-frequency features. This paper proposes a novel module called High-Frequency Feature Disentanglement and Recalibration (HFDR), which separates features into high-frequency and low-frequency components and recalibrates the high-frequency feature to capture latent useful semantics. Additionally, we introduce frequency attention regularization to magnitude the model's extraction of different frequency features and mitigate low-frequency bias during AT. Extensive experiments showcase the immense potential and superiority of our approach in resisting various white-box attacks, transfer attacks, and showcasing strong generalization capabilities.



## **35. TrackPGD: A White-box Attack using Binary Masks against Robust Transformer Trackers**

cs.CV

**SubmitDate**: 2024-07-04    [abs](http://arxiv.org/abs/2407.03946v1) [paper-pdf](http://arxiv.org/pdf/2407.03946v1)

**Authors**: Fatemeh Nourilenjan Nokabadi, Yann Batiste Pequignot, Jean-Francois Lalonde, Christian Gagné

**Abstract**: Object trackers with transformer backbones have achieved robust performance on visual object tracking datasets. However, the adversarial robustness of these trackers has not been well studied in the literature. Due to the backbone differences, the adversarial white-box attacks proposed for object tracking are not transferable to all types of trackers. For instance, transformer trackers such as MixFormerM still function well after black-box attacks, especially in predicting the object binary masks. We are proposing a novel white-box attack named TrackPGD, which relies on the predicted object binary mask to attack the robust transformer trackers. That new attack focuses on annotation masks by adapting the well-known SegPGD segmentation attack, allowing to successfully conduct the white-box attack on trackers relying on transformer backbones. The experimental results indicate that the TrackPGD is able to effectively attack transformer-based trackers such as MixFormerM, OSTrackSTS, and TransT-SEG on several tracking datasets.



## **36. EvolBA: Evolutionary Boundary Attack under Hard-label Black Box condition**

cs.CV

**SubmitDate**: 2024-07-04    [abs](http://arxiv.org/abs/2407.02248v2) [paper-pdf](http://arxiv.org/pdf/2407.02248v2)

**Authors**: Ayane Tajima, Satoshi Ono

**Abstract**: Research has shown that deep neural networks (DNNs) have vulnerabilities that can lead to the misrecognition of Adversarial Examples (AEs) with specifically designed perturbations. Various adversarial attack methods have been proposed to detect vulnerabilities under hard-label black box (HL-BB) conditions in the absence of loss gradients and confidence scores.However, these methods fall into local solutions because they search only local regions of the search space. Therefore, this study proposes an adversarial attack method named EvolBA to generate AEs using Covariance Matrix Adaptation Evolution Strategy (CMA-ES) under the HL-BB condition, where only a class label predicted by the target DNN model is available. Inspired by formula-driven supervised learning, the proposed method introduces domain-independent operators for the initialization process and a jump that enhances search exploration. Experimental results confirmed that the proposed method could determine AEs with smaller perturbations than previous methods in images where the previous methods have difficulty.



## **37. Is LLM-as-a-Judge Robust? Investigating Universal Adversarial Attacks on Zero-shot LLM Assessment**

cs.CL

**SubmitDate**: 2024-07-04    [abs](http://arxiv.org/abs/2402.14016v2) [paper-pdf](http://arxiv.org/pdf/2402.14016v2)

**Authors**: Vyas Raina, Adian Liusie, Mark Gales

**Abstract**: Large Language Models (LLMs) are powerful zero-shot assessors used in real-world situations such as assessing written exams and benchmarking systems. Despite these critical applications, no existing work has analyzed the vulnerability of judge-LLMs to adversarial manipulation. This work presents the first study on the adversarial robustness of assessment LLMs, where we demonstrate that short universal adversarial phrases can be concatenated to deceive judge LLMs to predict inflated scores. Since adversaries may not know or have access to the judge-LLMs, we propose a simple surrogate attack where a surrogate model is first attacked, and the learned attack phrase then transferred to unknown judge-LLMs. We propose a practical algorithm to determine the short universal attack phrases and demonstrate that when transferred to unseen models, scores can be drastically inflated such that irrespective of the assessed text, maximum scores are predicted. It is found that judge-LLMs are significantly more susceptible to these adversarial attacks when used for absolute scoring, as opposed to comparative assessment. Our findings raise concerns on the reliability of LLM-as-a-judge methods, and emphasize the importance of addressing vulnerabilities in LLM assessment methods before deployment in high-stakes real-world scenarios.



## **38. DART: Deep Adversarial Automated Red Teaming for LLM Safety**

cs.CR

**SubmitDate**: 2024-07-04    [abs](http://arxiv.org/abs/2407.03876v1) [paper-pdf](http://arxiv.org/pdf/2407.03876v1)

**Authors**: Bojian Jiang, Yi Jing, Tianhao Shen, Qing Yang, Deyi Xiong

**Abstract**: Manual Red teaming is a commonly-used method to identify vulnerabilities in large language models (LLMs), which, is costly and unscalable. In contrast, automated red teaming uses a Red LLM to automatically generate adversarial prompts to the Target LLM, offering a scalable way for safety vulnerability detection. However, the difficulty of building a powerful automated Red LLM lies in the fact that the safety vulnerabilities of the Target LLM are dynamically changing with the evolution of the Target LLM. To mitigate this issue, we propose a Deep Adversarial Automated Red Teaming (DART) framework in which the Red LLM and Target LLM are deeply and dynamically interacting with each other in an iterative manner. In each iteration, in order to generate successful attacks as many as possible, the Red LLM not only takes into account the responses from the Target LLM, but also adversarially adjust its attacking directions by monitoring the global diversity of generated attacks across multiple iterations. Simultaneously, to explore dynamically changing safety vulnerabilities of the Target LLM, we allow the Target LLM to enhance its safety via an active learning based data selection mechanism. Experimential results demonstrate that DART significantly reduces the safety risk of the target LLM. For human evaluation on Anthropic Harmless dataset, compared to the instruction-tuning target LLM, DART eliminates the violation risks by 53.4\%. We will release the datasets and codes of DART soon.



## **39. RobQuNNs: A Methodology for Robust Quanvolutional Neural Networks against Adversarial Attacks**

quant-ph

6 pages, 3 figures

**SubmitDate**: 2024-07-04    [abs](http://arxiv.org/abs/2407.03875v1) [paper-pdf](http://arxiv.org/pdf/2407.03875v1)

**Authors**: Walid El Maouaki, Alberto Marchisio, Taoufik Said, Muhammad Shafique, Mohamed Bennai

**Abstract**: Recent advancements in quantum computing have led to the emergence of hybrid quantum neural networks, such as Quanvolutional Neural Networks (QuNNs), which integrate quantum and classical layers. While the susceptibility of classical neural networks to adversarial attacks is well-documented, the impact on QuNNs remains less understood. This study introduces RobQuNN, a new methodology to enhance the robustness of QuNNs against adversarial attacks, utilizing quantum circuit expressibility and entanglement capability alongside different adversarial strategies. Additionally, the study investigates the transferability of adversarial examples between classical and quantum models using RobQuNN, enhancing our understanding of cross-model vulnerabilities and pointing to new directions in quantum cybersecurity. The findings reveal that QuNNs exhibit up to 60\% higher robustness compared to classical networks for the MNIST dataset, particularly at low levels of perturbation. This underscores the potential of quantum approaches in improving security defenses. In addition, RobQuNN revealed that QuNN does not exhibit enhanced resistance or susceptibility to cross-model adversarial examples regardless of the quantum circuit architecture.



## **40. Adversarial Robustness of VAEs across Intersectional Subgroups**

cs.LG

**SubmitDate**: 2024-07-04    [abs](http://arxiv.org/abs/2407.03864v1) [paper-pdf](http://arxiv.org/pdf/2407.03864v1)

**Authors**: Chethan Krishnamurthy Ramanaik, Arjun Roy, Eirini Ntoutsi

**Abstract**: Despite advancements in Autoencoders (AEs) for tasks like dimensionality reduction, representation learning and data generation, they remain vulnerable to adversarial attacks. Variational Autoencoders (VAEs), with their probabilistic approach to disentangling latent spaces, show stronger resistance to such perturbations compared to deterministic AEs; however, their resilience against adversarial inputs is still a concern. This study evaluates the robustness of VAEs against non-targeted adversarial attacks by optimizing minimal sample-specific perturbations to cause maximal damage across diverse demographic subgroups (combinations of age and gender). We investigate two questions: whether there are robustness disparities among subgroups, and what factors contribute to these disparities, such as data scarcity and representation entanglement. Our findings reveal that robustness disparities exist but are not always correlated with the size of the subgroup. By using downstream gender and age classifiers and examining latent embeddings, we highlight the vulnerability of subgroups like older women, who are prone to misclassification due to adversarial perturbations pushing their representations toward those of other subgroups.



## **41. True image construction in quantum-secured single-pixel imaging under spoofing attack**

quant-ph

10 pages, 6 figures

**SubmitDate**: 2024-07-04    [abs](http://arxiv.org/abs/2312.03465v3) [paper-pdf](http://arxiv.org/pdf/2312.03465v3)

**Authors**: Jaesung Heo, Taek Jeong, Nam Hun Park, Yonggi Jo

**Abstract**: In this paper, we introduce a quantum-secured single-pixel imaging (QS-SPI) technique designed to withstand spoofing attacks, wherein adversaries attempt to deceive imaging systems with fake signals. Unlike previous quantum-secured protocols that impose a threshold error rate limiting their operation, even with the existence of true signals, our approach not only identifies spoofing attacks but also facilitates the reconstruction of a true image. Our method involves the analysis of a specific mode correlation of a photon-pair, which is independent of the mode used for image construction, to check security. Through this analysis, we can identify both the targeted image region by the attack and the type of spoofing attack, enabling reconstruction of the true image. A proof-of-principle demonstration employing polarization-correlation of a photon-pair is provided, showcasing successful image reconstruction even under the condition of spoofing signals 2000 times stronger than the true signals. We expect our approach to be applied to quantum-secured signal processing such as quantum target detection or ranging.



## **42. On the Adversarial Robustness of Learning-based Image Compression Against Rate-Distortion Attacks**

eess.IV

**SubmitDate**: 2024-07-04    [abs](http://arxiv.org/abs/2405.07717v2) [paper-pdf](http://arxiv.org/pdf/2405.07717v2)

**Authors**: Chenhao Wu, Qingbo Wu, Haoran Wei, Shuai Chen, Lei Wang, King Ngi Ngan, Fanman Meng, Hongliang Li

**Abstract**: Despite demonstrating superior rate-distortion (RD) performance, learning-based image compression (LIC) algorithms have been found to be vulnerable to malicious perturbations in recent studies. However, the adversarial attacks considered in existing literature remain divergent from real-world scenarios, both in terms of the attack direction and bitrate. Additionally, existing methods focus solely on empirical observations of the model vulnerability, neglecting to identify the origin of it. These limitations hinder the comprehensive investigation and in-depth understanding of the adversarial robustness of LIC algorithms. To address the aforementioned issues, this paper considers the arbitrary nature of the attack direction and the uncontrollable compression ratio faced by adversaries, and presents two practical rate-distortion attack paradigms, i.e., Specific-ratio Rate-Distortion Attack (SRDA) and Agnostic-ratio Rate-Distortion Attack (ARDA). Using the performance variations as indicators, we evaluate the adversarial robustness of eight predominant LIC algorithms against diverse attacks. Furthermore, we propose two novel analytical tools for in-depth analysis, i.e., Entropy Causal Intervention and Layer-wise Distance Magnify Ratio, and reveal that hyperprior significantly increases the bitrate and Inverse Generalized Divisive Normalization (IGDN) significantly amplifies input perturbations when under attack. Lastly, we examine the efficacy of adversarial training and introduce the use of online updating for defense. By comparing their advantages and disadvantages, we provide a reference for constructing more robust LIC algorithms against the rate-distortion attacks.



## **43. Charging Ahead: A Hierarchical Adversarial Framework for Counteracting Advanced Cyber Threats in EV Charging Stations**

cs.CR

Accepted for IEEE VTC Spring 2024

**SubmitDate**: 2024-07-04    [abs](http://arxiv.org/abs/2407.03729v1) [paper-pdf](http://arxiv.org/pdf/2407.03729v1)

**Authors**: Mohammed Al-Mehdhar, Abdullatif Albaseer, Mohamed Abdallah, Ala Al-Fuqaha

**Abstract**: The increasing popularity of electric vehicles (EVs) necessitates robust defenses against sophisticated cyber threats. A significant challenge arises when EVs intentionally provide false information to gain higher charging priority, potentially causing grid instability. While various approaches have been proposed in existing literature to address this issue, they often overlook the possibility of attackers using advanced techniques like deep reinforcement learning (DRL) or other complex deep learning methods to achieve such attacks. In response to this, this paper introduces a hierarchical adversarial framework using DRL (HADRL), which effectively detects stealthy cyberattacks on EV charging stations, especially those leading to denial of charging. Our approach includes a dual approach, where the first scheme leverages DRL to develop advanced and stealthy attack methods that can bypass basic intrusion detection systems (IDS). Second, we implement a DRL-based scheme within the IDS at EV charging stations, aiming to detect and counter these sophisticated attacks. This scheme is trained with datasets created from the first scheme, resulting in a robust and efficient IDS. We evaluated the effectiveness of our framework against the recent literature approaches, and the results show that our IDS can accurately detect deceptive EVs with a low false alarm rate, even when confronted with attacks not represented in the training dataset.



## **44. Pseudorandom Permutations from Random Reversible Circuits**

cs.CC

v2: added references and comparison to subsequent work, removed claim  in previous Section 7.3 with error in proof

**SubmitDate**: 2024-07-04    [abs](http://arxiv.org/abs/2404.14648v2) [paper-pdf](http://arxiv.org/pdf/2404.14648v2)

**Authors**: William He, Ryan O'Donnell

**Abstract**: We study pseudorandomness properties of permutations on $\{0,1\}^n$ computed by random circuits made from reversible $3$-bit gates (permutations on $\{0,1\}^3$). Our main result is that a random circuit of depth $n \cdot \tilde{O}(k^2)$, with each layer consisting of $\approx n/3$ random gates in a fixed nearest-neighbor architecture, yields almost $k$-wise independent permutations. The main technical component is showing that the Markov chain on $k$-tuples of $n$-bit strings induced by a single random $3$-bit nearest-neighbor gate has spectral gap at least $1/n \cdot \tilde{O}(k)$. This improves on the original work of Gowers [Gowers96], who showed a gap of $1/\mathrm{poly}(n,k)$ for one random gate (with non-neighboring inputs); and, on subsequent work [HMMR05,BH08] improving the gap to $\Omega(1/n^2k)$ in the same setting.   From the perspective of cryptography, our result can be seen as a particularly simple/practical block cipher construction that gives provable statistical security against attackers with access to $k$~input-output pairs within few rounds. We also show that the Luby--Rackoff construction of pseudorandom permutations from pseudorandom functions can be implemented with reversible circuits. From this, we make progress on the complexity of the Minimum Reversible Circuit Size Problem (MRCSP), showing that block ciphers of fixed polynomial size are computationally secure against arbitrary polynomial-time adversaries, assuming the existence of one-way functions (OWFs).



## **45. Jailbreaking Black Box Large Language Models in Twenty Queries**

cs.LG

**SubmitDate**: 2024-07-03    [abs](http://arxiv.org/abs/2310.08419v3) [paper-pdf](http://arxiv.org/pdf/2310.08419v3)

**Authors**: Patrick Chao, Alexander Robey, Edgar Dobriban, Hamed Hassani, George J. Pappas, Eric Wong

**Abstract**: There is growing interest in ensuring that large language models (LLMs) align with human values. However, the alignment of such models is vulnerable to adversarial jailbreaks, which coax LLMs into overriding their safety guardrails. The identification of these vulnerabilities is therefore instrumental in understanding inherent weaknesses and preventing future misuse. To this end, we propose Prompt Automatic Iterative Refinement (PAIR), an algorithm that generates semantic jailbreaks with only black-box access to an LLM. PAIR -- which is inspired by social engineering attacks -- uses an attacker LLM to automatically generate jailbreaks for a separate targeted LLM without human intervention. In this way, the attacker LLM iteratively queries the target LLM to update and refine a candidate jailbreak. Empirically, PAIR often requires fewer than twenty queries to produce a jailbreak, which is orders of magnitude more efficient than existing algorithms. PAIR also achieves competitive jailbreaking success rates and transferability on open and closed-source LLMs, including GPT-3.5/4, Vicuna, and Gemini.



## **46. JailBreakV-28K: A Benchmark for Assessing the Robustness of MultiModal Large Language Models against Jailbreak Attacks**

cs.CR

**SubmitDate**: 2024-07-03    [abs](http://arxiv.org/abs/2404.03027v3) [paper-pdf](http://arxiv.org/pdf/2404.03027v3)

**Authors**: Weidi Luo, Siyuan Ma, Xiaogeng Liu, Xiaoyu Guo, Chaowei Xiao

**Abstract**: With the rapid advancements in Multimodal Large Language Models (MLLMs), securing these models against malicious inputs while aligning them with human values has emerged as a critical challenge. In this paper, we investigate an important and unexplored question of whether techniques that successfully jailbreak Large Language Models (LLMs) can be equally effective in jailbreaking MLLMs. To explore this issue, we introduce JailBreakV-28K, a pioneering benchmark designed to assess the transferability of LLM jailbreak techniques to MLLMs, thereby evaluating the robustness of MLLMs against diverse jailbreak attacks. Utilizing a dataset of 2, 000 malicious queries that is also proposed in this paper, we generate 20, 000 text-based jailbreak prompts using advanced jailbreak attacks on LLMs, alongside 8, 000 image-based jailbreak inputs from recent MLLMs jailbreak attacks, our comprehensive dataset includes 28, 000 test cases across a spectrum of adversarial scenarios. Our evaluation of 10 open-source MLLMs reveals a notably high Attack Success Rate (ASR) for attacks transferred from LLMs, highlighting a critical vulnerability in MLLMs that stems from their text-processing capabilities. Our findings underscore the urgent need for future research to address alignment vulnerabilities in MLLMs from both textual and visual inputs.



## **47. On Large Language Models in National Security Applications**

cs.CR

20 pages

**SubmitDate**: 2024-07-03    [abs](http://arxiv.org/abs/2407.03453v1) [paper-pdf](http://arxiv.org/pdf/2407.03453v1)

**Authors**: William N. Caballero, Phillip R. Jenkins

**Abstract**: The overwhelming success of GPT-4 in early 2023 highlighted the transformative potential of large language models (LLMs) across various sectors, including national security. This article explores the implications of LLM integration within national security contexts, analyzing their potential to revolutionize information processing, decision-making, and operational efficiency. Whereas LLMs offer substantial benefits, such as automating tasks and enhancing data analysis, they also pose significant risks, including hallucinations, data privacy concerns, and vulnerability to adversarial attacks. Through their coupling with decision-theoretic principles and Bayesian reasoning, LLMs can significantly improve decision-making processes within national security organizations. Namely, LLMs can facilitate the transition from data to actionable decisions, enabling decision-makers to quickly receive and distill available information with less manpower. Current applications within the US Department of Defense and beyond are explored, e.g., the USAF's use of LLMs for wargaming and automatic summarization, that illustrate their potential to streamline operations and support decision-making. However, these applications necessitate rigorous safeguards to ensure accuracy and reliability. The broader implications of LLM integration extend to strategic planning, international relations, and the broader geopolitical landscape, with adversarial nations leveraging LLMs for disinformation and cyber operations, emphasizing the need for robust countermeasures. Despite exhibiting "sparks" of artificial general intelligence, LLMs are best suited for supporting roles rather than leading strategic decisions. Their use in training and wargaming can provide valuable insights and personalized learning experiences for military personnel, thereby improving operational readiness.



## **48. Correlated Privacy Mechanisms for Differentially Private Distributed Mean Estimation**

cs.IT

**SubmitDate**: 2024-07-03    [abs](http://arxiv.org/abs/2407.03289v1) [paper-pdf](http://arxiv.org/pdf/2407.03289v1)

**Authors**: Sajani Vithana, Viveck R. Cadambe, Flavio P. Calmon, Haewon Jeong

**Abstract**: Differentially private distributed mean estimation (DP-DME) is a fundamental building block in privacy-preserving federated learning, where a central server estimates the mean of $d$-dimensional vectors held by $n$ users while ensuring $(\epsilon,\delta)$-DP. Local differential privacy (LDP) and distributed DP with secure aggregation (SecAgg) are the most common notions of DP used in DP-DME settings with an untrusted server. LDP provides strong resilience to dropouts, colluding users, and malicious server attacks, but suffers from poor utility. In contrast, SecAgg-based DP-DME achieves an $O(n)$ utility gain over LDP in DME, but requires increased communication and computation overheads and complex multi-round protocols to handle dropouts and malicious attacks. In this work, we propose CorDP-DME, a novel DP-DME mechanism that spans the gap between DME with LDP and distributed DP, offering a favorable balance between utility and resilience to dropout and collusion. CorDP-DME is based on correlated Gaussian noise, ensuring DP without the perfect conditional privacy guarantees of SecAgg-based approaches. We provide an information-theoretic analysis of CorDP-DME, and derive theoretical guarantees for utility under any given privacy parameters and dropout/colluding user thresholds. Our results demonstrate that (anti) correlated Gaussian DP mechanisms can significantly improve utility in mean estimation tasks compared to LDP -- even in adversarial settings -- while maintaining better resilience to dropouts and attacks compared to distributed DP.



## **49. Self-Evaluation as a Defense Against Adversarial Attacks on LLMs**

cs.LG

8 pages, 7 figures

**SubmitDate**: 2024-07-03    [abs](http://arxiv.org/abs/2407.03234v1) [paper-pdf](http://arxiv.org/pdf/2407.03234v1)

**Authors**: Hannah Brown, Leon Lin, Kenji Kawaguchi, Michael Shieh

**Abstract**: When LLMs are deployed in sensitive, human-facing settings, it is crucial that they do not output unsafe, biased, or privacy-violating outputs. For this reason, models are both trained and instructed to refuse to answer unsafe prompts such as "Tell me how to build a bomb." We find that, despite these safeguards, it is possible to break model defenses simply by appending a space to the end of a model's input. In a study of eight open-source models, we demonstrate that this acts as a strong enough attack to cause the majority of models to generate harmful outputs with very high success rates. We examine the causes of this behavior, finding that the contexts in which single spaces occur in tokenized training data encourage models to generate lists when prompted, overriding training signals to refuse to answer unsafe requests. Our findings underscore the fragile state of current model alignment and promote the importance of developing more robust alignment methods. Code and data will be made available at https://github.com/Linlt-leon/Adversarial-Alignments.



## **50. Venomancer: Towards Imperceptible and Target-on-Demand Backdoor Attacks in Federated Learning**

cs.CV

**SubmitDate**: 2024-07-03    [abs](http://arxiv.org/abs/2407.03144v1) [paper-pdf](http://arxiv.org/pdf/2407.03144v1)

**Authors**: Son Nguyen, Thinh Nguyen, Khoa Doan, Kok-Seng Wong

**Abstract**: Federated Learning (FL) is a distributed machine learning approach that maintains data privacy by training on decentralized data sources. Similar to centralized machine learning, FL is also susceptible to backdoor attacks. Most backdoor attacks in FL assume a predefined target class and require control over a large number of clients or knowledge of benign clients' information. Furthermore, they are not imperceptible and are easily detected by human inspection due to clear artifacts left on the poison data. To overcome these challenges, we propose Venomancer, an effective backdoor attack that is imperceptible and allows target-on-demand. Specifically, imperceptibility is achieved by using a visual loss function to make the poison data visually indistinguishable from the original data. Target-on-demand property allows the attacker to choose arbitrary target classes via conditional adversarial training. Additionally, experiments showed that the method is robust against state-of-the-art defenses such as Norm Clipping, Weak DP, Krum, and Multi-Krum. The source code is available at https://anonymous.4open.science/r/Venomancer-3426.



