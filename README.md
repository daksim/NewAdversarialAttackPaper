# Latest Adversarial Attack Papers
**update at 2023-10-25 10:04:50**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. Diffusion-Based Adversarial Purification for Speaker Verification**

eess.AS

**SubmitDate**: 2023-10-24    [abs](http://arxiv.org/abs/2310.14270v2) [paper-pdf](http://arxiv.org/pdf/2310.14270v2)

**Authors**: Yibo Bai, Xiao-Lei Zhang

**Abstract**: Recently, automatic speaker verification (ASV) based on deep learning is easily contaminated by adversarial attacks, which is a new type of attack that injects imperceptible perturbations to audio signals so as to make ASV produce wrong decisions. This poses a significant threat to the security and reliability of ASV systems. To address this issue, we propose a Diffusion-Based Adversarial Purification (DAP) method that enhances the robustness of ASV systems against such adversarial attacks. Our method leverages a conditional denoising diffusion probabilistic model to effectively purify the adversarial examples and mitigate the impact of perturbations. DAP first introduces controlled noise into adversarial examples, and then performs a reverse denoising process to reconstruct clean audio. Experimental results demonstrate the efficacy of the proposed DAP in enhancing the security of ASV and meanwhile minimizing the distortion of the purified audio signals.



## **2. A Survey on LLM-generated Text Detection: Necessity, Methods, and Future Directions**

cs.CL

**SubmitDate**: 2023-10-24    [abs](http://arxiv.org/abs/2310.14724v2) [paper-pdf](http://arxiv.org/pdf/2310.14724v2)

**Authors**: Junchao Wu, Shu Yang, Runzhe Zhan, Yulin Yuan, Derek F. Wong, Lidia S. Chao

**Abstract**: The powerful ability to understand, follow, and generate complex language emerging from large language models (LLMs) makes LLM-generated text flood many areas of our daily lives at an incredible speed and is widely accepted by humans. As LLMs continue to expand, there is an imperative need to develop detectors that can detect LLM-generated text. This is crucial to mitigate potential misuse of LLMs and safeguard realms like artistic expression and social networks from harmful influence of LLM-generated content. The LLM-generated text detection aims to discern if a piece of text was produced by an LLM, which is essentially a binary classification task. The detector techniques have witnessed notable advancements recently, propelled by innovations in watermarking techniques, zero-shot methods, fine-turning LMs methods, adversarial learning methods, LLMs as detectors, and human-assisted methods. In this survey, we collate recent research breakthroughs in this area and underscore the pressing need to bolster detector research. We also delve into prevalent datasets, elucidating their limitations and developmental requirements. Furthermore, we analyze various LLM-generated text detection paradigms, shedding light on challenges like out-of-distribution problems, potential attacks, and data ambiguity. Conclusively, we highlight interesting directions for future research in LLM-generated text detection to advance the implementation of responsible artificial intelligence (AI). Our aim with this survey is to provide a clear and comprehensive introduction for newcomers while also offering seasoned researchers a valuable update in the field of LLM-generated text detection. The useful resources are publicly available at: https://github.com/NLP2CT/LLM-generated-Text-Detection.



## **3. Momentum Gradient-based Untargeted Attack on Hypergraph Neural Networks**

cs.LG

**SubmitDate**: 2023-10-24    [abs](http://arxiv.org/abs/2310.15656v1) [paper-pdf](http://arxiv.org/pdf/2310.15656v1)

**Authors**: Yang Chen, Stjepan Picek, Zhonglin Ye, Zhaoyang Wang, Haixing Zhao

**Abstract**: Hypergraph Neural Networks (HGNNs) have been successfully applied in various hypergraph-related tasks due to their excellent higher-order representation capabilities. Recent works have shown that deep learning models are vulnerable to adversarial attacks. Most studies on graph adversarial attacks have focused on Graph Neural Networks (GNNs), and the study of adversarial attacks on HGNNs remains largely unexplored. In this paper, we try to reduce this gap. We design a new HGNNs attack model for the untargeted attack, namely MGHGA, which focuses on modifying node features. We consider the process of HGNNs training and use a surrogate model to implement the attack before hypergraph modeling. Specifically, MGHGA consists of two parts: feature selection and feature modification. We use a momentum gradient mechanism to choose the attack node features in the feature selection module. In the feature modification module, we use two feature generation approaches (direct modification and sign gradient) to enable MGHGA to be employed on discrete and continuous datasets. We conduct extensive experiments on five benchmark datasets to validate the attack performance of MGHGA in the node and the visual object classification tasks. The results show that MGHGA improves performance by an average of 2% compared to the than the baselines.



## **4. Deceptive Fairness Attacks on Graphs via Meta Learning**

cs.LG

23 pages, 11 tables

**SubmitDate**: 2023-10-24    [abs](http://arxiv.org/abs/2310.15653v1) [paper-pdf](http://arxiv.org/pdf/2310.15653v1)

**Authors**: Jian Kang, Yinglong Xia, Ross Maciejewski, Jiebo Luo, Hanghang Tong

**Abstract**: We study deceptive fairness attacks on graphs to answer the following question: How can we achieve poisoning attacks on a graph learning model to exacerbate the bias deceptively? We answer this question via a bi-level optimization problem and propose a meta learning-based framework named FATE. FATE is broadly applicable with respect to various fairness definitions and graph learning models, as well as arbitrary choices of manipulation operations. We further instantiate FATE to attack statistical parity and individual fairness on graph neural networks. We conduct extensive experimental evaluations on real-world datasets in the task of semi-supervised node classification. The experimental results demonstrate that FATE could amplify the bias of graph neural networks with or without fairness consideration while maintaining the utility on the downstream task. We hope this paper provides insights into the adversarial robustness of fair graph learning and can shed light on designing robust and fair graph learning in future studies.



## **5. Facial Data Minimization: Shallow Model as Your Privacy Filter**

cs.CR

14 pages, 11 figures

**SubmitDate**: 2023-10-24    [abs](http://arxiv.org/abs/2310.15590v1) [paper-pdf](http://arxiv.org/pdf/2310.15590v1)

**Authors**: Yuwen Pu, Jiahao Chen, Jiayu Pan, Hao li, Diqun Yan, Xuhong Zhang, Shouling Ji

**Abstract**: Face recognition service has been used in many fields and brings much convenience to people. However, once the user's facial data is transmitted to a service provider, the user will lose control of his/her private data. In recent years, there exist various security and privacy issues due to the leakage of facial data. Although many privacy-preserving methods have been proposed, they usually fail when they are not accessible to adversaries' strategies or auxiliary data. Hence, in this paper, by fully considering two cases of uploading facial images and facial features, which are very typical in face recognition service systems, we proposed a data privacy minimization transformation (PMT) method. This method can process the original facial data based on the shallow model of authorized services to obtain the obfuscated data. The obfuscated data can not only maintain satisfactory performance on authorized models and restrict the performance on other unauthorized models but also prevent original privacy data from leaking by AI methods and human visual theft. Additionally, since a service provider may execute preprocessing operations on the received data, we also propose an enhanced perturbation method to improve the robustness of PMT. Besides, to authorize one facial image to multiple service models simultaneously, a multiple restriction mechanism is proposed to improve the scalability of PMT. Finally, we conduct extensive experiments and evaluate the effectiveness of the proposed PMT in defending against face reconstruction, data abuse, and face attribute estimation attacks. These experimental results demonstrate that PMT performs well in preventing facial data abuse and privacy leakage while maintaining face recognition accuracy.



## **6. LLM Self Defense: By Self Examination, LLMs Know They Are Being Tricked**

cs.CL

**SubmitDate**: 2023-10-24    [abs](http://arxiv.org/abs/2308.07308v3) [paper-pdf](http://arxiv.org/pdf/2308.07308v3)

**Authors**: Mansi Phute, Alec Helbling, Matthew Hull, ShengYun Peng, Sebastian Szyller, Cory Cornelius, Duen Horng Chau

**Abstract**: Large language models (LLMs) are popular for high-quality text generation but can produce harmful content, even when aligned with human values through reinforcement learning. Adversarial prompts can bypass their safety measures. We propose LLM Self Defense, a simple approach to defend against these attacks by having an LLM screen the induced responses. Our method does not require any fine-tuning, input preprocessing, or iterative output generation. Instead, we incorporate the generated content into a pre-defined prompt and employ another instance of an LLM to analyze the text and predict whether it is harmful. We test LLM Self Defense on GPT 3.5 and Llama 2, two of the current most prominent LLMs against various types of attacks, such as forcefully inducing affirmative responses to prompts and prompt engineering attacks. Notably, LLM Self Defense succeeds in reducing the attack success rate to virtually 0 using both GPT 3.5 and Llama 2.



## **7. Fast Propagation is Better: Accelerating Single-Step Adversarial Training via Sampling Subnetworks**

cs.CV

**SubmitDate**: 2023-10-24    [abs](http://arxiv.org/abs/2310.15444v1) [paper-pdf](http://arxiv.org/pdf/2310.15444v1)

**Authors**: Xiaojun Jia, Jianshu Li, Jindong Gu, Yang Bai, Xiaochun Cao

**Abstract**: Adversarial training has shown promise in building robust models against adversarial examples. A major drawback of adversarial training is the computational overhead introduced by the generation of adversarial examples. To overcome this limitation, adversarial training based on single-step attacks has been explored. Previous work improves the single-step adversarial training from different perspectives, e.g., sample initialization, loss regularization, and training strategy. Almost all of them treat the underlying model as a black box. In this work, we propose to exploit the interior building blocks of the model to improve efficiency. Specifically, we propose to dynamically sample lightweight subnetworks as a surrogate model during training. By doing this, both the forward and backward passes can be accelerated for efficient adversarial training. Besides, we provide theoretical analysis to show the model robustness can be improved by the single-step adversarial training with sampled subnetworks. Furthermore, we propose a novel sampling strategy where the sampling varies from layer to layer and from iteration to iteration. Compared with previous methods, our method not only reduces the training cost but also achieves better model robustness. Evaluations on a series of popular datasets demonstrate the effectiveness of the proposed FB-Better. Our code has been released at https://github.com/jiaxiaojunQAQ/FP-Better.



## **8. Unsupervised Federated Learning: A Federated Gradient EM Algorithm for Heterogeneous Mixture Models with Robustness against Adversarial Attacks**

stat.ML

43 pages, 1 figure

**SubmitDate**: 2023-10-23    [abs](http://arxiv.org/abs/2310.15330v1) [paper-pdf](http://arxiv.org/pdf/2310.15330v1)

**Authors**: Ye Tian, Haolei Weng, Yang Feng

**Abstract**: While supervised federated learning approaches have enjoyed significant success, the domain of unsupervised federated learning remains relatively underexplored. In this paper, we introduce a novel federated gradient EM algorithm designed for the unsupervised learning of mixture models with heterogeneous mixture proportions across tasks. We begin with a comprehensive finite-sample theory that holds for general mixture models, then apply this general theory on Gaussian Mixture Models (GMMs) and Mixture of Regressions (MoRs) to characterize the explicit estimation error of model parameters and mixture proportions. Our proposed federated gradient EM algorithm demonstrates several key advantages: adaptability to unknown task similarity, resilience against adversarial attacks on a small fraction of data sources, protection of local data privacy, and computational and communication efficiency.



## **9. GRASP: Accelerating Shortest Path Attacks via Graph Attention**

cs.LG

**SubmitDate**: 2023-10-23    [abs](http://arxiv.org/abs/2310.07980v2) [paper-pdf](http://arxiv.org/pdf/2310.07980v2)

**Authors**: Zohair Shafi, Benjamin A. Miller, Ayan Chatterjee, Tina Eliassi-Rad, Rajmonda S. Caceres

**Abstract**: Recent advances in machine learning (ML) have shown promise in aiding and accelerating classical combinatorial optimization algorithms. ML-based speed ups that aim to learn in an end to end manner (i.e., directly output the solution) tend to trade off run time with solution quality. Therefore, solutions that are able to accelerate existing solvers while maintaining their performance guarantees, are of great interest. We consider an APX-hard problem, where an adversary aims to attack shortest paths in a graph by removing the minimum number of edges. We propose the GRASP algorithm: Graph Attention Accelerated Shortest Path Attack, an ML aided optimization algorithm that achieves run times up to 10x faster, while maintaining the quality of solution generated. GRASP uses a graph attention network to identify a smaller subgraph containing the combinatorial solution, thus effectively reducing the input problem size. Additionally, we demonstrate how careful representation of the input graph, including node features that correlate well with the optimization task, can highlight important structure in the optimization solution.



## **10. AutoDAN: Automatic and Interpretable Adversarial Attacks on Large Language Models**

cs.CR

**SubmitDate**: 2023-10-23    [abs](http://arxiv.org/abs/2310.15140v1) [paper-pdf](http://arxiv.org/pdf/2310.15140v1)

**Authors**: Sicheng Zhu, Ruiyi Zhang, Bang An, Gang Wu, Joe Barrow, Zichao Wang, Furong Huang, Ani Nenkova, Tong Sun

**Abstract**: Safety alignment of Large Language Models (LLMs) can be compromised with manual jailbreak attacks and (automatic) adversarial attacks. Recent work suggests that patching LLMs against these attacks is possible: manual jailbreak attacks are human-readable but often limited and public, making them easy to block; adversarial attacks generate gibberish prompts that can be detected using perplexity-based filters. In this paper, we show that these solutions may be too optimistic. We propose an interpretable adversarial attack, \texttt{AutoDAN}, that combines the strengths of both types of attacks. It automatically generates attack prompts that bypass perplexity-based filters while maintaining a high attack success rate like manual jailbreak attacks. These prompts are interpretable and diverse, exhibiting strategies commonly used in manual jailbreak attacks, and transfer better than their non-readable counterparts when using limited training data or a single proxy model. We also customize \texttt{AutoDAN}'s objective to leak system prompts, another jailbreak application not addressed in the adversarial attack literature. %, demonstrating the versatility of the approach. We can also customize the objective of \texttt{AutoDAN} to leak system prompts, beyond the ability to elicit harmful content from the model, demonstrating the versatility of the approach. Our work provides a new way to red-team LLMs and to understand the mechanism of jailbreak attacks.



## **11. On the Detection of Image-Scaling Attacks in Machine Learning**

cs.CR

Accepted at ACSAC'23

**SubmitDate**: 2023-10-23    [abs](http://arxiv.org/abs/2310.15085v1) [paper-pdf](http://arxiv.org/pdf/2310.15085v1)

**Authors**: Erwin Quiring, Andreas Müller, Konrad Rieck

**Abstract**: Image scaling is an integral part of machine learning and computer vision systems. Unfortunately, this preprocessing step is vulnerable to so-called image-scaling attacks where an attacker makes unnoticeable changes to an image so that it becomes a new image after scaling. This opens up new ways for attackers to control the prediction or to improve poisoning and backdoor attacks. While effective techniques exist to prevent scaling attacks, their detection has not been rigorously studied yet. Consequently, it is currently not possible to reliably spot these attacks in practice.   This paper presents the first in-depth systematization and analysis of detection methods for image-scaling attacks. We identify two general detection paradigms and derive novel methods from them that are simple in design yet significantly outperform previous work. We demonstrate the efficacy of these methods in a comprehensive evaluation with all major learning platforms and scaling algorithms. First, we show that image-scaling attacks modifying the entire scaled image can be reliably detected even under an adaptive adversary. Second, we find that our methods provide strong detection performance even if only minor parts of the image are manipulated. As a result, we can introduce a novel protection layer against image-scaling attacks.



## **12. Enhancing Adversarial Contrastive Learning via Adversarial Invariant Regularization**

cs.LG

NeurIPS 2023

**SubmitDate**: 2023-10-23    [abs](http://arxiv.org/abs/2305.00374v2) [paper-pdf](http://arxiv.org/pdf/2305.00374v2)

**Authors**: Xilie Xu, Jingfeng Zhang, Feng Liu, Masashi Sugiyama, Mohan Kankanhalli

**Abstract**: Adversarial contrastive learning (ACL) is a technique that enhances standard contrastive learning (SCL) by incorporating adversarial data to learn a robust representation that can withstand adversarial attacks and common corruptions without requiring costly annotations. To improve transferability, the existing work introduced the standard invariant regularization (SIR) to impose style-independence property to SCL, which can exempt the impact of nuisance style factors in the standard representation. However, it is unclear how the style-independence property benefits ACL-learned robust representations. In this paper, we leverage the technique of causal reasoning to interpret the ACL and propose adversarial invariant regularization (AIR) to enforce independence from style factors. We regulate the ACL using both SIR and AIR to output the robust representation. Theoretically, we show that AIR implicitly encourages the representational distance between different views of natural data and their adversarial variants to be independent of style factors. Empirically, our experimental results show that invariant regularization significantly improves the performance of state-of-the-art ACL methods in terms of both standard generalization and robustness on downstream tasks. To the best of our knowledge, we are the first to apply causal reasoning to interpret ACL and develop AIR for enhancing ACL-learned robust representations. Our source code is at https://github.com/GodXuxilie/Enhancing_ACL_via_AIR.



## **13. Efficient Adversarial Contrastive Learning via Robustness-Aware Coreset Selection**

cs.LG

NeurIPS 2023 Spotlight

**SubmitDate**: 2023-10-23    [abs](http://arxiv.org/abs/2302.03857v4) [paper-pdf](http://arxiv.org/pdf/2302.03857v4)

**Authors**: Xilie Xu, Jingfeng Zhang, Feng Liu, Masashi Sugiyama, Mohan Kankanhalli

**Abstract**: Adversarial contrastive learning (ACL) does not require expensive data annotations but outputs a robust representation that withstands adversarial attacks and also generalizes to a wide range of downstream tasks. However, ACL needs tremendous running time to generate the adversarial variants of all training data, which limits its scalability to large datasets. To speed up ACL, this paper proposes a robustness-aware coreset selection (RCS) method. RCS does not require label information and searches for an informative subset that minimizes a representational divergence, which is the distance of the representation between natural data and their virtual adversarial variants. The vanilla solution of RCS via traversing all possible subsets is computationally prohibitive. Therefore, we theoretically transform RCS into a surrogate problem of submodular maximization, of which the greedy search is an efficient solution with an optimality guarantee for the original problem. Empirically, our comprehensive results corroborate that RCS can speed up ACL by a large margin without significantly hurting the robustness transferability. Notably, to the best of our knowledge, we are the first to conduct ACL efficiently on the large-scale ImageNet-1K dataset to obtain an effective robust representation via RCS. Our source code is at https://github.com/GodXuxilie/Efficient_ACL_via_RCS.



## **14. Kidnapping Deep Learning-based Multirotors using Optimized Flying Adversarial Patches**

cs.RO

Accepted at MRS 2023, 7 pages, 5 figures. arXiv admin note:  substantial text overlap with arXiv:2305.12859

**SubmitDate**: 2023-10-23    [abs](http://arxiv.org/abs/2308.00344v2) [paper-pdf](http://arxiv.org/pdf/2308.00344v2)

**Authors**: Pia Hanfeld, Khaled Wahba, Marina M. -C. Höhne, Michael Bussmann, Wolfgang Hönig

**Abstract**: Autonomous flying robots, such as multirotors, often rely on deep learning models that make predictions based on a camera image, e.g. for pose estimation. These models can predict surprising results if applied to input images outside the training domain. This fault can be exploited by adversarial attacks, for example, by computing small images, so-called adversarial patches, that can be placed in the environment to manipulate the neural network's prediction. We introduce flying adversarial patches, where multiple images are mounted on at least one other flying robot and therefore can be placed anywhere in the field of view of a victim multirotor. By introducing the attacker robots, the system is extended to an adversarial multi-robot system. For an effective attack, we compare three methods that simultaneously optimize multiple adversarial patches and their position in the input image. We show that our methods scale well with the number of adversarial patches. Moreover, we demonstrate physical flights with two robots, where we employ a novel attack policy that uses the computed adversarial patches to kidnap a robot that was supposed to follow a human.



## **15. Beyond Hard Samples: Robust and Effective Grammatical Error Correction with Cycle Self-Augmenting**

cs.CL

**SubmitDate**: 2023-10-23    [abs](http://arxiv.org/abs/2310.13321v2) [paper-pdf](http://arxiv.org/pdf/2310.13321v2)

**Authors**: Zecheng Tang, Kaifeng Qi, Juntao Li, Min Zhang

**Abstract**: Recent studies have revealed that grammatical error correction methods in the sequence-to-sequence paradigm are vulnerable to adversarial attack, and simply utilizing adversarial examples in the pre-training or post-training process can significantly enhance the robustness of GEC models to certain types of attack without suffering too much performance loss on clean data. In this paper, we further conduct a thorough robustness evaluation of cutting-edge GEC methods for four different types of adversarial attacks and propose a simple yet very effective Cycle Self-Augmenting (CSA) method accordingly. By leveraging the augmenting data from the GEC models themselves in the post-training process and introducing regularization data for cycle training, our proposed method can effectively improve the model robustness of well-trained GEC models with only a few more training epochs as an extra cost. More concretely, further training on the regularization data can prevent the GEC models from over-fitting on easy-to-learn samples and thus can improve the generalization capability and robustness towards unseen data (adversarial noise/samples). Meanwhile, the self-augmented data can provide more high-quality pseudo pairs to improve model performance on the original testing data. Experiments on four benchmark datasets and seven strong models indicate that our proposed training method can significantly enhance the robustness of four types of attacks without using purposely built adversarial examples in training. Evaluation results on clean data further confirm that our proposed CSA method significantly improves the performance of four baselines and yields nearly comparable results with other state-of-the-art models. Our code is available at https://github.com/ZetangForward/CSA-GEC.



## **16. Semantic-Aware Adversarial Training for Reliable Deep Hashing Retrieval**

cs.CV

**SubmitDate**: 2023-10-23    [abs](http://arxiv.org/abs/2310.14637v1) [paper-pdf](http://arxiv.org/pdf/2310.14637v1)

**Authors**: Xu Yuan, Zheng Zhang, Xunguang Wang, Lin Wu

**Abstract**: Deep hashing has been intensively studied and successfully applied in large-scale image retrieval systems due to its efficiency and effectiveness. Recent studies have recognized that the existence of adversarial examples poses a security threat to deep hashing models, that is, adversarial vulnerability. Notably, it is challenging to efficiently distill reliable semantic representatives for deep hashing to guide adversarial learning, and thereby it hinders the enhancement of adversarial robustness of deep hashing-based retrieval models. Moreover, current researches on adversarial training for deep hashing are hard to be formalized into a unified minimax structure. In this paper, we explore Semantic-Aware Adversarial Training (SAAT) for improving the adversarial robustness of deep hashing models. Specifically, we conceive a discriminative mainstay features learning (DMFL) scheme to construct semantic representatives for guiding adversarial learning in deep hashing. Particularly, our DMFL with the strict theoretical guarantee is adaptively optimized in a discriminative learning manner, where both discriminative and semantic properties are jointly considered. Moreover, adversarial examples are fabricated by maximizing the Hamming distance between the hash codes of adversarial samples and mainstay features, the efficacy of which is validated in the adversarial attack trials. Further, we, for the first time, formulate the formalized adversarial training of deep hashing into a unified minimax optimization under the guidance of the generated mainstay codes. Extensive experiments on benchmark datasets show superb attack performance against the state-of-the-art algorithms, meanwhile, the proposed adversarial training can effectively eliminate adversarial perturbations for trustworthy deep hashing-based retrieval. Our code is available at https://github.com/xandery-geek/SAAT.



## **17. TrojLLM: A Black-box Trojan Prompt Attack on Large Language Models**

cs.CR

Accepted by NeurIPS'23

**SubmitDate**: 2023-10-23    [abs](http://arxiv.org/abs/2306.06815v2) [paper-pdf](http://arxiv.org/pdf/2306.06815v2)

**Authors**: Jiaqi Xue, Mengxin Zheng, Ting Hua, Yilin Shen, Yepeng Liu, Ladislau Boloni, Qian Lou

**Abstract**: Large Language Models (LLMs) are progressively being utilized as machine learning services and interface tools for various applications. However, the security implications of LLMs, particularly in relation to adversarial and Trojan attacks, remain insufficiently examined. In this paper, we propose TrojLLM, an automatic and black-box framework to effectively generate universal and stealthy triggers. When these triggers are incorporated into the input data, the LLMs' outputs can be maliciously manipulated. Moreover, the framework also supports embedding Trojans within discrete prompts, enhancing the overall effectiveness and precision of the triggers' attacks. Specifically, we propose a trigger discovery algorithm for generating universal triggers for various inputs by querying victim LLM-based APIs using few-shot data samples. Furthermore, we introduce a novel progressive Trojan poisoning algorithm designed to generate poisoned prompts that retain efficacy and transferability across a diverse range of models. Our experiments and results demonstrate TrojLLM's capacity to effectively insert Trojans into text prompts in real-world black-box LLM APIs including GPT-3.5 and GPT-4, while maintaining exceptional performance on clean test sets. Our work sheds light on the potential security risks in current models and offers a potential defensive approach. The source code of TrojLLM is available at https://github.com/UCF-ML-Research/TrojLLM.



## **18. ADoPT: LiDAR Spoofing Attack Detection Based on Point-Level Temporal Consistency**

cs.CV

BMVC 2023 (17 pages, 13 figures, and 1 table)

**SubmitDate**: 2023-10-23    [abs](http://arxiv.org/abs/2310.14504v1) [paper-pdf](http://arxiv.org/pdf/2310.14504v1)

**Authors**: Minkyoung Cho, Yulong Cao, Zixiang Zhou, Z. Morley Mao

**Abstract**: Deep neural networks (DNNs) are increasingly integrated into LiDAR (Light Detection and Ranging)-based perception systems for autonomous vehicles (AVs), requiring robust performance under adversarial conditions. We aim to address the challenge of LiDAR spoofing attacks, where attackers inject fake objects into LiDAR data and fool AVs to misinterpret their environment and make erroneous decisions. However, current defense algorithms predominantly depend on perception outputs (i.e., bounding boxes) thus face limitations in detecting attackers given the bounding boxes are generated by imperfect perception models processing limited points, acquired based on the ego vehicle's viewpoint. To overcome these limitations, we propose a novel framework, named ADoPT (Anomaly Detection based on Point-level Temporal consistency), which quantitatively measures temporal consistency across consecutive frames and identifies abnormal objects based on the coherency of point clusters. In our evaluation using the nuScenes dataset, our algorithm effectively counters various LiDAR spoofing attacks, achieving a low (< 10%) false positive ratio (FPR) and high (> 85%) true positive ratio (TPR), outperforming existing state-of-the-art defense methods, CARLO and 3D-TC2. Furthermore, our evaluation demonstrates the promising potential for accurate attack detection across various road environments.



## **19. Probabilistic and Semantic Descriptions of Image Manifolds and Their Applications**

cs.CV

26 pages, 17 figures, 1 table

**SubmitDate**: 2023-10-22    [abs](http://arxiv.org/abs/2307.02881v4) [paper-pdf](http://arxiv.org/pdf/2307.02881v4)

**Authors**: Peter Tu, Zhaoyuan Yang, Richard Hartley, Zhiwei Xu, Jing Zhang, Yiwei Fu, Dylan Campbell, Jaskirat Singh, Tianyu Wang

**Abstract**: This paper begins with a description of methods for estimating image probability density functions that reflects the observation that such data is usually constrained to lie in restricted regions of the high-dimensional image space-not every pattern of pixels is an image. It is common to say that images lie on a lower-dimensional manifold in the high-dimensional space. However, it is not the case that all points on the manifold have an equal probability of being images. Images are unevenly distributed on the manifold, and our task is to devise ways to model this distribution as a probability distribution. We therefore consider popular generative models. For our purposes, generative/probabilistic models should have the properties of 1) sample generation: the possibility to sample from this distribution with the modelled density function, and 2) probability computation: given a previously unseen sample from the dataset of interest, one should be able to compute its probability, at least up to a normalising constant. To this end, we investigate the use of methods such as normalising flow and diffusion models. We then show how semantic interpretations are used to describe points on the manifold. To achieve this, we consider an emergent language framework that uses variational encoders for a disentangled representation of points that reside on a given manifold. Trajectories between points on a manifold can then be described as evolving semantic descriptions. We also show that such probabilistic descriptions (bounded) can be used to improve semantic consistency by constructing defences against adversarial attacks. We evaluate our methods with improved semantic robustness and OoD detection capability, explainable and editable semantic interpolation, and improved classification accuracy under patch attacks. We also discuss the limitation in diffusion models.



## **20. Attacks Meet Interpretability (AmI) Evaluation and Findings**

cs.CR

Need to withdraw it. The current work needs to be changed at a large  extent which would take a longer time

**SubmitDate**: 2023-10-22    [abs](http://arxiv.org/abs/2310.08808v3) [paper-pdf](http://arxiv.org/pdf/2310.08808v3)

**Authors**: Qian Ma, Ziping Ye, Shagufta Mehnaz

**Abstract**: To investigate the effectiveness of the model explanation in detecting adversarial examples, we reproduce the results of two papers, Attacks Meet Interpretability: Attribute-steered Detection of Adversarial Samples and Is AmI (Attacks Meet Interpretability) Robust to Adversarial Examples. And then conduct experiments and case studies to identify the limitations of both works. We find that Attacks Meet Interpretability(AmI) is highly dependent on the selection of hyperparameters. Therefore, with a different hyperparameter choice, AmI is still able to detect Nicholas Carlini's attack. Finally, we propose recommendations for future work on the evaluation of defense techniques such as AmI.



## **21. HoSNN: Adversarially-Robust Homeostatic Spiking Neural Networks with Adaptive Firing Thresholds**

cs.NE

**SubmitDate**: 2023-10-22    [abs](http://arxiv.org/abs/2308.10373v2) [paper-pdf](http://arxiv.org/pdf/2308.10373v2)

**Authors**: Hejia Geng, Peng Li

**Abstract**: Spiking neural networks (SNNs) offer promise for efficient and powerful neurally inspired computation. Common to other types of neural networks, however, SNNs face the severe issue of vulnerability to adversarial attacks. We present the first study that draws inspiration from neural homeostasis to develop a bio-inspired solution that counters the susceptibilities of SNNs to adversarial onslaughts. At the heart of our approach is a novel threshold-adapting leaky integrate-and-fire (TA-LIF) neuron model, which we adopt to construct the proposed adversarially robust homeostatic SNN (HoSNN). Distinct from traditional LIF models, our TA-LIF model incorporates a self-stabilizing dynamic thresholding mechanism, curtailing adversarial noise propagation and safeguarding the robustness of HoSNNs in an unsupervised manner. Theoretical analysis is presented to shed light on the stability and convergence properties of the TA-LIF neurons, underscoring their superior dynamic robustness under input distributional shifts over traditional LIF neurons. Remarkably, without explicit adversarial training, our HoSNNs demonstrate inherent robustness on CIFAR-10, with accuracy improvements to 72.6% and 54.19% against FGSM and PGD attacks, up from 20.97% and 0.6%, respectively. Furthermore, with minimal FGSM adversarial training, our HoSNNs surpass previous models by 29.99% under FGSM and 47.83% under PGD attacks on CIFAR-10. Our findings offer a new perspective on harnessing biological principles for bolstering SNNs adversarial robustness and defense, paving the way to more resilient neuromorphic computing.



## **22. DANAA: Towards transferable attacks with double adversarial neuron attribution**

cs.CV

Accepted by 19th International Conference on Advanced Data Mining and  Applications. (ADMA 2023)

**SubmitDate**: 2023-10-22    [abs](http://arxiv.org/abs/2310.10427v2) [paper-pdf](http://arxiv.org/pdf/2310.10427v2)

**Authors**: Zhibo Jin, Zhiyu Zhu, Xinyi Wang, Jiayu Zhang, Jun Shen, Huaming Chen

**Abstract**: While deep neural networks have excellent results in many fields, they are susceptible to interference from attacking samples resulting in erroneous judgments. Feature-level attacks are one of the effective attack types, which targets the learnt features in the hidden layers to improve its transferability across different models. Yet it is observed that the transferability has been largely impacted by the neuron importance estimation results. In this paper, a double adversarial neuron attribution attack method, termed `DANAA', is proposed to obtain more accurate feature importance estimation. In our method, the model outputs are attributed to the middle layer based on an adversarial non-linear path. The goal is to measure the weight of individual neurons and retain the features that are more important towards transferability. We have conducted extensive experiments on the benchmark datasets to demonstrate the state-of-the-art performance of our method. Our code is available at: https://github.com/Davidjinzb/DANAA



## **23. Assessing the Influence of Different Types of Probing on Adversarial Decision-Making in a Deception Game**

cs.CR

**SubmitDate**: 2023-10-22    [abs](http://arxiv.org/abs/2310.10662v2) [paper-pdf](http://arxiv.org/pdf/2310.10662v2)

**Authors**: Md Abu Sayed, Mohammad Ariful Islam Khan, Bryant A Allsup, Joshua Zamora, Palvi Aggarwal

**Abstract**: Deception, which includes leading cyber-attackers astray with false information, has shown to be an effective method of thwarting cyber-attacks. There has been little investigation of the effect of probing action costs on adversarial decision-making, despite earlier studies on deception in cybersecurity focusing primarily on variables like network size and the percentage of honeypots utilized in games. Understanding human decision-making when prompted with choices of various costs is essential in many areas such as in cyber security. In this paper, we will use a deception game (DG) to examine different costs of probing on adversarial decisions. To achieve this we utilized an IBLT model and a delayed feedback mechanism to mimic knowledge of human actions. Our results were taken from an even split of deception and no deception to compare each influence. It was concluded that probing was slightly taken less as the cost of probing increased. The proportion of attacks stayed relatively the same as the cost of probing increased. Although a constant cost led to a slight decrease in attacks. Overall, our results concluded that the different probing costs do not have an impact on the proportion of attacks whereas it had a slightly noticeable impact on the proportion of probing.



## **24. Language Model Unalignment: Parametric Red-Teaming to Expose Hidden Harms and Biases**

cs.CL

Under Review

**SubmitDate**: 2023-10-22    [abs](http://arxiv.org/abs/2310.14303v1) [paper-pdf](http://arxiv.org/pdf/2310.14303v1)

**Authors**: Rishabh Bhardwaj, Soujanya Poria

**Abstract**: Red-teaming has been a widely adopted way to evaluate the harmfulness of Large Language Models (LLMs). It aims to jailbreak a model's safety behavior to make it act as a helpful agent disregarding the harmfulness of the query. Existing methods are primarily based on input text-based red-teaming such as adversarial prompts, low-resource prompts, or contextualized prompts to condition the model in a way to bypass its safe behavior. Bypassing the guardrails uncovers hidden harmful information and biases in the model that are left untreated or newly introduced by its safety training. However, prompt-based attacks fail to provide such a diagnosis owing to their low attack success rate, and applicability to specific models. In this paper, we present a new perspective on LLM safety research i.e., parametric red-teaming through Unalignment. It simply (instruction) tunes the model parameters to break model guardrails that are not deeply rooted in the model's behavior. Unalignment using as few as 100 examples can significantly bypass commonly referred to as CHATGPT, to the point where it responds with an 88% success rate to harmful queries on two safety benchmark datasets. On open-source models such as VICUNA-7B and LLAMA-2-CHAT 7B AND 13B, it shows an attack success rate of more than 91%. On bias evaluations, Unalignment exposes inherent biases in safety-aligned models such as CHATGPT and LLAMA- 2-CHAT where the model's responses are strongly biased and opinionated 64% of the time.



## **25. CT-GAT: Cross-Task Generative Adversarial Attack based on Transferability**

cs.CL

Accepted to EMNLP 2023 main conference

**SubmitDate**: 2023-10-22    [abs](http://arxiv.org/abs/2310.14265v1) [paper-pdf](http://arxiv.org/pdf/2310.14265v1)

**Authors**: Minxuan Lv, Chengwei Dai, Kun Li, Wei Zhou, Songlin Hu

**Abstract**: Neural network models are vulnerable to adversarial examples, and adversarial transferability further increases the risk of adversarial attacks. Current methods based on transferability often rely on substitute models, which can be impractical and costly in real-world scenarios due to the unavailability of training data and the victim model's structural details. In this paper, we propose a novel approach that directly constructs adversarial examples by extracting transferable features across various tasks. Our key insight is that adversarial transferability can extend across different tasks. Specifically, we train a sequence-to-sequence generative model named CT-GAT using adversarial sample data collected from multiple tasks to acquire universal adversarial features and generate adversarial examples for different tasks. We conduct experiments on ten distinct datasets, and the results demonstrate that our method achieves superior attack performance with small cost.



## **26. LoFT: Local Proxy Fine-tuning For Improving Transferability Of Adversarial Attacks Against Large Language Model**

cs.CL

**SubmitDate**: 2023-10-21    [abs](http://arxiv.org/abs/2310.04445v2) [paper-pdf](http://arxiv.org/pdf/2310.04445v2)

**Authors**: Muhammad Ahmed Shah, Roshan Sharma, Hira Dhamyal, Raphael Olivier, Ankit Shah, Joseph Konan, Dareen Alharthi, Hazim T Bukhari, Massa Baali, Soham Deshmukh, Michael Kuhlmann, Bhiksha Raj, Rita Singh

**Abstract**: It has been shown that Large Language Model (LLM) alignments can be circumvented by appending specially crafted attack suffixes with harmful queries to elicit harmful responses. To conduct attacks against private target models whose characterization is unknown, public models can be used as proxies to fashion the attack, with successful attacks being transferred from public proxies to private target models. The success rate of attack depends on how closely the proxy model approximates the private model. We hypothesize that for attacks to be transferrable, it is sufficient if the proxy can approximate the target model in the neighborhood of the harmful query. Therefore, in this paper, we propose \emph{Local Fine-Tuning (LoFT)}, \textit{i.e.}, fine-tuning proxy models on similar queries that lie in the lexico-semantic neighborhood of harmful queries to decrease the divergence between the proxy and target models. First, we demonstrate three approaches to prompt private target models to obtain similar queries given harmful queries. Next, we obtain data for local fine-tuning by eliciting responses from target models for the generated similar queries. Then, we optimize attack suffixes to generate attack prompts and evaluate the impact of our local fine-tuning on the attack's success rate. Experiments show that local fine-tuning of proxy models improves attack transferability and increases attack success rate by $39\%$, $7\%$, and $0.5\%$ (absolute) on target models ChatGPT, GPT-4, and Claude respectively.



## **27. Toward Stronger Textual Attack Detectors**

cs.CL

Findings EMNLP 2023

**SubmitDate**: 2023-10-21    [abs](http://arxiv.org/abs/2310.14001v1) [paper-pdf](http://arxiv.org/pdf/2310.14001v1)

**Authors**: Pierre Colombo, Marine Picot, Nathan Noiry, Guillaume Staerman, Pablo Piantanida

**Abstract**: The landscape of available textual adversarial attacks keeps growing, posing severe threats and raising concerns regarding the deep NLP system's integrity. However, the crucial problem of defending against malicious attacks has only drawn the attention of the NLP community. The latter is nonetheless instrumental in developing robust and trustworthy systems. This paper makes two important contributions in this line of search: (i) we introduce LAROUSSE, a new framework to detect textual adversarial attacks and (ii) we introduce STAKEOUT, a new benchmark composed of nine popular attack methods, three datasets, and two pre-trained models. LAROUSSE is ready-to-use in production as it is unsupervised, hyperparameter-free, and non-differentiable, protecting it against gradient-based methods. Our new benchmark STAKEOUT allows for a robust evaluation framework: we conduct extensive numerical experiments which demonstrate that LAROUSSE outperforms previous methods, and which allows to identify interesting factors of detection rate variations.



## **28. Adversarial Image Generation by Spatial Transformation in Perceptual Colorspaces**

cs.CV

**SubmitDate**: 2023-10-21    [abs](http://arxiv.org/abs/2310.13950v1) [paper-pdf](http://arxiv.org/pdf/2310.13950v1)

**Authors**: Ayberk Aydin, Alptekin Temizel

**Abstract**: Deep neural networks are known to be vulnerable to adversarial perturbations. The amount of these perturbations are generally quantified using $L_p$ metrics, such as $L_0$, $L_2$ and $L_\infty$. However, even when the measured perturbations are small, they tend to be noticeable by human observers since $L_p$ distance metrics are not representative of human perception. On the other hand, humans are less sensitive to changes in colorspace. In addition, pixel shifts in a constrained neighborhood are hard to notice. Motivated by these observations, we propose a method that creates adversarial examples by applying spatial transformations, which creates adversarial examples by changing the pixel locations independently to chrominance channels of perceptual colorspaces such as $YC_{b}C_{r}$ and $CIELAB$, instead of making an additive perturbation or manipulating pixel values directly. In a targeted white-box attack setting, the proposed method is able to obtain competitive fooling rates with very high confidence. The experimental evaluations show that the proposed method has favorable results in terms of approximate perceptual distance between benign and adversarially generated images. The source code is publicly available at https://github.com/ayberkydn/stadv-torch



## **29. Adversarial Robustness Unhardening via Backdoor Attacks in Federated Learning**

cs.LG

8 pages, 6 main pages of text, 4 figures, 2 tables. Made for a  Neurips workshop on backdoor attacks

**SubmitDate**: 2023-10-21    [abs](http://arxiv.org/abs/2310.11594v2) [paper-pdf](http://arxiv.org/pdf/2310.11594v2)

**Authors**: Taejin Kim, Jiarui Li, Shubhranshu Singh, Nikhil Madaan, Carlee Joe-Wong

**Abstract**: In today's data-driven landscape, the delicate equilibrium between safeguarding user privacy and unleashing data potential stands as a paramount concern. Federated learning, which enables collaborative model training without necessitating data sharing, has emerged as a privacy-centric solution. This decentralized approach brings forth security challenges, notably poisoning and backdoor attacks where malicious entities inject corrupted data. Our research, initially spurred by test-time evasion attacks, investigates the intersection of adversarial training and backdoor attacks within federated learning, introducing Adversarial Robustness Unhardening (ARU). ARU is employed by a subset of adversaries to intentionally undermine model robustness during decentralized training, rendering models susceptible to a broader range of evasion attacks. We present extensive empirical experiments evaluating ARU's impact on adversarial training and existing robust aggregation defenses against poisoning and backdoor attacks. Our findings inform strategies for enhancing ARU to counter current defensive measures and highlight the limitations of existing defenses, offering insights into bolstering defenses against ARU.



## **30. Characterizing Internal Evasion Attacks in Federated Learning**

cs.LG

16 pages, 8 figures (14 images if counting sub-figures separately),  Camera ready version for AISTATS 2023, longer version of paper submitted to  CrossFL 2022 poster workshop, code available at  (https://github.com/tj-kim/pFedDef_v1)

**SubmitDate**: 2023-10-21    [abs](http://arxiv.org/abs/2209.08412v3) [paper-pdf](http://arxiv.org/pdf/2209.08412v3)

**Authors**: Taejin Kim, Shubhranshu Singh, Nikhil Madaan, Carlee Joe-Wong

**Abstract**: Federated learning allows for clients in a distributed system to jointly train a machine learning model. However, clients' models are vulnerable to attacks during the training and testing phases. In this paper, we address the issue of adversarial clients performing "internal evasion attacks": crafting evasion attacks at test time to deceive other clients. For example, adversaries may aim to deceive spam filters and recommendation systems trained with federated learning for monetary gain. The adversarial clients have extensive information about the victim model in a federated learning setting, as weight information is shared amongst clients. We are the first to characterize the transferability of such internal evasion attacks for different learning methods and analyze the trade-off between model accuracy and robustness depending on the degree of similarities in client data. We show that adversarial training defenses in the federated learning setting only display limited improvements against internal attacks. However, combining adversarial training with personalized federated learning frameworks increases relative internal attack robustness by 60% compared to federated adversarial training and performs well under limited system resources.



## **31. The Hidden Adversarial Vulnerabilities of Medical Federated Learning**

cs.LG

**SubmitDate**: 2023-10-21    [abs](http://arxiv.org/abs/2310.13893v1) [paper-pdf](http://arxiv.org/pdf/2310.13893v1)

**Authors**: Erfan Darzi, Florian Dubost, Nanna. M. Sijtsema, P. M. A van Ooijen

**Abstract**: In this paper, we delve into the susceptibility of federated medical image analysis systems to adversarial attacks. Our analysis uncovers a novel exploitation avenue: using gradient information from prior global model updates, adversaries can enhance the efficiency and transferability of their attacks. Specifically, we demonstrate that single-step attacks (e.g. FGSM), when aptly initialized, can outperform the efficiency of their iterative counterparts but with reduced computational demand. Our findings underscore the need to revisit our understanding of AI security in federated healthcare settings.



## **32. Specify Robust Causal Representation from Mixed Observations**

cs.LG

arXiv admin note: substantial text overlap with arXiv:2202.08388

**SubmitDate**: 2023-10-21    [abs](http://arxiv.org/abs/2310.13892v1) [paper-pdf](http://arxiv.org/pdf/2310.13892v1)

**Authors**: Mengyue Yang, Xinyu Cai, Furui Liu, Weinan Zhang, Jun Wang

**Abstract**: Learning representations purely from observations concerns the problem of learning a low-dimensional, compact representation which is beneficial to prediction models. Under the hypothesis that the intrinsic latent factors follow some casual generative models, we argue that by learning a causal representation, which is the minimal sufficient causes of the whole system, we can improve the robustness and generalization performance of machine learning models. In this paper, we develop a learning method to learn such representation from observational data by regularizing the learning procedure with mutual information measures, according to the hypothetical factored causal graph. We theoretically and empirically show that the models trained with the learned causal representations are more robust under adversarial attacks and distribution shifts compared with baselines. The supplementary materials are available at https://github.com/ymy $4323460 / \mathrm{CaRI} /$.



## **33. Adversarial Attacks on Fairness of Graph Neural Networks**

cs.LG

32 pages, 5 figures

**SubmitDate**: 2023-10-20    [abs](http://arxiv.org/abs/2310.13822v1) [paper-pdf](http://arxiv.org/pdf/2310.13822v1)

**Authors**: Binchi Zhang, Yushun Dong, Chen Chen, Yada Zhu, Minnan Luo, Jundong Li

**Abstract**: Fairness-aware graph neural networks (GNNs) have gained a surge of attention as they can reduce the bias of predictions on any demographic group (e.g., female) in graph-based applications. Although these methods greatly improve the algorithmic fairness of GNNs, the fairness can be easily corrupted by carefully designed adversarial attacks. In this paper, we investigate the problem of adversarial attacks on fairness of GNNs and propose G-FairAttack, a general framework for attacking various types of fairness-aware GNNs in terms of fairness with an unnoticeable effect on prediction utility. In addition, we propose a fast computation technique to reduce the time complexity of G-FairAttack. The experimental study demonstrates that G-FairAttack successfully corrupts the fairness of different types of GNNs while keeping the attack unnoticeable. Our study on fairness attacks sheds light on potential vulnerabilities in fairness-aware GNNs and guides further research on the robustness of GNNs in terms of fairness. The open-source code is available at https://github.com/zhangbinchi/G-FairAttack.



## **34. Data-Free Knowledge Distillation Using Adversarially Perturbed OpenGL Shader Images**

cs.CV

**SubmitDate**: 2023-10-20    [abs](http://arxiv.org/abs/2310.13782v1) [paper-pdf](http://arxiv.org/pdf/2310.13782v1)

**Authors**: Logan Frank, Jim Davis

**Abstract**: Knowledge distillation (KD) has been a popular and effective method for model compression. One important assumption of KD is that the original training dataset is always available. However, this is not always the case due to privacy concerns and more. In recent years, "data-free" KD has emerged as a growing research topic which focuses on the scenario of performing KD when no data is provided. Many methods rely on a generator network to synthesize examples for distillation (which can be difficult to train) and can frequently produce images that are visually similar to the original dataset, which raises questions surrounding whether privacy is completely preserved. In this work, we propose a new approach to data-free KD that utilizes unnatural OpenGL images, combined with large amounts of data augmentation and adversarial attacks, to train a student network. We demonstrate that our approach achieves state-of-the-art results for a variety of datasets/networks and is more stable than existing generator-based data-free KD methods. Source code will be available in the future.



## **35. FLTracer: Accurate Poisoning Attack Provenance in Federated Learning**

cs.CR

18 pages, 27 figures

**SubmitDate**: 2023-10-20    [abs](http://arxiv.org/abs/2310.13424v1) [paper-pdf](http://arxiv.org/pdf/2310.13424v1)

**Authors**: Xinyu Zhang, Qingyu Liu, Zhongjie Ba, Yuan Hong, Tianhang Zheng, Feng Lin, Li Lu, Kui Ren

**Abstract**: Federated Learning (FL) is a promising distributed learning approach that enables multiple clients to collaboratively train a shared global model. However, recent studies show that FL is vulnerable to various poisoning attacks, which can degrade the performance of global models or introduce backdoors into them. In this paper, we first conduct a comprehensive study on prior FL attacks and detection methods. The results show that all existing detection methods are only effective against limited and specific attacks. Most detection methods suffer from high false positives, which lead to significant performance degradation, especially in not independent and identically distributed (non-IID) settings. To address these issues, we propose FLTracer, the first FL attack provenance framework to accurately detect various attacks and trace the attack time, objective, type, and poisoned location of updates. Different from existing methodologies that rely solely on cross-client anomaly detection, we propose a Kalman filter-based cross-round detection to identify adversaries by seeking the behavior changes before and after the attack. Thus, this makes it resilient to data heterogeneity and is effective even in non-IID settings. To further improve the accuracy of our detection method, we employ four novel features and capture their anomalies with the joint decisions. Extensive evaluations show that FLTracer achieves an average true positive rate of over $96.88\%$ at an average false positive rate of less than $2.67\%$, significantly outperforming SOTA detection methods. \footnote{Code is available at \url{https://github.com/Eyr3/FLTracer}.}



## **36. Vulnerabilities in Video Quality Assessment Models: The Challenge of Adversarial Attacks**

cs.CV

**SubmitDate**: 2023-10-20    [abs](http://arxiv.org/abs/2309.13609v3) [paper-pdf](http://arxiv.org/pdf/2309.13609v3)

**Authors**: Ao-Xiang Zhang, Yu Ran, Weixuan Tang, Yuan-Gen Wang

**Abstract**: No-Reference Video Quality Assessment (NR-VQA) plays an essential role in improving the viewing experience of end-users. Driven by deep learning, recent NR-VQA models based on Convolutional Neural Networks (CNNs) and Transformers have achieved outstanding performance. To build a reliable and practical assessment system, it is of great necessity to evaluate their robustness. However, such issue has received little attention in the academic community. In this paper, we make the first attempt to evaluate the robustness of NR-VQA models against adversarial attacks, and propose a patch-based random search method for black-box attack. Specifically, considering both the attack effect on quality score and the visual quality of adversarial video, the attack problem is formulated as misleading the estimated quality score under the constraint of just-noticeable difference (JND). Built upon such formulation, a novel loss function called Score-Reversed Boundary Loss is designed to push the adversarial video's estimated quality score far away from its ground-truth score towards a specific boundary, and the JND constraint is modeled as a strict $L_2$ and $L_\infty$ norm restriction. By this means, both white-box and black-box attacks can be launched in an effective and imperceptible manner. The source code is available at https://github.com/GZHU-DVL/AttackVQA.



## **37. Verifiable Learning for Robust Tree Ensembles**

cs.LG

19 pages, 5 figures; full version of the revised paper accepted at  ACM CCS 2023 with corrected proofs of Lemma A.6 and Lemma A.7

**SubmitDate**: 2023-10-20    [abs](http://arxiv.org/abs/2305.03626v3) [paper-pdf](http://arxiv.org/pdf/2305.03626v3)

**Authors**: Stefano Calzavara, Lorenzo Cazzaro, Giulio Ermanno Pibiri, Nicola Prezza

**Abstract**: Verifying the robustness of machine learning models against evasion attacks at test time is an important research problem. Unfortunately, prior work established that this problem is NP-hard for decision tree ensembles, hence bound to be intractable for specific inputs. In this paper, we identify a restricted class of decision tree ensembles, called large-spread ensembles, which admit a security verification algorithm running in polynomial time. We then propose a new approach called verifiable learning, which advocates the training of such restricted model classes which are amenable for efficient verification. We show the benefits of this idea by designing a new training algorithm that automatically learns a large-spread decision tree ensemble from labelled data, thus enabling its security verification in polynomial time. Experimental results on public datasets confirm that large-spread ensembles trained using our algorithm can be verified in a matter of seconds, using standard commercial hardware. Moreover, large-spread ensembles are more robust than traditional ensembles against evasion attacks, at the cost of an acceptable loss of accuracy in the non-adversarial setting.



## **38. An LLM can Fool Itself: A Prompt-Based Adversarial Attack**

cs.CR

**SubmitDate**: 2023-10-20    [abs](http://arxiv.org/abs/2310.13345v1) [paper-pdf](http://arxiv.org/pdf/2310.13345v1)

**Authors**: Xilie Xu, Keyi Kong, Ning Liu, Lizhen Cui, Di Wang, Jingfeng Zhang, Mohan Kankanhalli

**Abstract**: The wide-ranging applications of large language models (LLMs), especially in safety-critical domains, necessitate the proper evaluation of the LLM's adversarial robustness. This paper proposes an efficient tool to audit the LLM's adversarial robustness via a prompt-based adversarial attack (PromptAttack). PromptAttack converts adversarial textual attacks into an attack prompt that can cause the victim LLM to output the adversarial sample to fool itself. The attack prompt is composed of three important components: (1) original input (OI) including the original sample and its ground-truth label, (2) attack objective (AO) illustrating a task description of generating a new sample that can fool itself without changing the semantic meaning, and (3) attack guidance (AG) containing the perturbation instructions to guide the LLM on how to complete the task by perturbing the original sample at character, word, and sentence levels, respectively. Besides, we use a fidelity filter to ensure that PromptAttack maintains the original semantic meanings of the adversarial examples. Further, we enhance the attack power of PromptAttack by ensembling adversarial examples at different perturbation levels. Comprehensive empirical results using Llama2 and GPT-3.5 validate that PromptAttack consistently yields a much higher attack success rate compared to AdvGLUE and AdvGLUE++. Interesting findings include that a simple emoji can easily mislead GPT-3.5 to make wrong predictions.



## **39. Mitigating Backdoor Poisoning Attacks through the Lens of Spurious Correlation**

cs.CL

accepted to EMNLP2023 (main conference)

**SubmitDate**: 2023-10-20    [abs](http://arxiv.org/abs/2305.11596v2) [paper-pdf](http://arxiv.org/pdf/2305.11596v2)

**Authors**: Xuanli He, Qiongkai Xu, Jun Wang, Benjamin Rubinstein, Trevor Cohn

**Abstract**: Modern NLP models are often trained over large untrusted datasets, raising the potential for a malicious adversary to compromise model behaviour. For instance, backdoors can be implanted through crafting training instances with a specific textual trigger and a target label. This paper posits that backdoor poisoning attacks exhibit \emph{spurious correlation} between simple text features and classification labels, and accordingly, proposes methods for mitigating spurious correlation as means of defence. Our empirical study reveals that the malicious triggers are highly correlated to their target labels; therefore such correlations are extremely distinguishable compared to those scores of benign features, and can be used to filter out potentially problematic instances. Compared with several existing defences, our defence method significantly reduces attack success rates across backdoor attacks, and in the case of insertion-based attacks, our method provides a near-perfect defence.



## **40. Generalizable Lightweight Proxy for Robust NAS against Diverse Perturbations**

cs.LG

NeurIPS 2023, Code is available at  https://github.com/HyeonjeongHa/CRoZe

**SubmitDate**: 2023-10-20    [abs](http://arxiv.org/abs/2306.05031v2) [paper-pdf](http://arxiv.org/pdf/2306.05031v2)

**Authors**: Hyeonjeong Ha, Minseon Kim, Sung Ju Hwang

**Abstract**: Recent neural architecture search (NAS) frameworks have been successful in finding optimal architectures for given conditions (e.g., performance or latency). However, they search for optimal architectures in terms of their performance on clean images only, while robustness against various types of perturbations or corruptions is crucial in practice. Although there exist several robust NAS frameworks that tackle this issue by integrating adversarial training into one-shot NAS, however, they are limited in that they only consider robustness against adversarial attacks and require significant computational resources to discover optimal architectures for a single task, which makes them impractical in real-world scenarios. To address these challenges, we propose a novel lightweight robust zero-cost proxy that considers the consistency across features, parameters, and gradients of both clean and perturbed images at the initialization state. Our approach facilitates an efficient and rapid search for neural architectures capable of learning generalizable features that exhibit robustness across diverse perturbations. The experimental results demonstrate that our proxy can rapidly and efficiently search for neural architectures that are consistently robust against various perturbations on multiple benchmark datasets and diverse search spaces, largely outperforming existing clean zero-shot NAS and robust NAS with reduced search cost.



## **41. Detecting Shared Data Manipulation in Distributed Optimization Algorithms**

eess.SY

**SubmitDate**: 2023-10-20    [abs](http://arxiv.org/abs/2310.13252v1) [paper-pdf](http://arxiv.org/pdf/2310.13252v1)

**Authors**: Mohannad Alkhraijah, Rachel Harris, Samuel Litchfield, David Huggins, Daniel K. Molzahn

**Abstract**: This paper investigates the vulnerability of the Alternating Direction Method of Multipliers (ADMM) algorithm to shared data manipulation, with a focus on solving optimal power flow (OPF) problems. Deliberate data manipulation may cause the ADMM algorithm to converge to suboptimal solutions. We derive two sufficient conditions for detecting data manipulation based on the theoretical convergence trajectory of the ADMM algorithm. We evaluate the detection conditions' performance on three data manipulation strategies we previously proposed: simple, feedback, and bilevel optimization attacks. We then extend these three data manipulation strategies to avoid detection by considering both the detection conditions and a neural network (NN) detection model in the attacks. We also propose an adversarial NN training framework to detect shared data manipulation. We illustrate the performance of our data manipulation strategy and detection framework on OPF problems. The results show that the proposed detection conditions successfully detect most of the data manipulation attacks. However, a bilevel optimization attack strategy that incorporates the detection methods may avoid being detected. Countering this, our proposed adversarial training framework detects all the instances of the bilevel optimization attack.



## **42. Towards Robust Pruning: An Adaptive Knowledge-Retention Pruning Strategy for Language Models**

cs.CL

**SubmitDate**: 2023-10-19    [abs](http://arxiv.org/abs/2310.13191v1) [paper-pdf](http://arxiv.org/pdf/2310.13191v1)

**Authors**: Jianwei Li, Qi Lei, Wei Cheng, Dongkuan Xu

**Abstract**: The pruning objective has recently extended beyond accuracy and sparsity to robustness in language models. Despite this, existing methods struggle to enhance robustness against adversarial attacks when continually increasing model sparsity and require a retraining process. As humans step into the era of large language models, these issues become increasingly prominent. This paper proposes that the robustness of language models is proportional to the extent of pre-trained knowledge they encompass. Accordingly, we introduce a post-training pruning strategy designed to faithfully replicate the embedding space and feature space of dense language models, aiming to conserve more pre-trained knowledge during the pruning process. In this setup, each layer's reconstruction error not only originates from itself but also includes cumulative error from preceding layers, followed by an adaptive rectification. Compared to other state-of-art baselines, our approach demonstrates a superior balance between accuracy, sparsity, robustness, and pruning cost with BERT on datasets SST2, IMDB, and AGNews, marking a significant stride towards robust pruning in language models.



## **43. Quantum Key Distribution for Critical Infrastructures: Towards Cyber Physical Security for Hydropower and Dams**

quant-ph

20 pages, 7 figures, 6 tables

**SubmitDate**: 2023-10-19    [abs](http://arxiv.org/abs/2310.13100v1) [paper-pdf](http://arxiv.org/pdf/2310.13100v1)

**Authors**: Adrien Green, Jeremy Lawrence, George Siopsis, Nicholas Peters, Ali Passian

**Abstract**: Hydropower facilities are often remotely monitored or controlled from a centralized remote-control room. Additionally, major component manufacturers monitor the performance of installed components. While these communications enable efficiencies and increased reliability, they also expand the cyber-attack surface. Communications may use the internet to remote control a facility's control systems, or it may involve sending control commands over a network from a control room to a machine. The content could be encrypted and decrypted using a public key to protect the communicated information. These cryptographic encoding and decoding schemes have been shown to be vulnerable, a situation which is being exacerbated as more advances are made in computer technologies such as quantum computing. In contrast, quantum key distribution (QKD) is not based upon a computational problem, and offers an alternative to conventional public-key cryptography. Although the underlying mechanism of QKD ensures that any attempt by an adversary to observe the quantum part of the protocol will result in a detectable signature as an increased error rate, potentially even preventing key generation, it serves as a warning for further investigation. When the error rate is low enough and enough photons have been detected, a shared private key can be generated known only to the sender and receiver. We describe how this novel technology and its several modalities could benefit the critical infrastructures of dams or hydropower facilities. The presented discussions may be viewed as a precursor to a quantum cybersecurity roadmap for the identification of relevant threats and mitigation.



## **44. PatchCURE: Improving Certifiable Robustness, Model Utility, and Computation Efficiency of Adversarial Patch Defenses**

cs.CV

**SubmitDate**: 2023-10-19    [abs](http://arxiv.org/abs/2310.13076v1) [paper-pdf](http://arxiv.org/pdf/2310.13076v1)

**Authors**: Chong Xiang, Tong Wu, Sihui Dai, Jonathan Petit, Suman Jana, Prateek Mittal

**Abstract**: State-of-the-art defenses against adversarial patch attacks can now achieve strong certifiable robustness with a marginal drop in model utility. However, this impressive performance typically comes at the cost of 10-100x more inference-time computation compared to undefended models -- the research community has witnessed an intense three-way trade-off between certifiable robustness, model utility, and computation efficiency. In this paper, we propose a defense framework named PatchCURE to approach this trade-off problem. PatchCURE provides sufficient "knobs" for tuning defense performance and allows us to build a family of defenses: the most robust PatchCURE instance can match the performance of any existing state-of-the-art defense (without efficiency considerations); the most efficient PatchCURE instance has similar inference efficiency as undefended models. Notably, PatchCURE achieves state-of-the-art robustness and utility performance across all different efficiency levels, e.g., 16-23% absolute clean accuracy and certified robust accuracy advantages over prior defenses when requiring computation efficiency to be close to undefended models. The family of PatchCURE defenses enables us to flexibly choose appropriate defenses to satisfy given computation and/or utility constraints in practice.



## **45. Categorical composable cryptography: extended version**

cs.CR

Extended version of arXiv:2105.05949 which appeared in FoSSaCS 2022.  Very minor layout updates to the previous version as requested by the journal

**SubmitDate**: 2023-10-19    [abs](http://arxiv.org/abs/2208.13232v3) [paper-pdf](http://arxiv.org/pdf/2208.13232v3)

**Authors**: Anne Broadbent, Martti Karvonen

**Abstract**: We formalize the simulation paradigm of cryptography in terms of category theory and show that protocols secure against abstract attacks form a symmetric monoidal category, thus giving an abstract model of composable security definitions in cryptography. Our model is able to incorporate computational security, set-up assumptions and various attack models such as colluding or independently acting subsets of adversaries in a modular, flexible fashion. We conclude by using string diagrams to rederive the security of the one-time pad, correctness of Diffie-Hellman key exchange and no-go results concerning the limits of bipartite and tripartite cryptography, ruling out e.g., composable commitments and broadcasting. On the way, we exhibit two categorical constructions of resource theories that might be of independent interest: one capturing resources shared among multiple parties and one capturing resource conversions that succeed asymptotically.



## **46. OODRobustBench: benchmarking and analyzing adversarial robustness under distribution shift**

cs.LG

in submission

**SubmitDate**: 2023-10-19    [abs](http://arxiv.org/abs/2310.12793v1) [paper-pdf](http://arxiv.org/pdf/2310.12793v1)

**Authors**: Lin Li, Yifei Wang, Chawin Sitawarin, Michael Spratling

**Abstract**: Existing works have made great progress in improving adversarial robustness, but typically test their method only on data from the same distribution as the training data, i.e. in-distribution (ID) testing. As a result, it is unclear how such robustness generalizes under input distribution shifts, i.e. out-of-distribution (OOD) testing. This is a concerning omission as such distribution shifts are unavoidable when methods are deployed in the wild. To address this issue we propose a benchmark named OODRobustBench to comprehensively assess OOD adversarial robustness using 23 dataset-wise shifts (i.e. naturalistic shifts in input distribution) and 6 threat-wise shifts (i.e., unforeseen adversarial threat models). OODRobustBench is used to assess 706 robust models using 60.7K adversarial evaluations. This large-scale analysis shows that: 1) adversarial robustness suffers from a severe OOD generalization issue; 2) ID robustness correlates strongly with OOD robustness, in a positive linear way, under many distribution shifts. The latter enables the prediction of OOD robustness from ID robustness. Based on this, we are able to predict the upper limit of OOD robustness for existing robust training schemes. The results suggest that achieving OOD robustness requires designing novel methods beyond the conventional ones. Last, we discover that extra data, data augmentation, advanced model architectures and particular regularization approaches can improve OOD robustness. Noticeably, the discovered training schemes, compared to the baseline, exhibit dramatically higher robustness under threat shift while keeping high ID robustness, demonstrating new promising solutions for robustness against both multi-attack and unforeseen attacks.



## **47. TorKameleon: Improving Tor's Censorship Resistance with K-anonymization and Media-based Covert Channels**

cs.CR

**SubmitDate**: 2023-10-19    [abs](http://arxiv.org/abs/2303.17544v3) [paper-pdf](http://arxiv.org/pdf/2303.17544v3)

**Authors**: Afonso Vilalonga, João S. Resende, Henrique Domingos

**Abstract**: Anonymity networks like Tor significantly enhance online privacy but are vulnerable to correlation attacks by state-level adversaries. While covert channels encapsulated in media protocols, particularly WebRTC-based encapsulation, have demonstrated effectiveness against passive traffic correlation attacks, their resilience against active correlation attacks remains unexplored, and their compatibility with Tor has been limited. This paper introduces TorKameleon, a censorship evasion solution designed to protect Tor users from both passive and active correlation attacks. TorKameleon employs K-anonymization techniques to fragment and reroute traffic through multiple TorKameleon proxies, while also utilizing covert WebRTC-based channels or TLS tunnels to encapsulate user traffic.



## **48. WaveAttack: Asymmetric Frequency Obfuscation-based Backdoor Attacks Against Deep Neural Networks**

cs.CV

**SubmitDate**: 2023-10-19    [abs](http://arxiv.org/abs/2310.11595v2) [paper-pdf](http://arxiv.org/pdf/2310.11595v2)

**Authors**: Jun Xia, Zhihao Yue, Yingbo Zhou, Zhiwei Ling, Xian Wei, Mingsong Chen

**Abstract**: Due to the popularity of Artificial Intelligence (AI) technology, numerous backdoor attacks are designed by adversaries to mislead deep neural network predictions by manipulating training samples and training processes. Although backdoor attacks are effective in various real scenarios, they still suffer from the problems of both low fidelity of poisoned samples and non-negligible transfer in latent space, which make them easily detectable by existing backdoor detection algorithms. To overcome the weakness, this paper proposes a novel frequency-based backdoor attack method named WaveAttack, which obtains image high-frequency features through Discrete Wavelet Transform (DWT) to generate backdoor triggers. Furthermore, we introduce an asymmetric frequency obfuscation method, which can add an adaptive residual in the training and inference stage to improve the impact of triggers and further enhance the effectiveness of WaveAttack. Comprehensive experimental results show that WaveAttack not only achieves higher stealthiness and effectiveness, but also outperforms state-of-the-art (SOTA) backdoor attack methods in the fidelity of images by up to 28.27\% improvement in PSNR, 1.61\% improvement in SSIM, and 70.59\% reduction in IS.



## **49. Learn from the Past: A Proxy based Adversarial Defense Framework to Boost Robustness**

cs.LG

16 Pages

**SubmitDate**: 2023-10-19    [abs](http://arxiv.org/abs/2310.12713v1) [paper-pdf](http://arxiv.org/pdf/2310.12713v1)

**Authors**: Yaohua Liu, Jiaxin Gao, Zhu Liu, Xianghao Jiao, Xin Fan, Risheng Liu

**Abstract**: In light of the vulnerability of deep learning models to adversarial samples and the ensuing security issues, a range of methods, including Adversarial Training (AT) as a prominent representative, aimed at enhancing model robustness against various adversarial attacks, have seen rapid development. However, existing methods essentially assist the current state of target model to defend against parameter-oriented adversarial attacks with explicit or implicit computation burdens, which also suffers from unstable convergence behavior due to inconsistency of optimization trajectories. Diverging from previous work, this paper reconsiders the update rule of target model and corresponding deficiency to defend based on its current state. By introducing the historical state of the target model as a proxy, which is endowed with much prior information for defense, we formulate a two-stage update rule, resulting in a general adversarial defense framework, which we refer to as `LAST' ({\bf L}earn from the P{\bf ast}). Besides, we devise a Self Distillation (SD) based defense objective to constrain the update process of the proxy model without the introduction of larger teacher models. Experimentally, we demonstrate consistent and significant performance enhancements by refining a series of single-step and multi-step AT methods (e.g., up to $\bf 9.2\%$ and $\bf 20.5\%$ improvement of Robust Accuracy (RA) on CIFAR10 and CIFAR100 datasets, respectively) across various datasets, backbones and attack modalities, and validate its ability to enhance training stability and ameliorate catastrophic overfitting issues meanwhile.



## **50. Generating Robust Adversarial Examples against Online Social Networks (OSNs)**

cs.MM

26 pages, 9 figures

**SubmitDate**: 2023-10-19    [abs](http://arxiv.org/abs/2310.12708v1) [paper-pdf](http://arxiv.org/pdf/2310.12708v1)

**Authors**: Jun Liu, Jiantao Zhou, Haiwei Wu, Weiwei Sun, Jinyu Tian

**Abstract**: Online Social Networks (OSNs) have blossomed into prevailing transmission channels for images in the modern era. Adversarial examples (AEs) deliberately designed to mislead deep neural networks (DNNs) are found to be fragile against the inevitable lossy operations conducted by OSNs. As a result, the AEs would lose their attack capabilities after being transmitted over OSNs. In this work, we aim to design a new framework for generating robust AEs that can survive the OSN transmission; namely, the AEs before and after the OSN transmission both possess strong attack capabilities. To this end, we first propose a differentiable network termed SImulated OSN (SIO) to simulate the various operations conducted by an OSN. Specifically, the SIO network consists of two modules: 1) a differentiable JPEG layer for approximating the ubiquitous JPEG compression and 2) an encoder-decoder subnetwork for mimicking the remaining operations. Based upon the SIO network, we then formulate an optimization framework to generate robust AEs by enforcing model outputs with and without passing through the SIO to be both misled. Extensive experiments conducted over Facebook, WeChat and QQ demonstrate that our attack methods produce more robust AEs than existing approaches, especially under small distortion constraints; the performance gain in terms of Attack Success Rate (ASR) could be more than 60%. Furthermore, we build a public dataset containing more than 10,000 pairs of AEs processed by Facebook, WeChat or QQ, facilitating future research in the robust AEs generation. The dataset and code are available at https://github.com/csjunjun/RobustOSNAttack.git.



