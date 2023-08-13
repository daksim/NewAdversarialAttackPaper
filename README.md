# Latest Adversarial Attack Papers
**update at 2023-08-13 21:30:44**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. RobustPdM: Designing Robust Predictive Maintenance against Adversarial Attacks**

cs.CR

**SubmitDate**: 2023-08-10    [abs](http://arxiv.org/abs/2301.10822v2) [paper-pdf](http://arxiv.org/pdf/2301.10822v2)

**Authors**: Ayesha Siddique, Ripan Kumar Kundu, Gautam Raj Mode, Khaza Anuarul Hoque

**Abstract**: The state-of-the-art predictive maintenance (PdM) techniques have shown great success in reducing maintenance costs and downtime of complicated machines while increasing overall productivity through extensive utilization of Internet-of-Things (IoT) and Deep Learning (DL). Unfortunately, IoT sensors and DL algorithms are both prone to cyber-attacks. For instance, DL algorithms are known for their susceptibility to adversarial examples. Such adversarial attacks are vastly under-explored in the PdM domain. This is because the adversarial attacks in the computer vision domain for classification tasks cannot be directly applied to the PdM domain for multivariate time series (MTS) regression tasks. In this work, we propose an end-to-end methodology to design adversarially robust PdM systems by extensively analyzing the effect of different types of adversarial attacks and proposing a novel adversarial defense technique for DL-enabled PdM models. First, we propose novel MTS Projected Gradient Descent (PGD) and MTS PGD with random restarts (PGD_r) attacks. Then, we evaluate the impact of MTS PGD and PGD_r along with MTS Fast Gradient Sign Method (FGSM) and MTS Basic Iterative Method (BIM) on Long Short-Term Memory (LSTM), Gated Recurrent Unit (GRU), Convolutional Neural Network (CNN), and Bi-directional LSTM based PdM system. Our results using NASA's turbofan engine dataset show that adversarial attacks can cause a severe defect (up to 11X) in the RUL prediction, outperforming the effectiveness of the state-of-the-art PdM attacks by 3X. Furthermore, we present a novel approximate adversarial training method to defend against adversarial attacks. We observe that approximate adversarial training can significantly improve the robustness of PdM models (up to 54X) and outperforms the state-of-the-art PdM defense methods by offering 3X more robustness.



## **2. Diffusion Denoised Smoothing for Certified and Adversarial Robust Out-Of-Distribution Detection**

cs.LG

**SubmitDate**: 2023-08-10    [abs](http://arxiv.org/abs/2303.14961v3) [paper-pdf](http://arxiv.org/pdf/2303.14961v3)

**Authors**: Nicola Franco, Daniel Korth, Jeanette Miriam Lorenz, Karsten Roscher, Stephan Guennemann

**Abstract**: As the use of machine learning continues to expand, the importance of ensuring its safety cannot be overstated. A key concern in this regard is the ability to identify whether a given sample is from the training distribution, or is an "Out-Of-Distribution" (OOD) sample. In addition, adversaries can manipulate OOD samples in ways that lead a classifier to make a confident prediction. In this study, we present a novel approach for certifying the robustness of OOD detection within a $\ell_2$-norm around the input, regardless of network architecture and without the need for specific components or additional training. Further, we improve current techniques for detecting adversarial attacks on OOD samples, while providing high levels of certified and adversarial robustness on in-distribution samples. The average of all OOD detection metrics on CIFAR10/100 shows an increase of $\sim 13 \% / 5\%$ relative to previous approaches.



## **3. Hard No-Box Adversarial Attack on Skeleton-Based Human Action Recognition with Skeleton-Motion-Informed Gradient**

cs.CV

**SubmitDate**: 2023-08-10    [abs](http://arxiv.org/abs/2308.05681v1) [paper-pdf](http://arxiv.org/pdf/2308.05681v1)

**Authors**: Zhengzhi Lu, He Wang, Ziyi Chang, Guoan Yang, Hubert P. H. Shum

**Abstract**: Recently, methods for skeleton-based human activity recognition have been shown to be vulnerable to adversarial attacks. However, these attack methods require either the full knowledge of the victim (i.e. white-box attacks), access to training data (i.e. transfer-based attacks) or frequent model queries (i.e. black-box attacks). All their requirements are highly restrictive, raising the question of how detrimental the vulnerability is. In this paper, we show that the vulnerability indeed exists. To this end, we consider a new attack task: the attacker has no access to the victim model or the training data or labels, where we coin the term hard no-box attack. Specifically, we first learn a motion manifold where we define an adversarial loss to compute a new gradient for the attack, named skeleton-motion-informed (SMI) gradient. Our gradient contains information of the motion dynamics, which is different from existing gradient-based attack methods that compute the loss gradient assuming each dimension in the data is independent. The SMI gradient can augment many gradient-based attack methods, leading to a new family of no-box attack methods. Extensive evaluation and comparison show that our method imposes a real threat to existing classifiers. They also show that the SMI gradient improves the transferability and imperceptibility of adversarial samples in both no-box and transfer-based black-box settings.



## **4. Symmetry Defense Against XGBoost Adversarial Perturbation Attacks**

cs.LG

16 pages

**SubmitDate**: 2023-08-10    [abs](http://arxiv.org/abs/2308.05575v1) [paper-pdf](http://arxiv.org/pdf/2308.05575v1)

**Authors**: Blerta Lindqvist

**Abstract**: We examine whether symmetry can be used to defend tree-based ensemble classifiers such as gradient-boosting decision trees (GBDTs) against adversarial perturbation attacks. The idea is based on a recent symmetry defense for convolutional neural network classifiers (CNNs) that utilizes CNNs' lack of invariance with respect to symmetries. CNNs lack invariance because they can classify a symmetric sample, such as a horizontally flipped image, differently from the original sample. CNNs' lack of invariance also means that CNNs can classify symmetric adversarial samples differently from the incorrect classification of adversarial samples. Using CNNs' lack of invariance, the recent CNN symmetry defense has shown that the classification of symmetric adversarial samples reverts to the correct sample classification. In order to apply the same symmetry defense to GBDTs, we examine GBDT invariance and are the first to show that GBDTs also lack invariance with respect to symmetries. We apply and evaluate the GBDT symmetry defense for nine datasets against six perturbation attacks with a threat model that ranges from zero-knowledge to perfect-knowledge adversaries. Using the feature inversion symmetry against zero-knowledge adversaries, we achieve up to 100% accuracy on adversarial samples even when default and robust classifiers have 0% accuracy. Using the feature inversion and horizontal flip symmetries against perfect-knowledge adversaries, we achieve up to over 95% accuracy on adversarial samples for the GBDT classifier of the F-MNIST dataset even when default and robust classifiers have 0% accuracy.



## **5. Symmetry Defense Against CNN Adversarial Perturbation Attacks**

cs.LG

19 pages

**SubmitDate**: 2023-08-10    [abs](http://arxiv.org/abs/2210.04087v3) [paper-pdf](http://arxiv.org/pdf/2210.04087v3)

**Authors**: Blerta Lindqvist

**Abstract**: This paper uses symmetry to make Convolutional Neural Network classifiers (CNNs) robust against adversarial perturbation attacks. Such attacks add perturbation to original images to generate adversarial images that fool classifiers such as road sign classifiers of autonomous vehicles. Although symmetry is a pervasive aspect of the natural world, CNNs are unable to handle symmetry well. For example, a CNN can classify an image differently from its mirror image. For an adversarial image that misclassifies with a wrong label $l_w$, CNN inability to handle symmetry means that a symmetric adversarial image can classify differently from the wrong label $l_w$. Further than that, we find that the classification of a symmetric adversarial image reverts to the correct label. To classify an image when adversaries are unaware of the defense, we apply symmetry to the image and use the classification label of the symmetric image. To classify an image when adversaries are aware of the defense, we use mirror symmetry and pixel inversion symmetry to form a symmetry group. We apply all the group symmetries to the image and decide on the output label based on the agreement of any two of the classification labels of the symmetry images. Adaptive attacks fail because they need to rely on loss functions that use conflicting CNN output values for symmetric images. Without attack knowledge, the proposed symmetry defense succeeds against both gradient-based and random-search attacks, with up to near-default accuracies for ImageNet. The defense even improves the classification accuracy of original images.



## **6. Complex Network Effects on the Robustness of Graph Convolutional Networks**

cs.SI

39 pages, 8 figures. arXiv admin note: text overlap with  arXiv:2003.05822

**SubmitDate**: 2023-08-10    [abs](http://arxiv.org/abs/2308.05498v1) [paper-pdf](http://arxiv.org/pdf/2308.05498v1)

**Authors**: Benjamin A. Miller, Kevin Chan, Tina Eliassi-Rad

**Abstract**: Vertex classification -- the problem of identifying the class labels of nodes in a graph -- has applicability in a wide variety of domains. Examples include classifying subject areas of papers in citation networks or roles of machines in a computer network. Vertex classification using graph convolutional networks is susceptible to targeted poisoning attacks, in which both graph structure and node attributes can be changed in an attempt to misclassify a target node. This vulnerability decreases users' confidence in the learning method and can prevent adoption in high-stakes contexts. Defenses have also been proposed, focused on filtering edges before creating the model or aggregating information from neighbors more robustly. This paper considers an alternative: we leverage network characteristics in the training data selection process to improve robustness of vertex classifiers.   We propose two alternative methods of selecting training data: (1) to select the highest-degree nodes and (2) to iteratively select the node with the most neighbors minimally connected to the training set. In the datasets on which the original attack was demonstrated, we show that changing the training set can make the network much harder to attack. To maintain a given probability of attack success, the adversary must use far more perturbations; often a factor of 2--4 over the random training baseline. These training set selection methods often work in conjunction with the best recently published defenses to provide even greater robustness. While increasing the amount of randomly selected training data sometimes results in a more robust classifier, the proposed methods increase robustness substantially more. We also run a simulation study in which we demonstrate conditions under which each of the two methods outperforms the other, controlling for the graph topology, homophily of the labels, and node attributes.



## **7. Multi-metrics adaptively identifies backdoors in Federated learning**

cs.CR

14 pages, 8 figures and 7 tables; 2023 IEEE/CVF International  Conference on Computer Vision (ICCV)

**SubmitDate**: 2023-08-10    [abs](http://arxiv.org/abs/2303.06601v2) [paper-pdf](http://arxiv.org/pdf/2303.06601v2)

**Authors**: Siquan Huang, Yijiang Li, Chong Chen, Leyu Shi, Ying Gao

**Abstract**: The decentralized and privacy-preserving nature of federated learning (FL) makes it vulnerable to backdoor attacks aiming to manipulate the behavior of the resulting model on specific adversary-chosen inputs. However, most existing defenses based on statistical differences take effect only against specific attacks, especially when the malicious gradients are similar to benign ones or the data are highly non-independent and identically distributed (non-IID). In this paper, we revisit the distance-based defense methods and discover that i) Euclidean distance becomes meaningless in high dimensions and ii) malicious gradients with diverse characteristics cannot be identified by a single metric. To this end, we present a simple yet effective defense strategy with multi-metrics and dynamic weighting to identify backdoors adaptively. Furthermore, our novel defense has no reliance on predefined assumptions over attack settings or data distributions and little impact on benign performance. To evaluate the effectiveness of our approach, we conduct comprehensive experiments on different datasets under various attack settings, where our method achieves the best defensive performance. For instance, we achieve the lowest backdoor accuracy of 3.06% under the difficult Edge-case PGD, showing significant superiority over previous defenses. The results also demonstrate that our method can be well-adapted to a wide range of non-IID degrees without sacrificing the benign performance.



## **8. Adv-Inpainting: Generating Natural and Transferable Adversarial Patch via Attention-guided Feature Fusion**

cs.CV

**SubmitDate**: 2023-08-10    [abs](http://arxiv.org/abs/2308.05320v1) [paper-pdf](http://arxiv.org/pdf/2308.05320v1)

**Authors**: Yanjie Li, Mingxing Duan, Bin Xiao

**Abstract**: The rudimentary adversarial attacks utilize additive noise to attack facial recognition (FR) models. However, because manipulating the total face is impractical in the physical setting, most real-world FR attacks are based on adversarial patches, which limit perturbations to a small area. Previous adversarial patch attacks often resulted in unnatural patterns and clear boundaries that were easily noticeable. In this paper, we argue that generating adversarial patches with plausible content can result in stronger transferability than using additive noise or directly sampling from the latent space. To generate natural-looking and highly transferable adversarial patches, we propose an innovative two-stage coarse-to-fine attack framework called Adv-Inpainting. In the first stage, we propose an attention-guided StyleGAN (Att-StyleGAN) that adaptively combines texture and identity features based on the attention map to generate high-transferable and natural adversarial patches. In the second stage, we design a refinement network with a new boundary variance loss to further improve the coherence between the patch and its surrounding area. Experiment results demonstrate that Adv-Inpainting is stealthy and can produce adversarial patches with stronger transferability and improved visual quality than previous adversarial patch attacks.



## **9. Analyzing Privacy Leakage in Machine Learning via Multiple Hypothesis Testing: A Lesson From Fano**

cs.LG

**SubmitDate**: 2023-08-10    [abs](http://arxiv.org/abs/2210.13662v2) [paper-pdf](http://arxiv.org/pdf/2210.13662v2)

**Authors**: Chuan Guo, Alexandre Sablayrolles, Maziar Sanjabi

**Abstract**: Differential privacy (DP) is by far the most widely accepted framework for mitigating privacy risks in machine learning. However, exactly how small the privacy parameter $\epsilon$ needs to be to protect against certain privacy risks in practice is still not well-understood. In this work, we study data reconstruction attacks for discrete data and analyze it under the framework of multiple hypothesis testing. We utilize different variants of the celebrated Fano's inequality to derive upper bounds on the inferential power of a data reconstruction adversary when the model is trained differentially privately. Importantly, we show that if the underlying private data takes values from a set of size $M$, then the target privacy parameter $\epsilon$ can be $O(\log M)$ before the adversary gains significant inferential power. Our analysis offers theoretical evidence for the empirical effectiveness of DP against data reconstruction attacks even at relatively large values of $\epsilon$.



## **10. Benchmarking and Analyzing Robust Point Cloud Recognition: Bag of Tricks for Defending Adversarial Examples**

cs.CV

8 pages 6 figures

**SubmitDate**: 2023-08-10    [abs](http://arxiv.org/abs/2307.16361v2) [paper-pdf](http://arxiv.org/pdf/2307.16361v2)

**Authors**: Qiufan Ji, Lin Wang, Cong Shi, Shengshan Hu, Yingying Chen, Lichao Sun

**Abstract**: Deep Neural Networks (DNNs) for 3D point cloud recognition are vulnerable to adversarial examples, threatening their practical deployment. Despite the many research endeavors have been made to tackle this issue in recent years, the diversity of adversarial examples on 3D point clouds makes them more challenging to defend against than those on 2D images. For examples, attackers can generate adversarial examples by adding, shifting, or removing points. Consequently, existing defense strategies are hard to counter unseen point cloud adversarial examples. In this paper, we first establish a comprehensive, and rigorous point cloud adversarial robustness benchmark to evaluate adversarial robustness, which can provide a detailed understanding of the effects of the defense and attack methods. We then collect existing defense tricks in point cloud adversarial defenses and then perform extensive and systematic experiments to identify an effective combination of these tricks. Furthermore, we propose a hybrid training augmentation methods that consider various types of point cloud adversarial examples to adversarial training, significantly improving the adversarial robustness. By combining these tricks, we construct a more robust defense framework achieving an average accuracy of 83.45\% against various attacks, demonstrating its capability to enabling robust learners. Our codebase are open-sourced on: \url{https://github.com/qiufan319/benchmark_pc_attack.git}.



## **11. Byzantine-Robust Decentralized Stochastic Optimization with Stochastic Gradient Noise-Independent Learning Error**

cs.LG

**SubmitDate**: 2023-08-10    [abs](http://arxiv.org/abs/2308.05292v1) [paper-pdf](http://arxiv.org/pdf/2308.05292v1)

**Authors**: Jie Peng, Weiyu Li, Qing Ling

**Abstract**: This paper studies Byzantine-robust stochastic optimization over a decentralized network, where every agent periodically communicates with its neighbors to exchange local models, and then updates its own local model by stochastic gradient descent (SGD). The performance of such a method is affected by an unknown number of Byzantine agents, which conduct adversarially during the optimization process. To the best of our knowledge, there is no existing work that simultaneously achieves a linear convergence speed and a small learning error. We observe that the learning error is largely dependent on the intrinsic stochastic gradient noise. Motivated by this observation, we introduce two variance reduction methods, stochastic average gradient algorithm (SAGA) and loopless stochastic variance-reduced gradient (LSVRG), to Byzantine-robust decentralized stochastic optimization for eliminating the negative effect of the stochastic gradient noise. The two resulting methods, BRAVO-SAGA and BRAVO-LSVRG, enjoy both linear convergence speeds and stochastic gradient noise-independent learning errors. Such learning errors are optimal for a class of methods based on total variation (TV)-norm regularization and stochastic subgradient update. We conduct extensive numerical experiments to demonstrate their effectiveness under various Byzantine attacks.



## **12. Risk-based Security Measure Allocation Against Injection Attacks on Actuators**

eess.SY

Accepted to IEEE Open Journal of Control Systems (OJ-CSYS)

**SubmitDate**: 2023-08-09    [abs](http://arxiv.org/abs/2304.02055v2) [paper-pdf](http://arxiv.org/pdf/2304.02055v2)

**Authors**: Sribalaji C. Anand, André M. H. Teixeira

**Abstract**: This article considers the problem of risk-optimal allocation of security measures when the actuators of an uncertain control system are under attack. We consider an adversary injecting false data into the actuator channels. The attack impact is characterized by the maximum performance loss caused by a stealthy adversary with bounded energy. Since the impact is a random variable, due to system uncertainty, we use Conditional Value-at-Risk (CVaR) to characterize the risk associated with the attack. We then consider the problem of allocating the security measures which minimize the risk. We assume that there are only a limited number of security measures available. Under this constraint, we observe that the allocation problem is a mixed-integer optimization problem. Thus we use relaxation techniques to approximate the security allocation problem into a Semi-Definite Program (SDP). We also compare our allocation method $(i)$ across different risk measures: the worst-case measure, the average (nominal) measure, and $(ii)$ across different search algorithms: the exhaustive and the greedy search algorithms. We depict the efficacy of our approach through numerical examples.



## **13. Do Perceptually Aligned Gradients Imply Adversarial Robustness?**

cs.CV

**SubmitDate**: 2023-08-09    [abs](http://arxiv.org/abs/2207.11378v3) [paper-pdf](http://arxiv.org/pdf/2207.11378v3)

**Authors**: Roy Ganz, Bahjat Kawar, Michael Elad

**Abstract**: Adversarially robust classifiers possess a trait that non-robust models do not -- Perceptually Aligned Gradients (PAG). Their gradients with respect to the input align well with human perception. Several works have identified PAG as a byproduct of robust training, but none have considered it as a standalone phenomenon nor studied its own implications. In this work, we focus on this trait and test whether \emph{Perceptually Aligned Gradients imply Robustness}. To this end, we develop a novel objective to directly promote PAG in training classifiers and examine whether models with such gradients are more robust to adversarial attacks. Extensive experiments on multiple datasets and architectures validate that models with aligned gradients exhibit significant robustness, exposing the surprising bidirectional connection between PAG and robustness. Lastly, we show that better gradient alignment leads to increased robustness and harness this observation to boost the robustness of existing adversarial training techniques.



## **14. Adversarial ModSecurity: Countering Adversarial SQL Injections with Robust Machine Learning**

cs.LG

**SubmitDate**: 2023-08-09    [abs](http://arxiv.org/abs/2308.04964v1) [paper-pdf](http://arxiv.org/pdf/2308.04964v1)

**Authors**: Biagio Montaruli, Luca Demetrio, Andrea Valenza, Battista Biggio, Luca Compagna, Davide Balzarotti, Davide Ariu, Luca Piras

**Abstract**: ModSecurity is widely recognized as the standard open-source Web Application Firewall (WAF), maintained by the OWASP Foundation. It detects malicious requests by matching them against the Core Rule Set, identifying well-known attack patterns. Each rule in the CRS is manually assigned a weight, based on the severity of the corresponding attack, and a request is detected as malicious if the sum of the weights of the firing rules exceeds a given threshold. In this work, we show that this simple strategy is largely ineffective for detecting SQL injection (SQLi) attacks, as it tends to block many legitimate requests, while also being vulnerable to adversarial SQLi attacks, i.e., attacks intentionally manipulated to evade detection. To overcome these issues, we design a robust machine learning model, named AdvModSec, which uses the CRS rules as input features, and it is trained to detect adversarial SQLi attacks. Our experiments show that AdvModSec, being trained on the traffic directed towards the protected web services, achieves a better trade-off between detection and false positive rates, improving the detection rate of the vanilla version of ModSecurity with CRS by 21%. Moreover, our approach is able to improve its adversarial robustness against adversarial SQLi attacks by 42%, thereby taking a step forward towards building more robust and trustworthy WAFs.



## **15. Adversarial Deep Reinforcement Learning for Cyber Security in Software Defined Networks**

cs.CR

**SubmitDate**: 2023-08-09    [abs](http://arxiv.org/abs/2308.04909v1) [paper-pdf](http://arxiv.org/pdf/2308.04909v1)

**Authors**: Luke Borchjes, Clement Nyirenda, Louise Leenen

**Abstract**: This paper focuses on the impact of leveraging autonomous offensive approaches in Deep Reinforcement Learning (DRL) to train more robust agents by exploring the impact of applying adversarial learning to DRL for autonomous security in Software Defined Networks (SDN). Two algorithms, Double Deep Q-Networks (DDQN) and Neural Episodic Control to Deep Q-Network (NEC2DQN or N2D), are compared. NEC2DQN was proposed in 2018 and is a new member of the deep q-network (DQN) family of algorithms. The attacker has full observability of the environment and access to a causative attack that uses state manipulation in an attempt to poison the learning process. The implementation of the attack is done under a white-box setting, in which the attacker has access to the defender's model and experiences. Two games are played; in the first game, DDQN is a defender and N2D is an attacker, and in second game, the roles are reversed. The games are played twice; first, without an active causative attack and secondly, with an active causative attack. For execution, three sets of game results are recorded in which a single set consists of 10 game runs. The before and after results are then compared in order to see if there was actually an improvement or degradation. The results show that with minute parameter changes made to the algorithms, there was growth in the attacker's role, since it is able to win games. Implementation of the adversarial learning by the introduction of the causative attack showed the algorithms are still able to defend the network according to their strengths.



## **16. Rickrolling the Artist: Injecting Backdoors into Text Encoders for Text-to-Image Synthesis**

cs.LG

Published as a conference paper at ICCV 2023

**SubmitDate**: 2023-08-09    [abs](http://arxiv.org/abs/2211.02408v3) [paper-pdf](http://arxiv.org/pdf/2211.02408v3)

**Authors**: Lukas Struppek, Dominik Hintersdorf, Kristian Kersting

**Abstract**: While text-to-image synthesis currently enjoys great popularity among researchers and the general public, the security of these models has been neglected so far. Many text-guided image generation models rely on pre-trained text encoders from external sources, and their users trust that the retrieved models will behave as promised. Unfortunately, this might not be the case. We introduce backdoor attacks against text-guided generative models and demonstrate that their text encoders pose a major tampering risk. Our attacks only slightly alter an encoder so that no suspicious model behavior is apparent for image generations with clean prompts. By then inserting a single character trigger into the prompt, e.g., a non-Latin character or emoji, the adversary can trigger the model to either generate images with pre-defined attributes or images following a hidden, potentially malicious description. We empirically demonstrate the high effectiveness of our attacks on Stable Diffusion and highlight that the injection process of a single backdoor takes less than two minutes. Besides phrasing our approach solely as an attack, it can also force an encoder to forget phrases related to certain concepts, such as nudity or violence, and help to make image generation safer.



## **17. Data-Free Model Extraction Attacks in the Context of Object Detection**

cs.CR

Submitted to The 14th International Conference on Computer Vision  Systems (ICVS 2023), to be published in Springer, Lecture Notes in Computer  Science

**SubmitDate**: 2023-08-09    [abs](http://arxiv.org/abs/2308.05127v1) [paper-pdf](http://arxiv.org/pdf/2308.05127v1)

**Authors**: Harshit Shah, Aravindhan G, Pavan Kulkarni, Yuvaraj Govidarajulu, Manojkumar Parmar

**Abstract**: A significant number of machine learning models are vulnerable to model extraction attacks, which focus on stealing the models by using specially curated queries against the target model. This task is well accomplished by using part of the training data or a surrogate dataset to train a new model that mimics a target model in a white-box environment. In pragmatic situations, however, the target models are trained on private datasets that are inaccessible to the adversary. The data-free model extraction technique replaces this problem when it comes to using queries artificially curated by a generator similar to that used in Generative Adversarial Nets. We propose for the first time, to the best of our knowledge, an adversary black box attack extending to a regression problem for predicting bounding box coordinates in object detection. As part of our study, we found that defining a loss function and using a novel generator setup is one of the key aspects in extracting the target model. We find that the proposed model extraction method achieves significant results by using reasonable queries. The discovery of this object detection vulnerability will support future prospects for securing such models.



## **18. GIFD: A Generative Gradient Inversion Method with Feature Domain Optimization**

cs.CV

ICCV 2023

**SubmitDate**: 2023-08-09    [abs](http://arxiv.org/abs/2308.04699v1) [paper-pdf](http://arxiv.org/pdf/2308.04699v1)

**Authors**: Hao Fang, Bin Chen, Xuan Wang, Zhi Wang, Shu-Tao Xia

**Abstract**: Federated Learning (FL) has recently emerged as a promising distributed machine learning framework to preserve clients' privacy, by allowing multiple clients to upload the gradients calculated from their local data to a central server. Recent studies find that the exchanged gradients also take the risk of privacy leakage, e.g., an attacker can invert the shared gradients and recover sensitive data against an FL system by leveraging pre-trained generative adversarial networks (GAN) as prior knowledge. However, performing gradient inversion attacks in the latent space of the GAN model limits their expression ability and generalizability. To tackle these challenges, we propose \textbf{G}radient \textbf{I}nversion over \textbf{F}eature \textbf{D}omains (GIFD), which disassembles the GAN model and searches the feature domains of the intermediate layers. Instead of optimizing only over the initial latent code, we progressively change the optimized layer, from the initial latent space to intermediate layers closer to the output images. In addition, we design a regularizer to avoid unreal image generation by adding a small ${l_1}$ ball constraint to the searching range. We also extend GIFD to the out-of-distribution (OOD) setting, which weakens the assumption that the training sets of GANs and FL tasks obey the same data distribution. Extensive experiments demonstrate that our method can achieve pixel-level reconstruction and is superior to the existing methods. Notably, GIFD also shows great generalizability under different defense strategy settings and batch sizes.



## **19. SSL-Auth: An Authentication Framework by Fragile Watermarking for Pre-trained Encoders in Self-supervised Learning**

cs.CR

**SubmitDate**: 2023-08-09    [abs](http://arxiv.org/abs/2308.04673v1) [paper-pdf](http://arxiv.org/pdf/2308.04673v1)

**Authors**: Xiaobei Li, Changchun Yin, Liming Fang, Run Wang, Chenhao Lin

**Abstract**: Self-supervised learning (SSL) which leverages unlabeled datasets for pre-training powerful encoders has achieved significant success in recent years. These encoders are commonly used as feature extractors for various downstream tasks, requiring substantial data and computing resources for their training process. With the deployment of pre-trained encoders in commercial use, protecting the intellectual property of model owners and ensuring the trustworthiness of the models becomes crucial. Recent research has shown that encoders are threatened by backdoor attacks, adversarial attacks, etc. Therefore, a scheme to verify the integrity of pre-trained encoders is needed to protect users. In this paper, we propose SSL-Auth, the first fragile watermarking scheme for verifying the integrity of encoders without compromising model performance. Our method utilizes selected key samples as watermark information and trains a verification network to reconstruct the watermark information, thereby verifying the integrity of the encoder. By comparing the reconstruction results of the key samples, malicious modifications can be effectively detected, as altered models should not exhibit similar reconstruction performance as the original models. Extensive evaluations on various models and diverse datasets demonstrate the effectiveness and fragility of our proposed SSL-Auth.



## **20. Pelta: Shielding Transformers to Mitigate Evasion Attacks in Federated Learning**

cs.LG

**SubmitDate**: 2023-08-08    [abs](http://arxiv.org/abs/2308.04373v1) [paper-pdf](http://arxiv.org/pdf/2308.04373v1)

**Authors**: Simon Queyrut, Yérom-David Bromberg, Valerio Schiavoni

**Abstract**: The main premise of federated learning is that machine learning model updates are computed locally, in particular to preserve user data privacy, as those never leave the perimeter of their device. This mechanism supposes the general model, once aggregated, to be broadcast to collaborating and non malicious nodes. However, without proper defenses, compromised clients can easily probe the model inside their local memory in search of adversarial examples. For instance, considering image-based applications, adversarial examples consist of imperceptibly perturbed images (to the human eye) misclassified by the local model, which can be later presented to a victim node's counterpart model to replicate the attack. To mitigate such malicious probing, we introduce Pelta, a novel shielding mechanism leveraging trusted hardware. By harnessing the capabilities of Trusted Execution Environments (TEEs), Pelta masks part of the back-propagation chain rule, otherwise typically exploited by attackers for the design of malicious samples. We evaluate Pelta on a state of the art ensemble model and demonstrate its effectiveness against the Self Attention Gradient adversarial Attack.



## **21. Accurate, Explainable, and Private Models: Providing Recourse While Minimizing Training Data Leakage**

cs.LG

Proceedings of The Second Workshop on New Frontiers in Adversarial  Machine Learning (AdvML-Frontiers @ ICML 2023)

**SubmitDate**: 2023-08-08    [abs](http://arxiv.org/abs/2308.04341v1) [paper-pdf](http://arxiv.org/pdf/2308.04341v1)

**Authors**: Catherine Huang, Chelse Swoopes, Christina Xiao, Jiaqi Ma, Himabindu Lakkaraju

**Abstract**: Machine learning models are increasingly utilized across impactful domains to predict individual outcomes. As such, many models provide algorithmic recourse to individuals who receive negative outcomes. However, recourse can be leveraged by adversaries to disclose private information. This work presents the first attempt at mitigating such attacks. We present two novel methods to generate differentially private recourse: Differentially Private Model (DPM) and Laplace Recourse (LR). Using logistic regression classifiers and real world and synthetic datasets, we find that DPM and LR perform well in reducing what an adversary can infer, especially at low FPR. When training dataset size is large enough, we find particular success in preventing privacy leakage while maintaining model and recourse accuracy with our novel LR method.



## **22. Why Does Little Robustness Help? Understanding Adversarial Transferability From Surrogate Training**

cs.LG

Accepted by IEEE Symposium on Security and Privacy (Oakland) 2024; 21  pages, 11 figures, 13 tables

**SubmitDate**: 2023-08-08    [abs](http://arxiv.org/abs/2307.07873v3) [paper-pdf](http://arxiv.org/pdf/2307.07873v3)

**Authors**: Yechao Zhang, Shengshan Hu, Leo Yu Zhang, Junyu Shi, Minghui Li, Xiaogeng Liu, Wei Wan, Hai Jin

**Abstract**: Adversarial examples (AEs) for DNNs have been shown to be transferable: AEs that successfully fool white-box surrogate models can also deceive other black-box models with different architectures. Although a bunch of empirical studies have provided guidance on generating highly transferable AEs, many of these findings lack explanations and even lead to inconsistent advice. In this paper, we take a further step towards understanding adversarial transferability, with a particular focus on surrogate aspects. Starting from the intriguing little robustness phenomenon, where models adversarially trained with mildly perturbed adversarial samples can serve as better surrogates, we attribute it to a trade-off between two predominant factors: model smoothness and gradient similarity. Our investigations focus on their joint effects, rather than their separate correlations with transferability. Through a series of theoretical and empirical analyses, we conjecture that the data distribution shift in adversarial training explains the degradation of gradient similarity. Building on these insights, we explore the impacts of data augmentation and gradient regularization on transferability and identify that the trade-off generally exists in the various training mechanisms, thus building a comprehensive blueprint for the regulation mechanism behind transferability. Finally, we provide a general route for constructing better surrogates to boost transferability which optimizes both model smoothness and gradient similarity simultaneously, e.g., the combination of input gradient regularization and sharpness-aware minimization (SAM), validated by extensive experiments. In summary, we call for attention to the united impacts of these two factors for launching effective transfer attacks, rather than optimizing one while ignoring the other, and emphasize the crucial role of manipulating surrogate models.



## **23. FLIRT: Feedback Loop In-context Red Teaming**

cs.AI

**SubmitDate**: 2023-08-08    [abs](http://arxiv.org/abs/2308.04265v1) [paper-pdf](http://arxiv.org/pdf/2308.04265v1)

**Authors**: Ninareh Mehrabi, Palash Goyal, Christophe Dupuy, Qian Hu, Shalini Ghosh, Richard Zemel, Kai-Wei Chang, Aram Galstyan, Rahul Gupta

**Abstract**: Warning: this paper contains content that may be inappropriate or offensive.   As generative models become available for public use in various applications, testing and analyzing vulnerabilities of these models has become a priority. Here we propose an automatic red teaming framework that evaluates a given model and exposes its vulnerabilities against unsafe and inappropriate content generation. Our framework uses in-context learning in a feedback loop to red team models and trigger them into unsafe content generation. We propose different in-context attack strategies to automatically learn effective and diverse adversarial prompts for text-to-image models. Our experiments demonstrate that compared to baseline approaches, our proposed strategy is significantly more effective in exposing vulnerabilities in Stable Diffusion (SD) model, even when the latter is enhanced with safety features. Furthermore, we demonstrate that the proposed framework is effective for red teaming text-to-text models, resulting in significantly higher toxic response generation rate compared to previously reported numbers.



## **24. A semantic backdoor attack against Graph Convolutional Networks**

cs.LG

**SubmitDate**: 2023-08-08    [abs](http://arxiv.org/abs/2302.14353v3) [paper-pdf](http://arxiv.org/pdf/2302.14353v3)

**Authors**: Jiazhu Dai, Zhipeng Xiong

**Abstract**: Graph convolutional networks (GCNs) have been very effective in addressing the issue of various graph-structured related tasks, such as node classification and graph classification. However, recent research has shown that GCNs are vulnerable to a new type of threat called a backdoor attack, where the adversary can inject a hidden backdoor into GCNs so that the attacked model performs well on benign samples, but its prediction will be maliciously changed to the attacker-specified target label if the hidden backdoor is activated by the attacker-defined trigger. In this paper, we investigate whether such semantic backdoor attacks are possible for GCNs and propose a semantic backdoor attack against GCNs (SBAG) under the context of graph classification to reveal the existence of this security vulnerability in GCNs. SBAG uses a certain type of node in the samples as a backdoor trigger and injects a hidden backdoor into GCN models by poisoning training data. The backdoor will be activated, and the GCN models will give malicious classification results specified by the attacker even on unmodified samples as long as the samples contain enough trigger nodes. We evaluate SBAG on four graph datasets. The experimental results indicate that SBAG can achieve attack success rates of approximately 99.9% and over 82% for two kinds of attack samples, respectively, with poisoning rates of less than 5%.



## **25. Security of a Continuous-Variable based Quantum Position Verification Protocol**

quant-ph

17 pages, 2 figures

**SubmitDate**: 2023-08-08    [abs](http://arxiv.org/abs/2308.04166v1) [paper-pdf](http://arxiv.org/pdf/2308.04166v1)

**Authors**: Rene Allerstorfer, Llorenç Escolà-Farràs, Arpan Akash Ray, Boris Škorić, Florian Speelman, Philip Verduyn Lunel

**Abstract**: In this work we study quantum position verification with continuous-variable quantum states. In contrast to existing discrete protocols, we present and analyze a protocol that utilizes coherent states and its properties. Compared to discrete-variable photonic states, coherent states offer practical advantages since they can be efficiently prepared and manipulated with current technology. We prove security of the protocol against any unentangled attackers via entropic uncertainty relations, showing that the adversary has more uncertainty than the honest prover about the correct response as long as the noise in the quantum channel is below a certain threshold. Additionally, we show that attackers who pre-share one continuous-variable EPR pair can break the protocol.



## **26. UMD: Unsupervised Model Detection for X2X Backdoor Attacks**

cs.LG

Proceedings of the 40th International Conference on Machine Learning

**SubmitDate**: 2023-08-08    [abs](http://arxiv.org/abs/2305.18651v3) [paper-pdf](http://arxiv.org/pdf/2305.18651v3)

**Authors**: Zhen Xiang, Zidi Xiong, Bo Li

**Abstract**: Backdoor (Trojan) attack is a common threat to deep neural networks, where samples from one or more source classes embedded with a backdoor trigger will be misclassified to adversarial target classes. Existing methods for detecting whether a classifier is backdoor attacked are mostly designed for attacks with a single adversarial target (e.g., all-to-one attack). To the best of our knowledge, without supervision, no existing methods can effectively address the more general X2X attack with an arbitrary number of source classes, each paired with an arbitrary target class. In this paper, we propose UMD, the first Unsupervised Model Detection method that effectively detects X2X backdoor attacks via a joint inference of the adversarial (source, target) class pairs. In particular, we first define a novel transferability statistic to measure and select a subset of putative backdoor class pairs based on a proposed clustering approach. Then, these selected class pairs are jointly assessed based on an aggregation of their reverse-engineered trigger size for detection inference, using a robust and unsupervised anomaly detector we proposed. We conduct comprehensive evaluations on CIFAR-10, GTSRB, and Imagenette dataset, and show that our unsupervised UMD outperforms SOTA detectors (even with supervision) by 17%, 4%, and 8%, respectively, in terms of the detection accuracy against diverse X2X attacks. We also show the strong detection performance of UMD against several strong adaptive attacks.



## **27. Federated Zeroth-Order Optimization using Trajectory-Informed Surrogate Gradients**

cs.LG

**SubmitDate**: 2023-08-08    [abs](http://arxiv.org/abs/2308.04077v1) [paper-pdf](http://arxiv.org/pdf/2308.04077v1)

**Authors**: Yao Shu, Xiaoqiang Lin, Zhongxiang Dai, Bryan Kian Hsiang Low

**Abstract**: Federated optimization, an emerging paradigm which finds wide real-world applications such as federated learning, enables multiple clients (e.g., edge devices) to collaboratively optimize a global function. The clients do not share their local datasets and typically only share their local gradients. However, the gradient information is not available in many applications of federated optimization, which hence gives rise to the paradigm of federated zeroth-order optimization (ZOO). Existing federated ZOO algorithms suffer from the limitations of query and communication inefficiency, which can be attributed to (a) their reliance on a substantial number of function queries for gradient estimation and (b) the significant disparity between their realized local updates and the intended global updates. To this end, we (a) introduce trajectory-informed gradient surrogates which is able to use the history of function queries during optimization for accurate and query-efficient gradient estimation, and (b) develop the technique of adaptive gradient correction using these gradient surrogates to mitigate the aforementioned disparity. Based on these, we propose the federated zeroth-order optimization using trajectory-informed surrogate gradients (FZooS) algorithm for query- and communication-efficient federated ZOO. Our FZooS achieves theoretical improvements over the existing approaches, which is supported by our real-world experiments such as federated black-box adversarial attack and federated non-differentiable metric optimization.



## **28. Adversarial Coreset Selection for Efficient Robust Training**

cs.LG

Accepted to the International Journal of Computer Vision (IJCV).  Extended version of the ECCV2022 paper: arXiv:2112.00378. arXiv admin note:  substantial text overlap with arXiv:2112.00378

**SubmitDate**: 2023-08-08    [abs](http://arxiv.org/abs/2209.05785v2) [paper-pdf](http://arxiv.org/pdf/2209.05785v2)

**Authors**: Hadi M. Dolatabadi, Sarah Erfani, Christopher Leckie

**Abstract**: Neural networks are vulnerable to adversarial attacks: adding well-crafted, imperceptible perturbations to their input can modify their output. Adversarial training is one of the most effective approaches to training robust models against such attacks. Unfortunately, this method is much slower than vanilla training of neural networks since it needs to construct adversarial examples for the entire training data at every iteration. By leveraging the theory of coreset selection, we show how selecting a small subset of training data provides a principled approach to reducing the time complexity of robust training. To this end, we first provide convergence guarantees for adversarial coreset selection. In particular, we show that the convergence bound is directly related to how well our coresets can approximate the gradient computed over the entire training data. Motivated by our theoretical analysis, we propose using this gradient approximation error as our adversarial coreset selection objective to reduce the training set size effectively. Once built, we run adversarial training over this subset of the training data. Unlike existing methods, our approach can be adapted to a wide variety of training objectives, including TRADES, $\ell_p$-PGD, and Perceptual Adversarial Training. We conduct extensive experiments to demonstrate that our approach speeds up adversarial training by 2-3 times while experiencing a slight degradation in the clean and robust accuracy.



## **29. Improving Performance of Semi-Supervised Learning by Adversarial Attacks**

cs.LG

4 pages

**SubmitDate**: 2023-08-08    [abs](http://arxiv.org/abs/2308.04018v1) [paper-pdf](http://arxiv.org/pdf/2308.04018v1)

**Authors**: Dongyoon Yang, Kunwoong Kim, Yongdai Kim

**Abstract**: Semi-supervised learning (SSL) algorithm is a setup built upon a realistic assumption that access to a large amount of labeled data is tough. In this study, we present a generalized framework, named SCAR, standing for Selecting Clean samples with Adversarial Robustness, for improving the performance of recent SSL algorithms. By adversarially attacking pre-trained models with semi-supervision, our framework shows substantial advances in classifying images. We introduce how adversarial attacks successfully select high-confident unlabeled data to be labeled with current predictions. On CIFAR10, three recent SSL algorithms with SCAR result in significantly improved image classification.



## **30. PAIF: Perception-Aware Infrared-Visible Image Fusion for Attack-Tolerant Semantic Segmentation**

cs.CV

Accepted by ACM MM'2023;The source codes are available at  https://github.com/LiuZhu-CV/PAIF

**SubmitDate**: 2023-08-08    [abs](http://arxiv.org/abs/2308.03979v1) [paper-pdf](http://arxiv.org/pdf/2308.03979v1)

**Authors**: Zhu Liu, Jinyuan Liu, Benzhuang Zhang, Long Ma, Xin Fan, Risheng Liu

**Abstract**: Infrared and visible image fusion is a powerful technique that combines complementary information from different modalities for downstream semantic perception tasks. Existing learning-based methods show remarkable performance, but are suffering from the inherent vulnerability of adversarial attacks, causing a significant decrease in accuracy. In this work, a perception-aware fusion framework is proposed to promote segmentation robustness in adversarial scenes. We first conduct systematic analyses about the components of image fusion, investigating the correlation with segmentation robustness under adversarial perturbations. Based on these analyses, we propose a harmonized architecture search with a decomposition-based structure to balance standard accuracy and robustness. We also propose an adaptive learning strategy to improve the parameter robustness of image fusion, which can learn effective feature extraction under diverse adversarial perturbations. Thus, the goals of image fusion (\textit{i.e.,} extracting complementary features from source modalities and defending attack) can be realized from the perspectives of architectural and learning strategies. Extensive experimental results demonstrate that our scheme substantially enhances the robustness, with gains of 15.3% mIOU of segmentation in the adversarial scene, compared with advanced competitors. The source codes are available at https://github.com/LiuZhu-CV/PAIF.



## **31. Fixed Inter-Neuron Covariability Induces Adversarial Robustness**

cs.LG

**SubmitDate**: 2023-08-07    [abs](http://arxiv.org/abs/2308.03956v1) [paper-pdf](http://arxiv.org/pdf/2308.03956v1)

**Authors**: Muhammad Ahmed Shah, Bhiksha Raj

**Abstract**: The vulnerability to adversarial perturbations is a major flaw of Deep Neural Networks (DNNs) that raises question about their reliability when in real-world scenarios. On the other hand, human perception, which DNNs are supposed to emulate, is highly robust to such perturbations, indicating that there may be certain features of the human perception that make it robust but are not represented in the current class of DNNs. One such feature is that the activity of biological neurons is correlated and the structure of this correlation tends to be rather rigid over long spans of times, even if it hampers performance and learning. We hypothesize that integrating such constraints on the activations of a DNN would improve its adversarial robustness, and, to test this hypothesis, we have developed the Self-Consistent Activation (SCA) layer, which comprises of neurons whose activations are consistent with each other, as they conform to a fixed, but learned, covariability pattern. When evaluated on image and sound recognition tasks, the models with a SCA layer achieved high accuracy, and exhibited significantly greater robustness than multi-layer perceptron models to state-of-the-art Auto-PGD adversarial attacks \textit{without being trained on adversarially perturbed data



## **32. "Do Anything Now": Characterizing and Evaluating In-The-Wild Jailbreak Prompts on Large Language Models**

cs.CR

**SubmitDate**: 2023-08-07    [abs](http://arxiv.org/abs/2308.03825v1) [paper-pdf](http://arxiv.org/pdf/2308.03825v1)

**Authors**: Xinyue Shen, Zeyuan Chen, Michael Backes, Yun Shen, Yang Zhang

**Abstract**: The misuse of large language models (LLMs) has garnered significant attention from the general public and LLM vendors. In response, efforts have been made to align LLMs with human values and intent use. However, a particular type of adversarial prompts, known as jailbreak prompt, has emerged and continuously evolved to bypass the safeguards and elicit harmful content from LLMs. In this paper, we conduct the first measurement study on jailbreak prompts in the wild, with 6,387 prompts collected from four platforms over six months. Leveraging natural language processing technologies and graph-based community detection methods, we discover unique characteristics of jailbreak prompts and their major attack strategies, such as prompt injection and privilege escalation. We also observe that jailbreak prompts increasingly shift from public platforms to private ones, posing new challenges for LLM vendors in proactive detection. To assess the potential harm caused by jailbreak prompts, we create a question set comprising 46,800 samples across 13 forbidden scenarios. Our experiments show that current LLMs and safeguards cannot adequately defend jailbreak prompts in all scenarios. Particularly, we identify two highly effective jailbreak prompts which achieve 0.99 attack success rates on ChatGPT (GPT-3.5) and GPT-4, and they have persisted online for over 100 days. Our work sheds light on the severe and evolving threat landscape of jailbreak prompts. We hope our study can facilitate the research community and LLM vendors in promoting safer and regulated LLMs.



## **33. Assessing Adversarial Replay and Deep Learning-Driven Attacks on Specific Emitter Identification-based Security Approaches**

eess.SP

**SubmitDate**: 2023-08-07    [abs](http://arxiv.org/abs/2308.03579v1) [paper-pdf](http://arxiv.org/pdf/2308.03579v1)

**Authors**: Joshua H. Tyler, Mohamed K. M. Fadul, Matthew R. Hilling, Donald R. Reising, T. Daniel Loveless

**Abstract**: Specific Emitter Identification (SEI) detects, characterizes, and identifies emitters by exploiting distinct, inherent, and unintentional features in their transmitted signals. Since its introduction, a significant amount of work has been conducted; however, most assume the emitters are passive and that their identifying signal features are immutable and challenging to mimic. Suggesting the emitters are reluctant and incapable of developing and implementing effective SEI countermeasures; however, Deep Learning (DL) has been shown capable of learning emitter-specific features directly from their raw in-phase and quadrature signal samples, and Software-Defined Radios (SDRs) can manipulate them. Based on these capabilities, it is fair to question the ease at which an emitter can effectively mimic the SEI features of another or manipulate its own to hinder or defeat SEI. This work considers SEI mimicry using three signal features mimicking countermeasures; off-the-self DL; two SDRs of different sizes, weights, power, and cost (SWaP-C); handcrafted and DL-based SEI processes, and a coffee shop deployment. Our results show off-the-shelf DL algorithms, and SDR enables SEI mimicry; however, adversary success is hindered by: the use of decoy emitter preambles, the use of a denoising autoencoder, and SDR SWaP-C constraints.



## **34. Mondrian: Prompt Abstraction Attack Against Large Language Models for Cheaper API Pricing**

cs.CR

**SubmitDate**: 2023-08-07    [abs](http://arxiv.org/abs/2308.03558v1) [paper-pdf](http://arxiv.org/pdf/2308.03558v1)

**Authors**: Wai Man Si, Michael Backes, Yang Zhang

**Abstract**: The Machine Learning as a Service (MLaaS) market is rapidly expanding and becoming more mature. For example, OpenAI's ChatGPT is an advanced large language model (LLM) that generates responses for various queries with associated fees. Although these models can deliver satisfactory performance, they are far from perfect. Researchers have long studied the vulnerabilities and limitations of LLMs, such as adversarial attacks and model toxicity. Inevitably, commercial ML models are also not exempt from such issues, which can be problematic as MLaaS continues to grow. In this paper, we discover a new attack strategy against LLM APIs, namely the prompt abstraction attack. Specifically, we propose Mondrian, a simple and straightforward method that abstracts sentences, which can lower the cost of using LLM APIs. In this approach, the adversary first creates a pseudo API (with a lower established price) to serve as the proxy of the target API (with a higher established price). Next, the pseudo API leverages Mondrian to modify the user query, obtain the abstracted response from the target API, and forward it back to the end user. Our results show that Mondrian successfully reduces user queries' token length ranging from 13% to 23% across various tasks, including text classification, generation, and question answering. Meanwhile, these abstracted queries do not significantly affect the utility of task-specific and general language models like ChatGPT. Mondrian also reduces instruction prompts' token length by at least 11% without compromising output quality. As a result, the prompt abstraction attack enables the adversary to profit without bearing the cost of API development and deployment.



## **35. Exploring the Physical World Adversarial Robustness of Vehicle Detection**

cs.CV

**SubmitDate**: 2023-08-07    [abs](http://arxiv.org/abs/2308.03476v1) [paper-pdf](http://arxiv.org/pdf/2308.03476v1)

**Authors**: Wei Jiang, Tianyuan Zhang, Shuangcheng Liu, Weiyu Ji, Zichao Zhang, Gang Xiao

**Abstract**: Adversarial attacks can compromise the robustness of real-world detection models. However, evaluating these models under real-world conditions poses challenges due to resource-intensive experiments. Virtual simulations offer an alternative, but the absence of standardized benchmarks hampers progress. Addressing this, we propose an innovative instant-level data generation pipeline using the CARLA simulator. Through this pipeline, we establish the Discrete and Continuous Instant-level (DCI) dataset, enabling comprehensive experiments involving three detection models and three physical adversarial attacks. Our findings highlight diverse model performances under adversarial conditions. Yolo v6 demonstrates remarkable resilience, experiencing just a marginal 6.59% average drop in average precision (AP). In contrast, the ASA attack yields a substantial 14.51% average AP reduction, twice the effect of other algorithms. We also note that static scenes yield higher recognition AP values, and outcomes remain relatively consistent across varying weather conditions. Intriguingly, our study suggests that advancements in adversarial attack algorithms may be approaching its ``limitation''.In summary, our work underscores the significance of adversarial attacks in real-world contexts and introduces the DCI dataset as a versatile benchmark. Our findings provide valuable insights for enhancing the robustness of detection models and offer guidance for future research endeavors in the realm of adversarial attacks.



## **36. Imperceptible Physical Attack against Face Recognition Systems via LED Illumination Modulation**

cs.CV

**SubmitDate**: 2023-08-07    [abs](http://arxiv.org/abs/2307.13294v2) [paper-pdf](http://arxiv.org/pdf/2307.13294v2)

**Authors**: Junbin Fang, Canjian Jiang, You Jiang, Puxi Lin, Zhaojie Chen, Yujing Sun, Siu-Ming Yiu, Zoe L. Jiang

**Abstract**: Although face recognition starts to play an important role in our daily life, we need to pay attention that data-driven face recognition vision systems are vulnerable to adversarial attacks. However, the current two categories of adversarial attacks, namely digital attacks and physical attacks both have drawbacks, with the former ones impractical and the latter one conspicuous, high-computational and inexecutable. To address the issues, we propose a practical, executable, inconspicuous and low computational adversarial attack based on LED illumination modulation. To fool the systems, the proposed attack generates imperceptible luminance changes to human eyes through fast intensity modulation of scene LED illumination and uses the rolling shutter effect of CMOS image sensors in face recognition systems to implant luminance information perturbation to the captured face images. In summary,we present a denial-of-service (DoS) attack for face detection and a dodging attack for face verification. We also evaluate their effectiveness against well-known face detection models, Dlib, MTCNN and RetinaFace , and face verification models, Dlib, FaceNet,and ArcFace.The extensive experiments show that the success rates of DoS attacks against face detection models reach 97.67%, 100%, and 100%, respectively, and the success rates of dodging attacks against all face verification models reach 100%.



## **37. A reading survey on adversarial machine learning: Adversarial attacks and their understanding**

cs.LG

**SubmitDate**: 2023-08-07    [abs](http://arxiv.org/abs/2308.03363v1) [paper-pdf](http://arxiv.org/pdf/2308.03363v1)

**Authors**: Shashank Kotyan

**Abstract**: Deep Learning has empowered us to train neural networks for complex data with high performance. However, with the growing research, several vulnerabilities in neural networks have been exposed. A particular branch of research, Adversarial Machine Learning, exploits and understands some of the vulnerabilities that cause the neural networks to misclassify for near original input. A class of algorithms called adversarial attacks is proposed to make the neural networks misclassify for various tasks in different domains. With the extensive and growing research in adversarial attacks, it is crucial to understand the classification of adversarial attacks. This will help us understand the vulnerabilities in a systematic order and help us to mitigate the effects of adversarial attacks. This article provides a survey of existing adversarial attacks and their understanding based on different perspectives. We also provide a brief overview of existing adversarial defences and their limitations in mitigating the effect of adversarial attacks. Further, we conclude with a discussion on the future research directions in the field of adversarial machine learning.



## **38. Membership Inference Attacks against Language Models via Neighbourhood Comparison**

cs.CL

**SubmitDate**: 2023-08-07    [abs](http://arxiv.org/abs/2305.18462v2) [paper-pdf](http://arxiv.org/pdf/2305.18462v2)

**Authors**: Justus Mattern, Fatemehsadat Mireshghallah, Zhijing Jin, Bernhard Schölkopf, Mrinmaya Sachan, Taylor Berg-Kirkpatrick

**Abstract**: Membership Inference attacks (MIAs) aim to predict whether a data sample was present in the training data of a machine learning model or not, and are widely used for assessing the privacy risks of language models. Most existing attacks rely on the observation that models tend to assign higher probabilities to their training samples than non-training points. However, simple thresholding of the model score in isolation tends to lead to high false-positive rates as it does not account for the intrinsic complexity of a sample. Recent work has demonstrated that reference-based attacks which compare model scores to those obtained from a reference model trained on similar data can substantially improve the performance of MIAs. However, in order to train reference models, attacks of this kind make the strong and arguably unrealistic assumption that an adversary has access to samples closely resembling the original training data. Therefore, we investigate their performance in more realistic scenarios and find that they are highly fragile in relation to the data distribution used to train reference models. To investigate whether this fragility provides a layer of safety, we propose and evaluate neighbourhood attacks, which compare model scores for a given sample to scores of synthetically generated neighbour texts and therefore eliminate the need for access to the training data distribution. We show that, in addition to being competitive with reference-based attacks that have perfect knowledge about the training data distribution, our attack clearly outperforms existing reference-free attacks as well as reference-based attacks with imperfect knowledge, which demonstrates the need for a reevaluation of the threat model of adversarial attacks.



## **39. Token-Modification Adversarial Attacks for Natural Language Processing: A Survey**

cs.CL

Version 2: updated

**SubmitDate**: 2023-08-07    [abs](http://arxiv.org/abs/2103.00676v2) [paper-pdf](http://arxiv.org/pdf/2103.00676v2)

**Authors**: Tom Roth, Yansong Gao, Alsharif Abuadbba, Surya Nepal, Wei Liu

**Abstract**: There are now many adversarial attacks for natural language processing systems. Of these, a vast majority achieve success by modifying individual document tokens, which we call here a token-modification attack. Each token-modification attack is defined by a specific combination of fundamental components, such as a constraint on the adversary or a particular search algorithm. Motivated by this observation, we survey existing token-modification attacks and extract the components of each. We use an attack-independent framework to structure our survey which results in an effective categorisation of the field and an easy comparison of components. This survey aims to guide new researchers to this field and spark further research into individual attack components.



## **40. APBench: A Unified Benchmark for Availability Poisoning Attacks and Defenses**

cs.CV

**SubmitDate**: 2023-08-07    [abs](http://arxiv.org/abs/2308.03258v1) [paper-pdf](http://arxiv.org/pdf/2308.03258v1)

**Authors**: Tianrui Qin, Xitong Gao, Juanjuan Zhao, Kejiang Ye, Cheng-Zhong Xu

**Abstract**: The efficacy of availability poisoning, a method of poisoning data by injecting imperceptible perturbations to prevent its use in model training, has been a hot subject of investigation. Previous research suggested that it was difficult to effectively counteract such poisoning attacks. However, the introduction of various defense methods has challenged this notion. Due to the rapid progress in this field, the performance of different novel methods cannot be accurately validated due to variations in experimental setups. To further evaluate the attack and defense capabilities of these poisoning methods, we have developed a benchmark -- APBench for assessing the efficacy of adversarial poisoning. APBench consists of 9 state-of-the-art availability poisoning attacks, 8 defense algorithms, and 4 conventional data augmentation techniques. We also have set up experiments with varying different poisoning ratios, and evaluated the attacks on multiple datasets and their transferability across model architectures. We further conducted a comprehensive evaluation of 2 additional attacks specifically targeting unsupervised models. Our results reveal the glaring inadequacy of existing attacks in safeguarding individual privacy. APBench is open source and available to the deep learning community: https://github.com/lafeat/apbench.



## **41. Unsupervised Adversarial Detection without Extra Model: Training Loss Should Change**

cs.LG

AdvML in ICML 2023  code:https://github.com/CycleBooster/Unsupervised-adversarial-detection-without-extra-model

**SubmitDate**: 2023-08-07    [abs](http://arxiv.org/abs/2308.03243v1) [paper-pdf](http://arxiv.org/pdf/2308.03243v1)

**Authors**: Chien Cheng Chyou, Hung-Ting Su, Winston H. Hsu

**Abstract**: Adversarial robustness poses a critical challenge in the deployment of deep learning models for real-world applications. Traditional approaches to adversarial training and supervised detection rely on prior knowledge of attack types and access to labeled training data, which is often impractical. Existing unsupervised adversarial detection methods identify whether the target model works properly, but they suffer from bad accuracies owing to the use of common cross-entropy training loss, which relies on unnecessary features and strengthens adversarial attacks. We propose new training losses to reduce useless features and the corresponding detection method without prior knowledge of adversarial attacks. The detection rate (true positive rate) against all given white-box attacks is above 93.9% except for attacks without limits (DF($\infty$)), while the false positive rate is barely 2.5%. The proposed method works well in all tested attack types and the false positive rates are even better than the methods good at certain types.



## **42. CGBA: Curvature-aware Geometric Black-box Attack**

cs.CV

This paper is accepted to publish in ICCV

**SubmitDate**: 2023-08-06    [abs](http://arxiv.org/abs/2308.03163v1) [paper-pdf](http://arxiv.org/pdf/2308.03163v1)

**Authors**: Md Farhamdur Reza, Ali Rahmati, Tianfu Wu, Huaiyu Dai

**Abstract**: Decision-based black-box attacks often necessitate a large number of queries to craft an adversarial example. Moreover, decision-based attacks based on querying boundary points in the estimated normal vector direction often suffer from inefficiency and convergence issues. In this paper, we propose a novel query-efficient curvature-aware geometric decision-based black-box attack (CGBA) that conducts boundary search along a semicircular path on a restricted 2D plane to ensure finding a boundary point successfully irrespective of the boundary curvature. While the proposed CGBA attack can work effectively for an arbitrary decision boundary, it is particularly efficient in exploiting the low curvature to craft high-quality adversarial examples, which is widely seen and experimentally verified in commonly used classifiers under non-targeted attacks. In contrast, the decision boundaries often exhibit higher curvature under targeted attacks. Thus, we develop a new query-efficient variant, CGBA-H, that is adapted for the targeted attack. In addition, we further design an algorithm to obtain a better initial boundary point at the expense of some extra queries, which considerably enhances the performance of the targeted attack. Extensive experiments are conducted to evaluate the performance of our proposed methods against some well-known classifiers on the ImageNet and CIFAR10 datasets, demonstrating the superiority of CGBA and CGBA-H over state-of-the-art non-targeted and targeted attacks, respectively. The source code is available at https://github.com/Farhamdur/CGBA.



## **43. MM-BD: Post-Training Detection of Backdoor Attacks with Arbitrary Backdoor Pattern Types Using a Maximum Margin Statistic**

cs.LG

**SubmitDate**: 2023-08-06    [abs](http://arxiv.org/abs/2205.06900v2) [paper-pdf](http://arxiv.org/pdf/2205.06900v2)

**Authors**: Hang Wang, Zhen Xiang, David J. Miller, George Kesidis

**Abstract**: Backdoor attacks are an important type of adversarial threat against deep neural network classifiers, wherein test samples from one or more source classes will be (mis)classified to the attacker's target class when a backdoor pattern is embedded. In this paper, we focus on the post-training backdoor defense scenario commonly considered in the literature, where the defender aims to detect whether a trained classifier was backdoor-attacked without any access to the training set. Many post-training detectors are designed to detect attacks that use either one or a few specific backdoor embedding functions (e.g., patch-replacement or additive attacks). These detectors may fail when the backdoor embedding function used by the attacker (unknown to the defender) is different from the backdoor embedding function assumed by the defender. In contrast, we propose a post-training defense that detects backdoor attacks with arbitrary types of backdoor embeddings, without making any assumptions about the backdoor embedding type. Our detector leverages the influence of the backdoor attack, independent of the backdoor embedding mechanism, on the landscape of the classifier's outputs prior to the softmax layer. For each class, a maximum margin statistic is estimated. Detection inference is then performed by applying an unsupervised anomaly detector to these statistics. Thus, our detector does not need any legitimate clean samples, and can efficiently detect backdoor attacks with arbitrary numbers of source classes. These advantages over several state-of-the-art methods are demonstrated on four datasets, for three different types of backdoor patterns, and for a variety of attack configurations. Finally, we propose a novel, general approach for backdoor mitigation once a detection is made. The mitigation approach was the runner-up at the first IEEE Trojan Removal Competition. The code is online available.



## **44. Phase-shifted Adversarial Training**

cs.LG

Proceedings of Uncertainty in Artificial Intelligence, 2023 (UAI  2023)

**SubmitDate**: 2023-08-06    [abs](http://arxiv.org/abs/2301.04785v2) [paper-pdf](http://arxiv.org/pdf/2301.04785v2)

**Authors**: Yeachan Kim, Seongyeon Kim, Ihyeok Seo, Bonggun Shin

**Abstract**: Adversarial training has been considered an imperative component for safely deploying neural network-based applications to the real world. To achieve stronger robustness, existing methods primarily focus on how to generate strong attacks by increasing the number of update steps, regularizing the models with the smoothed loss function, and injecting the randomness into the attack. Instead, we analyze the behavior of adversarial training through the lens of response frequency. We empirically discover that adversarial training causes neural networks to have low convergence to high-frequency information, resulting in highly oscillated predictions near each data. To learn high-frequency contents efficiently and effectively, we first prove that a universal phenomenon of frequency principle, i.e., \textit{lower frequencies are learned first}, still holds in adversarial training. Based on that, we propose phase-shifted adversarial training (PhaseAT) in which the model learns high-frequency components by shifting these frequencies to the low-frequency range where the fast convergence occurs. For evaluations, we conduct the experiments on CIFAR-10 and ImageNet with the adaptive attack carefully designed for reliable evaluation. Comprehensive results show that PhaseAT significantly improves the convergence for high-frequency information. This results in improved adversarial robustness by enabling the model to have smoothed predictions near each data.



## **45. WASMixer: Binary Obfuscation for WebAssembly**

cs.CR

**SubmitDate**: 2023-08-06    [abs](http://arxiv.org/abs/2308.03123v1) [paper-pdf](http://arxiv.org/pdf/2308.03123v1)

**Authors**: Shangtong Cao, Ningyu He, Yao Guo, Haoyu Wang

**Abstract**: WebAssembly (Wasm) is an emerging binary format that draws great attention from our community. However, Wasm binaries are weakly protected, as they can be read, edited, and manipulated by adversaries using either the officially provided readable text format (i.e., wat) or some advanced binary analysis tools. Reverse engineering of Wasm binaries is often used for nefarious intentions, e.g., identifying and exploiting both classic vulnerabilities and Wasm specific vulnerabilities exposed in the binaries. However, no Wasm-specific obfuscator is available in our community to secure the Wasm binaries. To fill the gap, in this paper, we present WASMixer, the first general-purpose Wasm binary obfuscator, enforcing data-level (string literals and function names) and code-level (control flow and instructions) obfuscation for Wasm binaries. We propose a series of key techniques to overcome challenges during Wasm binary rewriting, including an on-demand decryption method to minimize the impact brought by decrypting the data in memory area, and code splitting/reconstructing algorithms to handle structured control flow in Wasm. Extensive experiments demonstrate the correctness, effectiveness and efficiency of WASMixer. Our research has shed light on the promising direction of Wasm binary research, including Wasm code protection, Wasm binary diversification, and the attack-defense arm race of Wasm binaries.



## **46. SAAM: Stealthy Adversarial Attack on Monoculor Depth Estimation**

cs.CV

**SubmitDate**: 2023-08-06    [abs](http://arxiv.org/abs/2308.03108v1) [paper-pdf](http://arxiv.org/pdf/2308.03108v1)

**Authors**: Amira Guesmi, Muhammad Abdullah Hanif, Bassem Ouni, Muhammad Shafique

**Abstract**: In this paper, we investigate the vulnerability of MDE to adversarial patches. We propose a novel \underline{S}tealthy \underline{A}dversarial \underline{A}ttacks on \underline{M}DE (SAAM) that compromises MDE by either corrupting the estimated distance or causing an object to seamlessly blend into its surroundings. Our experiments, demonstrate that the designed stealthy patch successfully causes a DNN-based MDE to misestimate the depth of objects. In fact, our proposed adversarial patch achieves a significant 60\% depth error with 99\% ratio of the affected region. Importantly, despite its adversarial nature, the patch maintains a naturalistic appearance, making it inconspicuous to human observers. We believe that this work sheds light on the threat of adversarial attacks in the context of MDE on edge devices. We hope it raises awareness within the community about the potential real-life harm of such attacks and encourages further research into developing more robust and adaptive defense mechanisms.



## **47. Using Overlapping Methods to Counter Adversaries in Community Detection**

cs.SI

28 pages, 10 figures

**SubmitDate**: 2023-08-06    [abs](http://arxiv.org/abs/2308.03081v1) [paper-pdf](http://arxiv.org/pdf/2308.03081v1)

**Authors**: Benjamin A. Miller, Kevin Chan, Tina Eliassi-Rad

**Abstract**: When dealing with large graphs, community detection is a useful data triage tool that can identify subsets of the network that a data analyst should investigate. In an adversarial scenario, the graph may be manipulated to avoid scrutiny of certain nodes by the analyst. Robustness to such behavior is an important consideration for data analysts in high-stakes scenarios such as cyber defense and counterterrorism. In this paper, we evaluate the use of overlapping community detection methods in the presence of adversarial attacks aimed at lowering the priority of a specific vertex. We formulate the data analyst's choice as a Stackelberg game in which the analyst chooses a community detection method and the attacker chooses an attack strategy in response. Applying various attacks from the literature to seven real network datasets, we find that, when the attacker has a sufficient budget, overlapping community detection methods outperform non-overlapping methods, often overwhelmingly so. This is the case when the attacker can only add edges that connect to the target and when the capability is added to add edges between neighbors of the target. We also analyze the tradeoff between robustness in the presence of an attack and performance when there is no attack. Our extensible analytic framework enables network data analysts to take these considerations into account and incorporate new attacks and community detection methods as they are developed.



## **48. D4: Detection of Adversarial Diffusion Deepfakes Using Disjoint Ensembles**

cs.LG

**SubmitDate**: 2023-08-06    [abs](http://arxiv.org/abs/2202.05687v3) [paper-pdf](http://arxiv.org/pdf/2202.05687v3)

**Authors**: Ashish Hooda, Neal Mangaokar, Ryan Feng, Kassem Fawaz, Somesh Jha, Atul Prakash

**Abstract**: Detecting diffusion-generated deepfake images remains an open problem. Current detection methods fail against an adversary who adds imperceptible adversarial perturbations to the deepfake to evade detection. In this work, we propose Disjoint Diffusion Deepfake Detection (D4), a deepfake detector designed to improve black-box adversarial robustness beyond de facto solutions such as adversarial training. D4 uses an ensemble of models over disjoint subsets of the frequency spectrum to significantly improve adversarial robustness. Our key insight is to leverage a redundancy in the frequency domain and apply a saliency partitioning technique to disjointly distribute frequency components across multiple models. We formally prove that these disjoint ensembles lead to a reduction in the dimensionality of the input subspace where adversarial deepfakes lie, thereby making adversarial deepfakes harder to find for black-box attacks. We then empirically validate the D4 method against several black-box attacks and find that D4 significantly outperforms existing state-of-the-art defenses applied to diffusion-generated deepfake detection. We also demonstrate that D4 provides robustness against adversarial deepfakes from unseen data distributions as well as unseen generative techniques.



## **49. FLAME: Taming Backdoors in Federated Learning (Extended Version 1)**

cs.CR

This extended version incorporates a novel section (Section 10) that  provides a comprehensive analysis of recent proposed attacks, notably "3DFed:  Adaptive and extensible framework for covert backdoor attack in federated  learning" by Li et al. This new section addresses flawed assertions made in  the papers that aim to bypass FLAME or misinterpreted its fundamental design  principles

**SubmitDate**: 2023-08-05    [abs](http://arxiv.org/abs/2101.02281v5) [paper-pdf](http://arxiv.org/pdf/2101.02281v5)

**Authors**: Thien Duc Nguyen, Phillip Rieger, Huili Chen, Hossein Yalame, Helen Möllering, Hossein Fereidooni, Samuel Marchal, Markus Miettinen, Azalia Mirhoseini, Shaza Zeitouni, Farinaz Koushanfar, Ahmad-Reza Sadeghi, Thomas Schneider

**Abstract**: Federated Learning (FL) is a collaborative machine learning approach allowing participants to jointly train a model without having to share their private, potentially sensitive local datasets with others. Despite its benefits, FL is vulnerable to backdoor attacks, in which an adversary injects manipulated model updates into the model aggregation process so that the resulting model will provide targeted false predictions for specific adversary-chosen inputs. Proposed defenses against backdoor attacks based on detecting and filtering out malicious model updates consider only very specific and limited attacker models, whereas defenses based on differential privacy-inspired noise injection significantly deteriorate the benign performance of the aggregated model. To address these deficiencies, we introduce FLAME, a defense framework that estimates the sufficient amount of noise to be injected to ensure the elimination of backdoors while maintaining the model performance. To minimize the required amount of noise, FLAME uses a model clustering and weight clipping approach. Our evaluation of FLAME on several datasets stemming from application areas including image classification, word prediction, and IoT intrusion detection demonstrates that FLAME removes backdoors effectively with a negligible impact on the benign performance of the models. Furthermore, following the considerable attention that our research has received after its presentation at USENIX SEC 2022, FLAME has become the subject of numerous investigations proposing diverse attack methodologies in an attempt to circumvent it. As a response to these endeavors, we provide a comprehensive analysis of these attempts. Our findings show that these papers (e.g., 3DFed [36]) have not fully comprehended nor correctly employed the fundamental principles underlying FLAME, i.e., our defense mechanism effectively repels these attempted attacks.



## **50. An AI-Enabled Framework to Defend Ingenious MDT-based Attacks on the Emerging Zero Touch Cellular Networks**

cs.LG

15 pages, 5 figures, 1 table

**SubmitDate**: 2023-08-05    [abs](http://arxiv.org/abs/2308.02923v1) [paper-pdf](http://arxiv.org/pdf/2308.02923v1)

**Authors**: Aneeqa Ijaz, Waseem Raza, Hasan Farooq, Marvin Manalastas, Ali Imran

**Abstract**: Deep automation provided by self-organizing network (SON) features and their emerging variants such as zero touch automation solutions is a key enabler for increasingly dense wireless networks and pervasive Internet of Things (IoT). To realize their objectives, most automation functionalities rely on the Minimization of Drive Test (MDT) reports. The MDT reports are used to generate inferences about network state and performance, thus dynamically change network parameters accordingly. However, the collection of MDT reports from commodity user devices, particularly low cost IoT devices, make them a vulnerable entry point to launch an adversarial attack on emerging deeply automated wireless networks. This adds a new dimension to the security threats in the IoT and cellular networks. Existing literature on IoT, SON, or zero touch automation does not address this important problem. In this paper, we investigate an impactful, first of its kind adversarial attack that can be launched by exploiting the malicious MDT reports from the compromised user equipment (UE). We highlight the detrimental repercussions of this attack on the performance of common network automation functions. We also propose a novel Malicious MDT Reports Identification framework (MRIF) as a countermeasure to detect and eliminate the malicious MDT reports using Machine Learning and verify it through a use-case. Thus, the defense mechanism can provide the resilience and robustness for zero touch automation SON engines against the adversarial MDT attacks



