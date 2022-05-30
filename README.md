# Latest Adversarial Attack Papers
**update at 2022-05-31 06:31:32**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. A Single-Adversary-Single-Detector Zero-Sum Game in Networked Control Systems**

math.OC

6 pages, 6 figures, 1 table, accepted to the 9th IFAC Conference on  Networked Systems, Zurich, July 2022

**SubmitDate**: 2022-05-27    [paper-pdf](http://arxiv.org/pdf/2205.14001v1)

**Authors**: Anh Tung Nguyen, André M. H. Teixeira, Alexander Medvedev

**Abstracts**: This paper proposes a game-theoretic approach to address the problem of optimal sensor placement for detecting cyber-attacks in networked control systems. The problem is formulated as a zero-sum game with two players, namely a malicious adversary and a detector. Given a protected target vertex, the detector places a sensor at a single vertex to monitor the system and detect the presence of the adversary. On the other hand, the adversary selects a single vertex through which to conduct a cyber-attack that maximally disrupts the target vertex while remaining undetected by the detector. As our first contribution, for a given pair of attack and monitor vertices and a known target vertex, the game payoff function is defined as the output-to-output gain of the respective system. Then, the paper characterizes the set of feasible actions by the detector that ensures bounded values of the game payoff. Finally, an algebraic sufficient condition is proposed to examine whether a given vertex belongs to the set of feasible monitor vertices. The optimal sensor placement is then determined by computing the mixed-strategy Nash equilibrium of the zero-sum game through linear programming. The approach is illustrated via a numerical example of a 10-vertex networked control system with a given target vertex.



## **2. Standalone Neural ODEs with Sensitivity Analysis**

cs.LG

25 pages, 15 figures

**SubmitDate**: 2022-05-27    [paper-pdf](http://arxiv.org/pdf/2205.13933v1)

**Authors**: Rym Jaroudi, Lukáš Malý, Gabriel Eilertsen, Tomas B. Johansson, Jonas Unger, George Baravdish

**Abstracts**: This paper presents the Standalone Neural ODE (sNODE), a continuous-depth neural ODE model capable of describing a full deep neural network. This uses a novel nonlinear conjugate gradient (NCG) descent optimization scheme for training, where the Sobolev gradient can be incorporated to improve smoothness of model weights. We also present a general formulation of the neural sensitivity problem and show how it is used in the NCG training. The sensitivity analysis provides a reliable measure of uncertainty propagation throughout a network, and can be used to study model robustness and to generate adversarial attacks. Our evaluations demonstrate that our novel formulations lead to increased robustness and performance as compared to ResNet models, and that it opens up for new opportunities for designing and developing machine learning with improved explainability.



## **3. Evaluating the Robustness of Deep Reinforcement Learning for Autonomous and Adversarial Policies in a Multi-agent Urban Driving Environment**

cs.AI

**SubmitDate**: 2022-05-27    [paper-pdf](http://arxiv.org/pdf/2112.11947v2)

**Authors**: Aizaz Sharif, Dusica Marijan

**Abstracts**: Deep reinforcement learning is actively used for training autonomous and adversarial car policies in a simulated driving environment. Due to the large availability of various reinforcement learning algorithms and the lack of their systematic comparison across different driving scenarios, we are unsure of which ones are more effective for training and testing autonomous car software in single-agent as well as multi-agent driving environments. A benchmarking framework for the comparison of deep reinforcement learning in a vision-based autonomous driving will open up the possibilities for training better autonomous car driving policies. Furthermore, autonomous cars trained on deep reinforcement learning-based algorithms are known for being vulnerable to adversarial attacks. To guard against adversarial attacks, we can train autonomous cars on adversarial driving policies. However, we lack the knowledge of which deep reinforcement learning algorithms would act as good adversarial agents able to effectively test autonomous cars. To address these challenges, we provide an open and reusable benchmarking framework for systematic evaluation and comparative analysis of deep reinforcement learning algorithms for autonomous and adversarial driving in a single- and multi-agent environment. Using the framework, we perform a comparative study of five discrete and two continuous action space deep reinforcement learning algorithms. We run the experiments in a vision-only high fidelity urban driving simulated environments. The results indicate that only some of the deep reinforcement learning algorithms perform consistently better across single and multi-agent scenarios when trained in a multi-agent-only setting.



## **4. Adversarial Deep Reinforcement Learning for Improving the Robustness of Multi-agent Autonomous Driving Policies**

cs.AI

**SubmitDate**: 2022-05-27    [paper-pdf](http://arxiv.org/pdf/2112.11937v2)

**Authors**: Aizaz Sharif, Dusica Marijan

**Abstracts**: Autonomous cars are well known for being vulnerable to adversarial attacks that can compromise the safety of the car and pose danger to other road users. To effectively defend against adversaries, it is required to not only test autonomous cars for finding driving errors, but to improve the robustness of the cars to these errors. To this end, in this paper, we propose a two-step methodology for autonomous cars that consists of (i) finding failure states in autonomous cars by training the adversarial driving agent, and (ii) improving the robustness of autonomous cars by retraining them with effective adversarial inputs. Our methodology supports testing ACs in a multi-agent environment, where we train and compare adversarial car policy on two custom reward functions to test the driving control decision of autonomous cars. We run experiments in a vision-based high fidelity urban driving simulated environment. Our results show that adversarial testing can be used for finding erroneous autonomous driving behavior, followed by adversarial training for improving the robustness of deep reinforcement learning based autonomous driving policies. We demonstrate that the autonomous cars retrained using the effective adversarial inputs noticeably increase the performance of their driving policies in terms of reduced collision and offroad steering errors.



## **5. fakeWeather: Adversarial Attacks for Deep Neural Networks Emulating Weather Conditions on the Camera Lens of Autonomous Systems**

cs.LG

To appear at the 2022 International Joint Conference on Neural  Networks (IJCNN), at the 2022 IEEE World Congress on Computational  Intelligence (WCCI), July 2022, Padua, Italy

**SubmitDate**: 2022-05-27    [paper-pdf](http://arxiv.org/pdf/2205.13807v1)

**Authors**: Alberto Marchisio, Giovanni Caramia, Maurizio Martina, Muhammad Shafique

**Abstracts**: Recently, Deep Neural Networks (DNNs) have achieved remarkable performances in many applications, while several studies have enhanced their vulnerabilities to malicious attacks. In this paper, we emulate the effects of natural weather conditions to introduce plausible perturbations that mislead the DNNs. By observing the effects of such atmospheric perturbations on the camera lenses, we model the patterns to create different masks that fake the effects of rain, snow, and hail. Even though the perturbations introduced by our attacks are visible, their presence remains unnoticed due to their association with natural events, which can be especially catastrophic for fully-autonomous and unmanned vehicles. We test our proposed fakeWeather attacks on multiple Convolutional Neural Network and Capsule Network models, and report noticeable accuracy drops in the presence of such adversarial perturbations. Our work introduces a new security threat for DNNs, which is especially severe for safety-critical applications and autonomous systems.



## **6. Face Morphing: Fooling a Face Recognition System Is Simple!**

cs.CV

**SubmitDate**: 2022-05-27    [paper-pdf](http://arxiv.org/pdf/2205.13796v1)

**Authors**: Stefan Hörmann, Tianlin Kong, Torben Teepe, Fabian Herzog, Martin Knoche, Gerhard Rigoll

**Abstracts**: State-of-the-art face recognition (FR) approaches have shown remarkable results in predicting whether two faces belong to the same identity, yielding accuracies between 92% and 100% depending on the difficulty of the protocol. However, the accuracy drops substantially when exposed to morphed faces, specifically generated to look similar to two identities. To generate morphed faces, we integrate a simple pretrained FR model into a generative adversarial network (GAN) and modify several loss functions for face morphing. In contrast to previous works, our approach and analyses are not limited to pairs of frontal faces with the same ethnicity and gender. Our qualitative and quantitative results affirm that our approach achieves a seamless change between two faces even in unconstrained scenarios. Despite using features from a simpler FR model for face morphing, we demonstrate that even recent FR systems struggle to distinguish the morphed face from both identities obtaining an accuracy of only 55-70%. Besides, we provide further insights into how knowing the FR system makes it particularly vulnerable to face morphing attacks.



## **7. Adversarial attacks and defenses in Speaker Recognition Systems: A survey**

cs.CR

38pages, 2 figures, 2 tables. Journal of Systems Architecture,2022

**SubmitDate**: 2022-05-27    [paper-pdf](http://arxiv.org/pdf/2205.13685v1)

**Authors**: Jiahe Lan, Rui Zhang, Zheng Yan, Jie Wang, Yu Chen, Ronghui Hou

**Abstracts**: Speaker recognition has become very popular in many application scenarios, such as smart homes and smart assistants, due to ease of use for remote control and economic-friendly features. The rapid development of SRSs is inseparable from the advancement of machine learning, especially neural networks. However, previous work has shown that machine learning models are vulnerable to adversarial attacks in the image domain, which inspired researchers to explore adversarial attacks and defenses in Speaker Recognition Systems (SRS). Unfortunately, existing literature lacks a thorough review of this topic. In this paper, we fill this gap by performing a comprehensive survey on adversarial attacks and defenses in SRSs. We first introduce the basics of SRSs and concepts related to adversarial attacks. Then, we propose two sets of criteria to evaluate the performance of attack methods and defense methods in SRSs, respectively. After that, we provide taxonomies of existing attack methods and defense methods, and further review them by employing our proposed criteria. Finally, based on our review, we find some open issues and further specify a number of future directions to motivate the research of SRSs security.



## **8. Sequential Nature of Recommender Systems Disrupts the Evaluation Process**

cs.IR

To Appear in Third International Workshop on Algorithmic Bias in  Search and Recommendation (Bias 2022)

**SubmitDate**: 2022-05-26    [paper-pdf](http://arxiv.org/pdf/2205.13681v1)

**Authors**: Ali Shirali

**Abstracts**: Datasets are often generated in a sequential manner, where the previous samples and intermediate decisions or interventions affect subsequent samples. This is especially prominent in cases where there are significant human-AI interactions, such as in recommender systems. To characterize the importance of this relationship across samples, we propose to use adversarial attacks on popular evaluation processes. We present sequence-aware boosting attacks and provide a lower bound on the amount of extra information that can be exploited from a confidential test set solely based on the order of the observed data. We use real and synthetic data to test our methods and show that the evaluation process on the MovieLense-100k dataset can be affected by $\sim1\%$ which is important when considering the close competition. Codes are publicly available.



## **9. Membership Inference Attack Using Self Influence Functions**

cs.LG

**SubmitDate**: 2022-05-26    [paper-pdf](http://arxiv.org/pdf/2205.13680v1)

**Authors**: Gilad Cohen, Raja Giryes

**Abstracts**: Member inference (MI) attacks aim to determine if a specific data sample was used to train a machine learning model. Thus, MI is a major privacy threat to models trained on private sensitive data, such as medical records. In MI attacks one may consider the black-box settings, where the model's parameters and activations are hidden from the adversary, or the white-box case where they are available to the attacker. In this work, we focus on the latter and present a novel MI attack for it that employs influence functions, or more specifically the samples' self-influence scores, to perform the MI prediction. We evaluate our attack on CIFAR-10, CIFAR-100, and Tiny ImageNet datasets, using versatile architectures such as AlexNet, ResNet, and DenseNet. Our attack method achieves new state-of-the-art results for both training with and without data augmentations. Code is available at https://github.com/giladcohen/sif_mi_attack.



## **10. On the Anonymity of Peer-To-Peer Network Anonymity Schemes Used by Cryptocurrencies**

cs.CR

**SubmitDate**: 2022-05-26    [paper-pdf](http://arxiv.org/pdf/2201.11860v2)

**Authors**: Piyush Kumar Sharma, Devashish Gosain, Claudia Diaz

**Abstracts**: Cryptocurrency systems can be subject to deanonimization attacks by exploiting the network-level communication on their peer-to-peer network. Adversaries who control a set of colluding node(s) within the peer-to-peer network can observe transactions being exchanged and infer the parties involved. Thus, various network anonymity schemes have been proposed to mitigate this problem, with some solutions providing theoretical anonymity guarantees.   In this work, we model such peer-to-peer network anonymity solutions and evaluate their anonymity guarantees. To do so, we propose a novel framework that uses Bayesian inference to obtain the probability distributions linking transactions to their possible originators. We characterize transaction anonymity with those distributions, using entropy as metric of adversarial uncertainty on the originator's identity. In particular, we model Dandelion, Dandelion++ and Lightning Network. We study different configurations and demonstrate that none of them offers acceptable anonymity to their users. For instance, our analysis reveals that in the widely deployed Lightning Network, with 1% strategically chosen colluding nodes the adversary can uniquely determine the originator for about 50% of the total transactions in the network. In Dandelion, an adversary that controls 15% of the nodes has on average uncertainty among only 8 possible originators. Moreover, we observe that due to the way Dandelion and Dandelion++ are designed, increasing the network size does not correspond to an increase in the anonymity set of potential originators. Alarmingly, our longitudinal analysis of Lightning Network reveals rather an inverse trend -- with the growth of the network the overall anonymity decreases.



## **11. Towards Practical Deployment-Stage Backdoor Attack on Deep Neural Networks**

cs.CR

**SubmitDate**: 2022-05-26    [paper-pdf](http://arxiv.org/pdf/2111.12965v2)

**Authors**: Xiangyu Qi, Tinghao Xie, Ruizhe Pan, Jifeng Zhu, Yong Yang, Kai Bu

**Abstracts**: One major goal of the AI security community is to securely and reliably produce and deploy deep learning models for real-world applications. To this end, data poisoning based backdoor attacks on deep neural networks (DNNs) in the production stage (or training stage) and corresponding defenses are extensively explored in recent years. Ironically, backdoor attacks in the deployment stage, which can often happen in unprofessional users' devices and are thus arguably far more threatening in real-world scenarios, draw much less attention of the community. We attribute this imbalance of vigilance to the weak practicality of existing deployment-stage backdoor attack algorithms and the insufficiency of real-world attack demonstrations. To fill the blank, in this work, we study the realistic threat of deployment-stage backdoor attacks on DNNs. We base our study on a commonly used deployment-stage attack paradigm -- adversarial weight attack, where adversaries selectively modify model weights to embed backdoor into deployed DNNs. To approach realistic practicality, we propose the first gray-box and physically realizable weights attack algorithm for backdoor injection, namely subnet replacement attack (SRA), which only requires architecture information of the victim model and can support physical triggers in the real world. Extensive experimental simulations and system-level real-world attack demonstrations are conducted. Our results not only suggest the effectiveness and practicality of the proposed attack algorithm, but also reveal the practical risk of a novel type of computer virus that may widely spread and stealthily inject backdoor into DNN models in user devices. By our study, we call for more attention to the vulnerability of DNNs in the deployment stage.



## **12. Denial-of-Service Attack on Object Detection Model Using Universal Adversarial Perturbation**

cs.CV

**SubmitDate**: 2022-05-26    [paper-pdf](http://arxiv.org/pdf/2205.13618v1)

**Authors**: Avishag Shapira, Alon Zolfi, Luca Demetrio, Battista Biggio, Asaf Shabtai

**Abstracts**: Adversarial attacks against deep learning-based object detectors have been studied extensively in the past few years. The proposed attacks aimed solely at compromising the models' integrity (i.e., trustworthiness of the model's prediction), while adversarial attacks targeting the models' availability, a critical aspect in safety-critical domains such as autonomous driving, have not been explored by the machine learning research community. In this paper, we propose NMS-Sponge, a novel approach that negatively affects the decision latency of YOLO, a state-of-the-art object detector, and compromises the model's availability by applying a universal adversarial perturbation (UAP). In our experiments, we demonstrate that the proposed UAP is able to increase the processing time of individual frames by adding "phantom" objects while preserving the detection of the original objects.



## **13. Circumventing Backdoor Defenses That Are Based on Latent Separability**

cs.LG

**SubmitDate**: 2022-05-26    [paper-pdf](http://arxiv.org/pdf/2205.13613v1)

**Authors**: Xiangyu Qi, Tinghao Xie, Saeed Mahloujifar, Prateek Mittal

**Abstracts**: Deep learning models are vulnerable to backdoor poisoning attacks. In particular, adversaries can embed hidden backdoors into a model by only modifying a very small portion of its training data. On the other hand, it has also been commonly observed that backdoor poisoning attacks tend to leave a tangible signature in the latent space of the backdoored model i.e. poison samples and clean samples form two separable clusters in the latent space. These observations give rise to the popularity of latent separability assumption, which states that the backdoored DNN models will learn separable latent representations for poison and clean populations. A number of popular defenses (e.g. Spectral Signature, Activation Clustering, SCAn, etc.) are exactly built upon this assumption. However, in this paper, we show that the latent separation can be significantly suppressed via designing adaptive backdoor poisoning attacks with more sophisticated poison strategies, which consequently render state-of-the-art defenses based on this assumption less effective (and often completely fail). More interestingly, we find that our adaptive attacks can even evade some other typical backdoor defenses that do not explicitly build on this separability assumption. Our results show that adaptive backdoor poisoning attacks that can breach the latent separability assumption should be seriously considered for evaluating existing and future defenses.



## **14. PerDoor: Persistent Non-Uniform Backdoors in Federated Learning using Adversarial Perturbations**

cs.CR

**SubmitDate**: 2022-05-26    [paper-pdf](http://arxiv.org/pdf/2205.13523v1)

**Authors**: Manaar Alam, Esha Sarkar, Michail Maniatakos

**Abstracts**: Federated Learning (FL) enables numerous participants to train deep learning models collaboratively without exposing their personal, potentially sensitive data, making it a promising solution for data privacy in collaborative training. The distributed nature of FL and unvetted data, however, makes it inherently vulnerable to backdoor attacks: In this scenario, an adversary injects backdoor functionality into the centralized model during training, which can be triggered to cause the desired misclassification for a specific adversary-chosen input. A range of prior work establishes successful backdoor injection in an FL system; however, these backdoors are not demonstrated to be long-lasting. The backdoor functionality does not remain in the system if the adversary is removed from the training process since the centralized model parameters continuously mutate during successive FL training rounds. Therefore, in this work, we propose PerDoor, a persistent-by-construction backdoor injection technique for FL, driven by adversarial perturbation and targeting parameters of the centralized model that deviate less in successive FL rounds and contribute the least to the main task accuracy. An exhaustive evaluation considering an image classification scenario portrays on average $10.5\times$ persistence over multiple FL rounds compared to traditional backdoor attacks. Through experiments, we further exhibit the potency of PerDoor in the presence of state-of-the-art backdoor prevention techniques in an FL system. Additionally, the operation of adversarial perturbation also assists PerDoor in developing non-uniform trigger patterns for backdoor inputs compared to uniform triggers (with fixed patterns and locations) of existing backdoor techniques, which are prone to be easily mitigated.



## **15. An Analytic Framework for Robust Training of Artificial Neural Networks**

cs.LG

**SubmitDate**: 2022-05-26    [paper-pdf](http://arxiv.org/pdf/2205.13502v1)

**Authors**: Ramin Barati, Reza Safabakhsh, Mohammad Rahmati

**Abstracts**: The reliability of a learning model is key to the successful deployment of machine learning in various industries. Creating a robust model, particularly one unaffected by adversarial attacks, requires a comprehensive understanding of the adversarial examples phenomenon. However, it is difficult to describe the phenomenon due to the complicated nature of the problems in machine learning. Consequently, many studies investigate the phenomenon by proposing a simplified model of how adversarial examples occur and validate it by predicting some aspect of the phenomenon. While these studies cover many different characteristics of the adversarial examples, they have not reached a holistic approach to the geometric and analytic modeling of the phenomenon. This paper propose a formal framework to study the phenomenon in learning theory and make use of complex analysis and holomorphicity to offer a robust learning rule for artificial neural networks. With the help of complex analysis, we can effortlessly move between geometric and analytic perspectives of the phenomenon and offer further insights on the phenomenon by revealing its connection with harmonic functions. Using our model, we can explain some of the most intriguing characteristics of adversarial examples, including transferability of adversarial examples, and pave the way for novel approaches to mitigate the effects of the phenomenon.



## **16. Towards Understanding and Harnessing the Effect of Image Transformation in Adversarial Detection**

cs.CV

**SubmitDate**: 2022-05-26    [paper-pdf](http://arxiv.org/pdf/2201.01080v3)

**Authors**: Hui Liu, Bo Zhao, Yuefeng Peng, Weidong Li, Peng Liu

**Abstracts**: Deep neural networks (DNNs) are threatened by adversarial examples. Adversarial detection, which distinguishes adversarial images from benign images, is fundamental for robust DNN-based services. Image transformation is one of the most effective approaches to detect adversarial examples. During the last few years, a variety of image transformations have been studied and discussed to design reliable adversarial detectors. In this paper, we systematically synthesize the recent progress on adversarial detection via image transformations with a novel classification method. Then, we conduct extensive experiments to test the detection performance of image transformations against state-of-the-art adversarial attacks. Furthermore, we reveal that each individual transformation is not capable of detecting adversarial examples in a robust way, and propose a DNN-based approach referred to as \emph{AdvJudge}, which combines scores of 9 image transformations. Without knowing which individual scores are misleading or not misleading, AdvJudge can make the right judgment, and achieve a significant improvement in detection rate. Finally, we utilize an explainable AI tool to show the contribution of each image transformation to adversarial detection. Experimental results show that the contribution of image transformations to adversarial detection is significantly different, the combination of them can significantly improve the generic detection ability against state-of-the-art adversarial attacks.



## **17. A Physical-World Adversarial Attack Against 3D Face Recognition**

cs.CV

10 pages, 5 figures, Submit to NeurIPS 2022

**SubmitDate**: 2022-05-26    [paper-pdf](http://arxiv.org/pdf/2205.13412v1)

**Authors**: Yanjie Li, Yiquan Li, Bin Xiao

**Abstracts**: 3D face recognition systems have been widely employed in intelligent terminals, among which structured light imaging is a common method to measure the 3D shape. However, this method could be easily attacked, leading to inaccurate 3D face recognition. In this paper, we propose a novel, physically-achievable attack on the fringe structured light system, named structured light attack. The attack utilizes a projector to project optical adversarial fringes on faces to generate point clouds with well-designed noises. We firstly propose a 3D transform-invariant loss function to enhance the robustness of 3D adversarial examples in the physical-world attack. Then we reverse the 3D adversarial examples to the projector's input to place noises on phase-shift images, which models the process of structured light imaging. A real-world structured light system is constructed for the attack and several state-of-the-art 3D face recognition neural networks are tested. Experiments show that our method can attack the physical system successfully and only needs minor modifications of projected images.



## **18. BppAttack: Stealthy and Efficient Trojan Attacks against Deep Neural Networks via Image Quantization and Contrastive Adversarial Learning**

cs.CV

**SubmitDate**: 2022-05-26    [paper-pdf](http://arxiv.org/pdf/2205.13383v1)

**Authors**: Zhenting Wang, Juan Zhai, Shiqing Ma

**Abstracts**: Deep neural networks are vulnerable to Trojan attacks. Existing attacks use visible patterns (e.g., a patch or image transformations) as triggers, which are vulnerable to human inspection. In this paper, we propose stealthy and efficient Trojan attacks, BppAttack. Based on existing biology literature on human visual systems, we propose to use image quantization and dithering as the Trojan trigger, making imperceptible changes. It is a stealthy and efficient attack without training auxiliary models. Due to the small changes made to images, it is hard to inject such triggers during training. To alleviate this problem, we propose a contrastive learning based approach that leverages adversarial attacks to generate negative sample pairs so that the learned trigger is precise and accurate. The proposed method achieves high attack success rates on four benchmark datasets, including MNIST, CIFAR-10, GTSRB, and CelebA. It also effectively bypasses existing Trojan defenses and human inspection. Our code can be found in https://github.com/RU-System-Software-and-Security/BppAttack.



## **19. Random Walks for Adversarial Meshes**

cs.CV

**SubmitDate**: 2022-05-26    [paper-pdf](http://arxiv.org/pdf/2202.07453v2)

**Authors**: Amir Belder, Gal Yefet, Ran Ben Izhak, Ayellet Tal

**Abstracts**: A polygonal mesh is the most-commonly used representation of surfaces in computer graphics. Therefore, it is not surprising that a number of mesh classification networks have recently been proposed. However, while adversarial attacks are wildly researched in 2D, the field of adversarial meshes is under explored. This paper proposes a novel, unified, and general adversarial attack, which leads to misclassification of several state-of-the-art mesh classification neural networks. Our attack approach is black-box, i.e. it has access only to the network's predictions, but not to the network's full architecture or gradients. The key idea is to train a network to imitate a given classification network. This is done by utilizing random walks along the mesh surface, which gather geometric information. These walks provide insight onto the regions of the mesh that are important for the correct prediction of the given classification network. These mesh regions are then modified more than other regions in order to attack the network in a manner that is barely visible to the naked eye.



## **20. Denial-of-Service Attacks on Learned Image Compression**

cs.CV

**SubmitDate**: 2022-05-26    [paper-pdf](http://arxiv.org/pdf/2205.13253v1)

**Authors**: Kang Liu, Di Wu, Yiru Wang, Dan Feng, Benjamin Tan, Siddharth Garg

**Abstracts**: Deep learning techniques have shown promising results in image compression, with competitive bitrate and image reconstruction quality from compressed latent. However, while image compression has progressed towards higher peak signal-to-noise ratio (PSNR) and fewer bits per pixel (bpp), their robustness to corner-case images has never received deliberation. In this work, we, for the first time, investigate the robustness of image compression systems where imperceptible perturbation of input images can precipitate a significant increase in the bitrate of their compressed latent. To characterize the robustness of state-of-the-art learned image compression, we mount white and black-box attacks. Our results on several image compression models with various bitrate qualities show that they are surprisingly fragile, where the white-box attack achieves up to 56.326x and black-box 1.947x bpp change. To improve robustness, we propose a novel model which incorporates attention modules and a basic factorized entropy model, resulting in a promising trade-off between the PSNR/bpp ratio and robustness to adversarial attacks that surpasses existing learned image compressors.



## **21. Certified Robustness Against Natural Language Attacks by Causal Intervention**

cs.LG

**SubmitDate**: 2022-05-26    [paper-pdf](http://arxiv.org/pdf/2205.12331v2)

**Authors**: Haiteng Zhao, Chang Ma*, Xinshuai Dong, Anh Tuan Luu, Zhi-Hong Deng, Hanwang Zhang

**Abstracts**: Deep learning models have achieved great success in many fields, yet they are vulnerable to adversarial examples. This paper follows a causal perspective to look into the adversarial vulnerability and proposes Causal Intervention by Semantic Smoothing (CISS), a novel framework towards robustness against natural language attacks. Instead of merely fitting observational data, CISS learns causal effects p(y|do(x)) by smoothing in the latent semantic space to make robust predictions, which scales to deep architectures and avoids tedious construction of noise customized for specific attacks. CISS is provably robust against word substitution attacks, as well as empirically robust even when perturbations are strengthened by unknown attack algorithms. For example, on YELP, CISS surpasses the runner-up by 6.7% in terms of certified robustness against word substitutions, and achieves 79.4% empirical robustness when syntactic attacks are integrated.



## **22. Transferable Adversarial Attack based on Integrated Gradients**

cs.LG

**SubmitDate**: 2022-05-26    [paper-pdf](http://arxiv.org/pdf/2205.13152v1)

**Authors**: Yi Huang, Adams Wai-Kin Kong

**Abstracts**: The vulnerability of deep neural networks to adversarial examples has drawn tremendous attention from the community. Three approaches, optimizing standard objective functions, exploiting attention maps, and smoothing decision surfaces, are commonly used to craft adversarial examples. By tightly integrating the three approaches, we propose a new and simple algorithm named Transferable Attack based on Integrated Gradients (TAIG) in this paper, which can find highly transferable adversarial examples for black-box attacks. Unlike previous methods using multiple computational terms or combining with other methods, TAIG integrates the three approaches into one single term. Two versions of TAIG that compute their integrated gradients on a straight-line path and a random piecewise linear path are studied. Both versions offer strong transferability and can seamlessly work together with the previous methods. Experimental results demonstrate that TAIG outperforms the state-of-the-art methods. The code will available at https://github.com/yihuang2016/TAIG



## **23. Textual Backdoor Attacks with Iterative Trigger Injection**

cs.CL

**SubmitDate**: 2022-05-25    [paper-pdf](http://arxiv.org/pdf/2205.12700v1)

**Authors**: Jun Yan, Vansh Gupta, Xiang Ren

**Abstracts**: The backdoor attack has become an emerging threat for Natural Language Processing (NLP) systems. A victim model trained on poisoned data can be embedded with a "backdoor", making it predict the adversary-specified output (e.g., the positive sentiment label) on inputs satisfying the trigger pattern (e.g., containing a certain keyword). In this paper, we demonstrate that it's possible to design an effective and stealthy backdoor attack by iteratively injecting "triggers" into a small set of training data. While all triggers are common words that fit into the context, our poisoning process strongly associates them with the target label, forming the model backdoor. Experiments on sentiment analysis and hate speech detection show that our proposed attack is both stealthy and effective, raising alarm on the usage of untrusted training data. We further propose a defense method to combat this threat.



## **24. Deniable Steganography**

cs.CR

**SubmitDate**: 2022-05-25    [paper-pdf](http://arxiv.org/pdf/2205.12587v1)

**Authors**: Yong Xu, Zhihua Xia, Zichi Wang, Xinpeng Zhang, Jian Weng

**Abstracts**: Steganography conceals the secret message into the cover media, generating a stego media which can be transmitted on public channels without drawing suspicion. As its countermeasure, steganalysis mainly aims to detect whether the secret message is hidden in a given media. Although the steganography techniques are improving constantly, the sophisticated steganalysis can always break a known steganographic method to some extent. With a stego media discovered, the adversary could find out the sender or receiver and coerce them to disclose the secret message, which we name as coercive attack in this paper. Inspired by the idea of deniable encryption, we build up the concepts of deniable steganography for the first time and discuss the feasible constructions for it. As an example, we propose a receiver-deniable steganographic scheme to deal with the receiver-side coercive attack using deep neural networks (DNN). Specifically, besides the real secret message, a piece of fake message is also embedded into the cover. On the receiver side, the real message can be extracted with an extraction module; while once the receiver has to surrender a piece of secret message under coercive attack, he can extract the fake message to deceive the adversary with another extraction module. Experiments demonstrate the scalability and sensitivity of the DNN-based receiver-deniable steganographic scheme.



## **25. Misleading Deep-Fake Detection with GAN Fingerprints**

cs.CV

In IEEE Deep Learning and Security Workshop (DLS) 2022

**SubmitDate**: 2022-05-25    [paper-pdf](http://arxiv.org/pdf/2205.12543v1)

**Authors**: Vera Wesselkamp, Konrad Rieck, Daniel Arp, Erwin Quiring

**Abstracts**: Generative adversarial networks (GANs) have made remarkable progress in synthesizing realistic-looking images that effectively outsmart even humans. Although several detection methods can recognize these deep fakes by checking for image artifacts from the generation process, multiple counterattacks have demonstrated their limitations. These attacks, however, still require certain conditions to hold, such as interacting with the detection method or adjusting the GAN directly. In this paper, we introduce a novel class of simple counterattacks that overcomes these limitations. In particular, we show that an adversary can remove indicative artifacts, the GAN fingerprint, directly from the frequency spectrum of a generated image. We explore different realizations of this removal, ranging from filtering high frequencies to more nuanced frequency-peak cleansing. We evaluate the performance of our attack with different detection methods, GAN architectures, and datasets. Our results show that an adversary can often remove GAN fingerprints and thus evade the detection of generated images.



## **26. A Survey of Graph-Theoretic Approaches for Analyzing the Resilience of Networked Control Systems**

eess.SY

**SubmitDate**: 2022-05-25    [paper-pdf](http://arxiv.org/pdf/2205.12498v1)

**Authors**: Mohammad Pirani, Aritra Mitra, Shreyas Sundaram

**Abstracts**: As the scale of networked control systems increases and interactions between different subsystems become more sophisticated, questions of the resilience of such networks increase in importance. The need to redefine classical system and control-theoretic notions using the language of graphs has recently started to gain attention as a fertile and important area of research. This paper presents an overview of graph-theoretic methods for analyzing the resilience of networked control systems. We discuss various distributed algorithms operating on networked systems and investigate their resilience against adversarial actions by looking at the structural properties of their underlying networks. We present graph-theoretic methods to quantify the attack impact, and reinterpret some system-theoretic notions of robustness from a graph-theoretic standpoint to mitigate the impact of the attacks. Moreover, we discuss miscellaneous problems in the security of networked control systems which use graph-theory as a tool in their analyses. We conclude by introducing some avenues for further research in this field.



## **27. Label Leakage and Protection from Forward Embedding in Vertical Federated Learning**

cs.LG

**SubmitDate**: 2022-05-25    [paper-pdf](http://arxiv.org/pdf/2203.01451v3)

**Authors**: Jiankai Sun, Xin Yang, Yuanshun Yao, Chong Wang

**Abstracts**: Vertical federated learning (vFL) has gained much attention and been deployed to solve machine learning problems with data privacy concerns in recent years. However, some recent work demonstrated that vFL is vulnerable to privacy leakage even though only the forward intermediate embedding (rather than raw features) and backpropagated gradients (rather than raw labels) are communicated between the involved participants. As the raw labels often contain highly sensitive information, some recent work has been proposed to prevent the label leakage from the backpropagated gradients effectively in vFL. However, these work only identified and defended the threat of label leakage from the backpropagated gradients. None of these work has paid attention to the problem of label leakage from the intermediate embedding. In this paper, we propose a practical label inference method which can steal private labels effectively from the shared intermediate embedding even though some existing protection methods such as label differential privacy and gradients perturbation are applied. The effectiveness of the label attack is inseparable from the correlation between the intermediate embedding and corresponding private labels. To mitigate the issue of label leakage from the forward embedding, we add an additional optimization goal at the label party to limit the label stealing ability of the adversary by minimizing the distance correlation between the intermediate embedding and corresponding private labels. We conducted massive experiments to demonstrate the effectiveness of our proposed protection methods.



## **28. Recipe2Vec: Multi-modal Recipe Representation Learning with Graph Neural Networks**

cs.LG

Accepted by IJCAI 2022

**SubmitDate**: 2022-05-24    [paper-pdf](http://arxiv.org/pdf/2205.12396v1)

**Authors**: Yijun Tian, Chuxu Zhang, Zhichun Guo, Yihong Ma, Ronald Metoyer, Nitesh V. Chawla

**Abstracts**: Learning effective recipe representations is essential in food studies. Unlike what has been developed for image-based recipe retrieval or learning structural text embeddings, the combined effect of multi-modal information (i.e., recipe images, text, and relation data) receives less attention. In this paper, we formalize the problem of multi-modal recipe representation learning to integrate the visual, textual, and relational information into recipe embeddings. In particular, we first present Large-RG, a new recipe graph data with over half a million nodes, making it the largest recipe graph to date. We then propose Recipe2Vec, a novel graph neural network based recipe embedding model to capture multi-modal information. Additionally, we introduce an adversarial attack strategy to ensure stable learning and improve performance. Finally, we design a joint objective function of node classification and adversarial learning to optimize the model. Extensive experiments demonstrate that Recipe2Vec outperforms state-of-the-art baselines on two classic food study tasks, i.e., cuisine category classification and region prediction. Dataset and codes are available at https://github.com/meettyj/Recipe2Vec.



## **29. Label Leakage and Protection in Two-party Split Learning**

cs.LG

Accepted to ICLR 2022 (https://openreview.net/forum?id=cOtBRgsf2fO)

**SubmitDate**: 2022-05-24    [paper-pdf](http://arxiv.org/pdf/2102.08504v3)

**Authors**: Oscar Li, Jiankai Sun, Xin Yang, Weihao Gao, Hongyi Zhang, Junyuan Xie, Virginia Smith, Chong Wang

**Abstracts**: Two-party split learning is a popular technique for learning a model across feature-partitioned data. In this work, we explore whether it is possible for one party to steal the private label information from the other party during split training, and whether there are methods that can protect against such attacks. Specifically, we first formulate a realistic threat model and propose a privacy loss metric to quantify label leakage in split learning. We then show that there exist two simple yet effective methods within the threat model that can allow one party to accurately recover private ground-truth labels owned by the other party. To combat these attacks, we propose several random perturbation techniques, including $\texttt{Marvell}$, an approach that strategically finds the structure of the noise perturbation by minimizing the amount of label leakage (measured through our quantification metric) of a worst-case adversary. We empirically demonstrate the effectiveness of our protection techniques against the identified attacks, and show that $\texttt{Marvell}$ in particular has improved privacy-utility tradeoffs relative to baseline approaches.



## **30. PORTFILER: Port-Level Network Profiling for Self-Propagating Malware Detection**

cs.CR

An earlier version is accepted to be published in IEEE Conference on  Communications and Network Security (CNS) 2021

**SubmitDate**: 2022-05-24    [paper-pdf](http://arxiv.org/pdf/2112.13798v2)

**Authors**: Talha Ongun, Oliver Spohngellert, Benjamin Miller, Simona Boboila, Alina Oprea, Tina Eliassi-Rad, Jason Hiser, Alastair Nottingham, Jack Davidson, Malathi Veeraraghavan

**Abstracts**: Recent self-propagating malware (SPM) campaigns compromised hundred of thousands of victim machines on the Internet. It is challenging to detect these attacks in their early stages, as adversaries utilize common network services, use novel techniques, and can evade existing detection mechanisms. We propose PORTFILER (PORT-Level Network Traffic ProFILER), a new machine learning system applied to network traffic for detecting SPM attacks. PORTFILER extracts port-level features from the Zeek connection logs collected at a border of a monitored network, applies anomaly detection techniques to identify suspicious events, and ranks the alerts across ports for investigation by the Security Operations Center (SOC). We propose a novel ensemble methodology for aggregating individual models in PORTFILER that increases resilience against several evasion strategies compared to standard ML baselines. We extensively evaluate PORTFILER on traffic collected from two university networks, and show that it can detect SPM attacks with different patterns, such as WannaCry and Mirai, and performs well under evasion. Ranking across ports achieves precision over 0.94 with low false positive rates in the top ranked alerts. When deployed on the university networks, PORTFILER detected anomalous SPM-like activity on one of the campus networks, confirmed by the university SOC as malicious. PORTFILER also detected a Mirai attack recreated on the two university networks with higher precision and recall than deep-learning-based autoencoder methods.



## **31. Self-Supervised Contrastive Learning with Adversarial Perturbations for Defending Word Substitution-based Attacks**

cs.CL

In Findings of NAACL 2022

**SubmitDate**: 2022-05-24    [paper-pdf](http://arxiv.org/pdf/2107.07610v3)

**Authors**: Zhao Meng, Yihan Dong, Mrinmaya Sachan, Roger Wattenhofer

**Abstracts**: In this paper, we present an approach to improve the robustness of BERT language models against word substitution-based adversarial attacks by leveraging adversarial perturbations for self-supervised contrastive learning. We create a word-level adversarial attack generating hard positives on-the-fly as adversarial examples during contrastive learning. In contrast to previous works, our method improves model robustness without using any labeled data. Experimental results show that our method improves robustness of BERT against four different word substitution-based adversarial attacks, and combining our method with adversarial training gives higher robustness than adversarial training alone. As our method improves the robustness of BERT purely with unlabeled data, it opens up the possibility of using large text datasets to train robust language models against word substitution-based adversarial attacks.



## **32. Adversarial Attack on Attackers: Post-Process to Mitigate Black-Box Score-Based Query Attacks**

cs.LG

**SubmitDate**: 2022-05-24    [paper-pdf](http://arxiv.org/pdf/2205.12134v1)

**Authors**: Sizhe Chen, Zhehao Huang, Qinghua Tao, Yingwen Wu, Cihang Xie, Xiaolin Huang

**Abstracts**: The score-based query attacks (SQAs) pose practical threats to deep neural networks by crafting adversarial perturbations within dozens of queries, only using the model's output scores. Nonetheless, we note that if the loss trend of the outputs is slightly perturbed, SQAs could be easily misled and thereby become much less effective. Following this idea, we propose a novel defense, namely Adversarial Attack on Attackers (AAA), to confound SQAs towards incorrect attack directions by slightly modifying the output logits. In this way, (1) SQAs are prevented regardless of the model's worst-case robustness; (2) the original model predictions are hardly changed, i.e., no degradation on clean accuracy; (3) the calibration of confidence scores can be improved simultaneously. Extensive experiments are provided to verify the above advantages. For example, by setting $\ell_\infty=8/255$ on CIFAR-10, our proposed AAA helps WideResNet-28 secure $80.59\%$ accuracy under Square attack ($2500$ queries), while the best prior defense (i.e., adversarial training) only attains $67.44\%$. Since AAA attacks SQA's general greedy strategy, such advantages of AAA over 8 defenses can be consistently observed on 8 CIFAR-10/ImageNet models under 6 SQAs, using different attack targets and bounds. Moreover, AAA calibrates better without hurting the accuracy. Our code would be released.



## **33. Defending a Music Recommender Against Hubness-Based Adversarial Attacks**

eess.AS

6 pages, to be published in Proceedings of the 19th Sound and Music  Computing Conference 2022 (SMC-22)

**SubmitDate**: 2022-05-24    [paper-pdf](http://arxiv.org/pdf/2205.12032v1)

**Authors**: Katharina Hoedt, Arthur Flexer, Gerhard Widmer

**Abstracts**: Adversarial attacks can drastically degrade performance of recommenders and other machine learning systems, resulting in an increased demand for defence mechanisms. We present a new line of defence against attacks which exploit a vulnerability of recommenders that operate in high dimensional data spaces (the so-called hubness problem). We use a global data scaling method, namely Mutual Proximity (MP), to defend a real-world music recommender which previously was susceptible to attacks that inflated the number of times a particular song was recommended. We find that using MP as a defence greatly increases robustness of the recommender against a range of attacks, with success rates of attacks around 44% (before defence) dropping to less than 6% (after defence). Additionally, adversarial examples still able to fool the defended system do so at the price of noticeably lower audio quality as shown by a decreased average SNR.



## **34. Phrase-level Textual Adversarial Attack with Label Preservation**

cs.CL

NAACL-HLT 2022 Findings (Long), 9 pages + 2 pages references + 8  pages appendix

**SubmitDate**: 2022-05-24    [paper-pdf](http://arxiv.org/pdf/2205.10710v2)

**Authors**: Yibin Lei, Yu Cao, Dianqi Li, Tianyi Zhou, Meng Fang, Mykola Pechenizkiy

**Abstracts**: Generating high-quality textual adversarial examples is critical for investigating the pitfalls of natural language processing (NLP) models and further promoting their robustness. Existing attacks are usually realized through word-level or sentence-level perturbations, which either limit the perturbation space or sacrifice fluency and textual quality, both affecting the attack effectiveness. In this paper, we propose Phrase-Level Textual Adversarial aTtack (PLAT) that generates adversarial samples through phrase-level perturbations. PLAT first extracts the vulnerable phrases as attack targets by a syntactic parser, and then perturbs them by a pre-trained blank-infilling model. Such flexible perturbation design substantially expands the search space for more effective attacks without introducing too many modifications, and meanwhile maintaining the textual fluency and grammaticality via contextualized generation using surrounding texts. Moreover, we develop a label-preservation filter leveraging the likelihoods of language models fine-tuned on each class, rather than textual similarity, to rule out those perturbations that potentially alter the original class label for humans. Extensive experiments and human evaluation demonstrate that PLAT has a superior attack effectiveness as well as a better label consistency than strong baselines.



## **35. Can Adversarial Training Be Manipulated By Non-Robust Features?**

cs.LG

**SubmitDate**: 2022-05-24    [paper-pdf](http://arxiv.org/pdf/2201.13329v2)

**Authors**: Lue Tao, Lei Feng, Hongxin Wei, Jinfeng Yi, Sheng-Jun Huang, Songcan Chen

**Abstracts**: Adversarial training, originally designed to resist test-time adversarial examples, has shown to be promising in mitigating training-time availability attacks. This defense ability, however, is challenged in this paper. We identify a novel threat model named stability attacks, which aims to hinder robust availability by slightly manipulating the training data. Under this threat, we show that adversarial training using a conventional defense budget $\epsilon$ provably fails to provide test robustness in a simple statistical setting, where the non-robust features of the training data can be reinforced by $\epsilon$-bounded perturbation. Further, we analyze the necessity of enlarging the defense budget to counter stability attacks. Finally, comprehensive experiments demonstrate that stability attacks are harmful on benchmark datasets, and thus the adaptive defense is necessary to maintain robustness.



## **36. Smart Grid: Cyber Attacks, Critical Defense Approaches, and Digital Twin**

cs.CR

**SubmitDate**: 2022-05-24    [paper-pdf](http://arxiv.org/pdf/2205.11783v1)

**Authors**: Tianming Zheng, Ming Liu, Deepak Puthal, Ping Yi, Yue Wu, Xiangjian He

**Abstracts**: As a national critical infrastructure, the smart grid has attracted widespread attention for its cybersecurity issues. The development towards an intelligent, digital, and Internetconnected smart grid has attracted external adversaries for malicious activities. It is necessary to enhance its cybersecurity by either improving the existing defense approaches or introducing novel developed technologies to the smart grid context. As an emerging technology, digital twin (DT) is considered as an enabler for enhanced security. However, the practical implementation is quite challenging. This is due to the knowledge barriers among smart grid designers, security experts, and DT developers. Each single domain is a complicated system covering various components and technologies. As a result, works are needed to sort out relevant contents so that DT can be better embedded in the security architecture design of smart grid. In order to meet this demand, our paper covers the above three domains, i.e., smart grid, cybersecurity, and DT. Specifically, the paper i) introduces the background of the smart grid; ii) reviews external cyber attacks from attack incidents and attack methods; iii) introduces critical defense approaches in industrial cyber systems, which include device identification, vulnerability discovery, intrusion detection systems (IDSs), honeypots, attribution, and threat intelligence (TI); iv) reviews the relevant content of DT, including its basic concepts, applications in the smart grid, and how DT enhances the security. In the end, the paper puts forward our security considerations on the future development of DT-based smart grid. The survey is expected to help developers break knowledge barriers among smart grid, cybersecurity, and DT, and provide guidelines for future security design of DT-based smart grid.



## **37. Alleviating Robust Overfitting of Adversarial Training With Consistency Regularization**

cs.LG

**SubmitDate**: 2022-05-24    [paper-pdf](http://arxiv.org/pdf/2205.11744v1)

**Authors**: Shudong Zhang, Haichang Gao, Tianwei Zhang, Yunyi Zhou, Zihui Wu

**Abstracts**: Adversarial training (AT) has proven to be one of the most effective ways to defend Deep Neural Networks (DNNs) against adversarial attacks. However, the phenomenon of robust overfitting, i.e., the robustness will drop sharply at a certain stage, always exists during AT. It is of great importance to decrease this robust generalization gap in order to obtain a robust model. In this paper, we present an in-depth study towards the robust overfitting from a new angle. We observe that consistency regularization, a popular technique in semi-supervised learning, has a similar goal as AT and can be used to alleviate robust overfitting. We empirically validate this observation, and find a majority of prior solutions have implicit connections to consistency regularization. Motivated by this, we introduce a new AT solution, which integrates the consistency regularization and Mean Teacher (MT) strategy into AT. Specifically, we introduce a teacher model, coming from the average weights of the student models over the training steps. Then we design a consistency loss function to make the prediction distribution of the student models over adversarial examples consistent with that of the teacher model over clean samples. Experiments show that our proposed method can effectively alleviate robust overfitting and improve the robustness of DNN models against common adversarial attacks.



## **38. Learning to Ignore Adversarial Attacks**

cs.CL

14 pages, 2 figures

**SubmitDate**: 2022-05-23    [paper-pdf](http://arxiv.org/pdf/2205.11551v1)

**Authors**: Yiming Zhang, Yangqiaoyu Zhou, Samuel Carton, Chenhao Tan

**Abstracts**: Despite the strong performance of current NLP models, they can be brittle against adversarial attacks. To enable effective learning against adversarial inputs, we introduce the use of rationale models that can explicitly learn to ignore attack tokens. We find that the rationale models can successfully ignore over 90\% of attack tokens. This approach leads to consistent sizable improvements ($\sim$10\%) over baseline models in robustness on three datasets for both BERT and RoBERTa, and also reliably outperforms data augmentation with adversarial examples alone. In many cases, we find that our method is able to close the gap between model performance on a clean test set and an attacked test set and hence reduce the effect of adversarial attacks.



## **39. Graph Layer Security: Encrypting Information via Common Networked Physics**

eess.SP

**SubmitDate**: 2022-05-23    [paper-pdf](http://arxiv.org/pdf/2006.03568v3)

**Authors**: Zhuangkun Wei, Liang Wang, Schyler Chengyao Sun, Bin Li, Weisi Guo

**Abstracts**: The proliferation of low-cost Internet of Things (IoT) devices has led to a race between wireless security and channel attacks. Traditional cryptography requires high-computational power and is not suitable for low-power IoT scenarios. Whist, recently developed physical layer security (PLS) can exploit common wireless channel state information (CSI), its sensitivity to channel estimation makes them vulnerable from attacks. In this work, we exploit an alternative common physics shared between IoT transceivers: the monitored channel-irrelevant physical networked dynamics (e.g., water/oil/gas/electrical signal-flows). Leveraging this, we propose for the first time, graph layer security (GLS), by exploiting the dependency in physical dynamics among network nodes for information encryption and decryption. A graph Fourier transform (GFT) operator is used to characterize such dependency into a graph-bandlimted subspace, which allows the generations of channel-irrelevant cipher keys by maximizing the secrecy rate. We evaluate our GLS against designed active and passive attackers, using IEEE 39-Bus system. Results demonstrate that, GLS is not reliant on wireless CSI, and can combat attackers that have partial networked dynamic knowledge (realistic access to full dynamic and critical nodes remains challenging). We believe this novel GLS has widespread applicability in secure health monitoring and for Digital Twins in adversarial radio environments.



## **40. Detection of Stealthy Adversaries for Networked Unmanned Aerial Vehicles***

eess.SY

to appear at the 2022 Int'l Conference on Unmanned Aircraft Systems  (ICUAS)

**SubmitDate**: 2022-05-22    [paper-pdf](http://arxiv.org/pdf/2202.09661v2)

**Authors**: Mohammad Bahrami, Hamidreza Jafarnejadsani

**Abstracts**: A network of unmanned aerial vehicles (UAVs) provides distributed coverage, reconfigurability, and maneuverability in performing complex cooperative tasks. However, it relies on wireless communications that can be susceptible to cyber adversaries and intrusions, disrupting the entire network's operation. This paper develops model-based centralized and decentralized observer techniques for detecting a class of stealthy intrusions, namely zero-dynamics and covert attacks, on networked UAVs in formation control settings. The centralized observer that runs in a control center leverages switching in the UAVs' communication topology for attack detection, and the decentralized observers, implemented onboard each UAV in the network, use the model of networked UAVs and locally available measurements. Experimental results are provided to show the effectiveness of the proposed detection schemes in different case studies.



## **41. Inverse-Inverse Reinforcement Learning. How to Hide Strategy from an Adversarial Inverse Reinforcement Learner**

cs.LG

**SubmitDate**: 2022-05-22    [paper-pdf](http://arxiv.org/pdf/2205.10802v1)

**Authors**: Kunal Pattanayak, Vikram Krishnamurthy, Christopher Berry

**Abstracts**: Inverse reinforcement learning (IRL) deals with estimating an agent's utility function from its actions. In this paper, we consider how an agent can hide its strategy and mitigate an adversarial IRL attack; we call this inverse IRL (I-IRL). How should the decision maker choose its response to ensure a poor reconstruction of its strategy by an adversary performing IRL to estimate the agent's strategy? This paper comprises four results: First, we present an adversarial IRL algorithm that estimates the agent's strategy while controlling the agent's utility function. Our second result for I-IRL result spoofs the IRL algorithm used by the adversary. Our I-IRL results are based on revealed preference theory in micro-economics. The key idea is for the agent to deliberately choose sub-optimal responses that sufficiently masks its true strategy. Third, we give a sample complexity result for our main I-IRL result when the agent has noisy estimates of the adversary specified utility function. Finally, we illustrate our I-IRL scheme in a radar problem where a meta-cognitive radar is trying to mitigate an adversarial target.



## **42. Post-breach Recovery: Protection against White-box Adversarial Examples for Leaked DNN Models**

cs.CR

**SubmitDate**: 2022-05-21    [paper-pdf](http://arxiv.org/pdf/2205.10686v1)

**Authors**: Shawn Shan, Wenxin Ding, Emily Wenger, Haitao Zheng, Ben Y. Zhao

**Abstracts**: Server breaches are an unfortunate reality on today's Internet. In the context of deep neural network (DNN) models, they are particularly harmful, because a leaked model gives an attacker "white-box" access to generate adversarial examples, a threat model that has no practical robust defenses. For practitioners who have invested years and millions into proprietary DNNs, e.g. medical imaging, this seems like an inevitable disaster looming on the horizon.   In this paper, we consider the problem of post-breach recovery for DNN models. We propose Neo, a new system that creates new versions of leaked models, alongside an inference time filter that detects and removes adversarial examples generated on previously leaked models. The classification surfaces of different model versions are slightly offset (by introducing hidden distributions), and Neo detects the overfitting of attacks to the leaked model used in its generation. We show that across a variety of tasks and attack methods, Neo is able to filter out attacks from leaked models with very high accuracy, and provides strong protection (7--10 recoveries) against attackers who repeatedly breach the server. Neo performs well against a variety of strong adaptive attacks, dropping slightly in # of breaches recoverable, and demonstrates potential as a complement to DNN defenses in the wild.



## **43. Gradient Concealment: Free Lunch for Defending Adversarial Attacks**

cs.CV

**SubmitDate**: 2022-05-21    [paper-pdf](http://arxiv.org/pdf/2205.10617v1)

**Authors**: Sen Pei, Jiaxi Sun, Xiaopeng Zhang, Gaofeng Meng

**Abstracts**: Recent studies show that the deep neural networks (DNNs) have achieved great success in various tasks. However, even the \emph{state-of-the-art} deep learning based classifiers are extremely vulnerable to adversarial examples, resulting in sharp decay of discrimination accuracy in the presence of enormous unknown attacks. Given the fact that neural networks are widely used in the open world scenario which can be safety-critical situations, mitigating the adversarial effects of deep learning methods has become an urgent need. Generally, conventional DNNs can be attacked with a dramatically high success rate since their gradient is exposed thoroughly in the white-box scenario, making it effortless to ruin a well trained classifier with only imperceptible perturbations in the raw data space. For tackling this problem, we propose a plug-and-play layer that is training-free, termed as \textbf{G}radient \textbf{C}oncealment \textbf{M}odule (GCM), concealing the vulnerable direction of gradient while guaranteeing the classification accuracy during the inference time. GCM reports superior defense results on the ImageNet classification benchmark, improving up to 63.41\% top-1 attack robustness (AR) when faced with adversarial inputs compared to the vanilla DNNs. Moreover, we use GCM in the CVPR 2022 Robust Classification Challenge, currently achieving \textbf{2nd} place in Phase II with only a tiny version of ConvNext. The code will be made available.



## **44. SERVFAIL: The Unintended Consequences of Algorithm Agility in DNSSEC**

cs.CR

**SubmitDate**: 2022-05-21    [paper-pdf](http://arxiv.org/pdf/2205.10608v1)

**Authors**: Elias Heftrig, Jean-Pierre Seifert, Haya Shulman, Peter Thomassen, Michael Waidner, Nils Wisiol

**Abstracts**: Cryptographic algorithm agility is an important property for DNSSEC: it allows easy deployment of new algorithms if the existing ones are no longer secure. Significant operational and research efforts are dedicated to pushing the deployment of new algorithms in DNSSEC forward. Recent research shows that DNSSEC is gradually achieving algorithm agility: most DNSSEC supporting resolvers can validate a number of different algorithms and domains are increasingly signed with cryptographically strong ciphers.   In this work we show for the first time that the cryptographic agility in DNSSEC, although critical for making DNS secure with strong cryptography, also introduces a severe vulnerability. We find that under certain conditions, when new algorithms are listed in signed DNS responses, the resolvers do not validate DNSSEC. As a result, domains that deploy new ciphers, risk exposing the validating resolvers to cache poisoning attacks.   We use this to develop DNSSEC-downgrade attacks and show that in some situations these attacks can be launched even by off-path adversaries. We experimentally and ethically evaluate our attacks against popular DNS resolver implementations, public DNS providers, and DNS services used by web clients worldwide. We validate the success of DNSSEC-downgrade attacks by poisoning the resolvers: we inject fake records, in signed domains, into the caches of validating resolvers. We find that major DNS providers, such as Google Public DNS and Cloudflare, as well as 70% of DNS resolvers used by web clients are vulnerable to our attacks.   We trace the factors that led to this situation and provide recommendations.



## **45. On the Feasibility and Generality of Patch-based Adversarial Attacks on Semantic Segmentation Problems**

cs.CV

**SubmitDate**: 2022-05-21    [paper-pdf](http://arxiv.org/pdf/2205.10539v1)

**Authors**: Soma Kontar, Andras Horvath

**Abstracts**: Deep neural networks were applied with success in a myriad of applications, but in safety critical use cases adversarial attacks still pose a significant threat. These attacks were demonstrated on various classification and detection tasks and are usually considered general in a sense that arbitrary network outputs can be generated by them.   In this paper we will demonstrate through simple case studies both in simulation and in real-life, that patch based attacks can be utilised to alter the output of segmentation networks. Through a few examples and the investigation of network complexity, we will also demonstrate that the number of possible output maps which can be generated via patch-based attacks of a given size is typically smaller than the area they effect or areas which should be attacked in case of practical applications.   We will prove that based on these results most patch-based attacks cannot be general in practice, namely they can not generate arbitrary output maps or if they could, they are spatially limited and this limit is significantly smaller than the receptive field of the patches.



## **46. PRADA: Practical Black-Box Adversarial Attacks against Neural Ranking Models**

cs.IR

**SubmitDate**: 2022-05-21    [paper-pdf](http://arxiv.org/pdf/2204.01321v2)

**Authors**: Chen Wu, Ruqing Zhang, Jiafeng Guo, Maarten de Rijke, Yixing Fan, Xueqi Cheng

**Abstracts**: Neural ranking models (NRMs) have shown remarkable success in recent years, especially with pre-trained language models. However, deep neural models are notorious for their vulnerability to adversarial examples. Adversarial attacks may become a new type of web spamming technique given our increased reliance on neural information retrieval models. Therefore, it is important to study potential adversarial attacks to identify vulnerabilities of NRMs before they are deployed.   In this paper, we introduce the Word Substitution Ranking Attack (WSRA) task against NRMs, which aims to promote a target document in rankings by adding adversarial perturbations to its text. We focus on the decision-based black-box attack setting, where the attackers have no access to the model parameters and gradients, but can only acquire the rank positions of the partial retrieved list by querying the target model. This attack setting is realistic in real-world search engines. We propose a novel Pseudo Relevance-based ADversarial ranking Attack method (PRADA) that learns a surrogate model based on Pseudo Relevance Feedback (PRF) to generate gradients for finding the adversarial perturbations.   Experiments on two web search benchmark datasets show that PRADA can outperform existing attack strategies and successfully fool the NRM with small indiscernible perturbations of text.



## **47. Robust Sensible Adversarial Learning of Deep Neural Networks for Image Classification**

cs.CR

**SubmitDate**: 2022-05-20    [paper-pdf](http://arxiv.org/pdf/2205.10457v1)

**Authors**: Jungeum Kim, Xiao Wang

**Abstracts**: The idea of robustness is central and critical to modern statistical analysis. However, despite the recent advances of deep neural networks (DNNs), many studies have shown that DNNs are vulnerable to adversarial attacks. Making imperceptible changes to an image can cause DNN models to make the wrong classification with high confidence, such as classifying a benign mole as a malignant tumor and a stop sign as a speed limit sign. The trade-off between robustness and standard accuracy is common for DNN models. In this paper, we introduce sensible adversarial learning and demonstrate the synergistic effect between pursuits of standard natural accuracy and robustness. Specifically, we define a sensible adversary which is useful for learning a robust model while keeping high natural accuracy. We theoretically establish that the Bayes classifier is the most robust multi-class classifier with the 0-1 loss under sensible adversarial learning. We propose a novel and efficient algorithm that trains a robust model using implicit loss truncation. We apply sensible adversarial learning for large-scale image classification to a handwritten digital image dataset called MNIST and an object recognition colored image dataset called CIFAR10. We have performed an extensive comparative study to compare our method with other competitive methods. Our experiments empirically demonstrate that our method is not sensitive to its hyperparameter and does not collapse even with a small model capacity while promoting robustness against various attacks and keeping high natural accuracy.



## **48. Vulnerability Analysis and Performance Enhancement of Authentication Protocol in Dynamic Wireless Power Transfer Systems**

cs.CR

16 pages, conference

**SubmitDate**: 2022-05-20    [paper-pdf](http://arxiv.org/pdf/2205.10292v1)

**Authors**: Tommaso Bianchi, Surudhi Asokraj, Alessandro Brighente, Mauro Conti, Radha Poovendran

**Abstracts**: Recent advancements in wireless charging technology, as well as the possibility of utilizing it in the Electric Vehicle (EV) domain for dynamic charging solutions, have fueled the demand for a secure and usable protocol in the Dynamic Wireless Power Transfer (DWPT) technology. The DWPT must operate in the presence of malicious adversaries that can undermine the charging process and harm the customer service quality, while preserving the privacy of the users. Recently, it was shown that the DWPT system is susceptible to adversarial attacks, including replay, denial-of-service and free-riding attacks, which can lead to the adversary blocking the authorized user from charging, enabling free charging for free riders and exploiting the location privacy of the customers. In this paper, we study the current State-Of-The-Art (SOTA) authentication protocols and make the following two contributions: a) we show that the SOTA is vulnerable to the tracking of the user activity and b) we propose an enhanced authentication protocol that eliminates the vulnerability while providing improved efficiency compared to the SOTA authentication protocols. By adopting authentication messages based only on exclusive OR operations, hashing, and hash chains, we optimize the protocol to achieve a complexity that varies linearly with the number of charging pads, providing improved scalability. Compared to SOTA, the proposed scheme has a performance gain in the computational cost of around 90% on average for each pad.



## **49. Adversarial Body Shape Search for Legged Robots**

cs.RO

6 pages, 7 figures

**SubmitDate**: 2022-05-20    [paper-pdf](http://arxiv.org/pdf/2205.10187v1)

**Authors**: Takaaki Azakami, Hiroshi Kera, Kazuhiko Kawamoto

**Abstracts**: We propose an evolutionary computation method for an adversarial attack on the length and thickness of parts of legged robots by deep reinforcement learning. This attack changes the robot body shape and interferes with walking-we call the attacked body as adversarial body shape. The evolutionary computation method searches adversarial body shape by minimizing the expected cumulative reward earned through walking simulation. To evaluate the effectiveness of the proposed method, we perform experiments with three-legged robots, Walker2d, Ant-v2, and Humanoid-v2 in OpenAI Gym. The experimental results reveal that Walker2d and Ant-v2 are more vulnerable to the attack on the length than the thickness of the body parts, whereas Humanoid-v2 is vulnerable to the attack on both of the length and thickness. We further identify that the adversarial body shapes break left-right symmetry or shift the center of gravity of the legged robots. Finding adversarial body shape can be used to proactively diagnose the vulnerability of legged robot walking.



## **50. Getting a-Round Guarantees: Floating-Point Attacks on Certified Robustness**

cs.CR

**SubmitDate**: 2022-05-20    [paper-pdf](http://arxiv.org/pdf/2205.10159v1)

**Authors**: Jiankai Jin, Olga Ohrimenko, Benjamin I. P. Rubinstein

**Abstracts**: Adversarial examples pose a security risk as they can alter a classifier's decision through slight perturbations to a benign input. Certified robustness has been proposed as a mitigation strategy where given an input $x$, a classifier returns a prediction and a radius with a provable guarantee that any perturbation to $x$ within this radius (e.g., under the $L_2$ norm) will not alter the classifier's prediction. In this work, we show that these guarantees can be invalidated due to limitations of floating-point representation that cause rounding errors. We design a rounding search method that can efficiently exploit this vulnerability to find adversarial examples within the certified radius. We show that the attack can be carried out against several linear classifiers that have exact certifiable guarantees and against neural network verifiers that return a certified lower bound on a robust radius. Our experiments demonstrate over 50% attack success rate on random linear classifiers, up to 35% on a breast cancer dataset for logistic regression, and a 9% attack success rate on the MNIST dataset for a neural network whose certified radius was verified by a prominent bound propagation method. We also show that state-of-the-art random smoothed classifiers for neural networks are also susceptible to adversarial examples (e.g., up to 2% attack rate on CIFAR10)-validating the importance of accounting for the error rate of robustness guarantees of such classifiers in practice. Finally, as a mitigation, we advocate the use of rounded interval arithmetic to account for rounding errors.



