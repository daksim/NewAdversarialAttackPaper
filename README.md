# Latest Adversarial Attack Papers
**update at 2022-06-16 06:31:33**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. A Layered Reference Model for Penetration Testing with Reinforcement Learning and Attack Graphs**

cs.CR

**SubmitDate**: 2022-06-14    [paper-pdf](http://arxiv.org/pdf/2206.06934v1)

**Authors**: Tyler Cody

**Abstracts**: This paper considers key challenges to using reinforcement learning (RL) with attack graphs to automate penetration testing in real-world applications from a systems perspective. RL approaches to automated penetration testing are actively being developed, but there is no consensus view on the representation of computer networks with which RL should be interacting. Moreover, there are significant open challenges to how those representations can be grounded to the real networks where RL solution methods are applied. This paper elaborates on representation and grounding using topic challenges of interacting with real networks in real-time, emulating realistic adversary behavior, and handling unstable, evolving networks. These challenges are both practical and mathematical, and they directly concern the reliability and dependability of penetration testing systems. This paper proposes a layered reference model to help organize related research and engineering efforts. The presented layered reference model contrasts traditional models of attack graph workflows because it is not scoped to a sequential, feed-forward generation and analysis process, but to broader aspects of lifecycle and continuous deployment. Researchers and practitioners can use the presented layered reference model as a first-principles outline to help orient the systems engineering of their penetration testing systems.



## **2. Creating a Secure Underlay for the Internet**

cs.NI

Usenix Security 2022

**SubmitDate**: 2022-06-14    [paper-pdf](http://arxiv.org/pdf/2206.06879v1)

**Authors**: Henry Birge-Lee, Joel Wanner, Grace Cimaszewski, Jonghoon Kwon, Liang Wang, Francois Wirz, Prateek Mittal, Adrian Perrig, Yixin Sun

**Abstracts**: Adversaries can exploit inter-domain routing vulnerabilities to intercept communication and compromise the security of critical Internet applications. Meanwhile the deployment of secure routing solutions such as Border Gateway Protocol Security (BGPsec) and Scalability, Control and Isolation On Next-generation networks (SCION) are still limited. How can we leverage emerging secure routing backbones and extend their security properties to the broader Internet?   We design and deploy an architecture to bootstrap secure routing. Our key insight is to abstract the secure routing backbone as a virtual Autonomous System (AS), called Secure Backbone AS (SBAS). While SBAS appears as one AS to the Internet, it is a federated network where routes are exchanged between participants using a secure backbone. SBAS makes BGP announcements for its customers' IP prefixes at multiple locations (referred to as Points of Presence or PoPs) allowing traffic from non-participating hosts to be routed to a nearby SBAS PoP (where it is then routed over the secure backbone to the true prefix owner). In this manner, we are the first to integrate a federated secure non-BGP routing backbone with the BGP-speaking Internet.   We present a real-world deployment of our architecture that uses SCIONLab to emulate the secure backbone and the PEERING framework to make BGP announcements to the Internet. A combination of real-world attacks and Internet-scale simulations shows that SBAS substantially reduces the threat of routing attacks. Finally, we survey network operators to better understand optimal governance and incentive models.



## **3. When adversarial attacks become interpretable counterfactual explanations**

cs.AI

**SubmitDate**: 2022-06-14    [paper-pdf](http://arxiv.org/pdf/2206.06854v1)

**Authors**: Mathieu Serrurier, Franck Mamalet, Thomas Fel, Louis Béthune, Thibaut Boissin

**Abstracts**: We argue that, when learning a 1-Lipschitz neural network with the dual loss of an optimal transportation problem, the gradient of the model is both the direction of the transportation plan and the direction to the closest adversarial attack. Traveling along the gradient to the decision boundary is no more an adversarial attack but becomes a counterfactual explanation, explicitly transporting from one class to the other. Through extensive experiments on XAI metrics, we find that the simple saliency map method, applied on such networks, becomes a reliable explanation, and outperforms the state-of-the-art explanation approaches on unconstrained models. The proposed networks were already known to be certifiably robust, and we prove that they are also explainable with a fast and simple method.



## **4. Exploring Adversarial Attacks and Defenses in Vision Transformers trained with DINO**

cs.CV

6 pages workshop paper accepted at AdvML Frontiers (ICML 2022)

**SubmitDate**: 2022-06-14    [paper-pdf](http://arxiv.org/pdf/2206.06761v1)

**Authors**: Javier Rando, Nasib Naimi, Thomas Baumann, Max Mathys

**Abstracts**: This work conducts the first analysis on the robustness against adversarial attacks on self-supervised Vision Transformers trained using DINO. First, we evaluate whether features learned through self-supervision are more robust to adversarial attacks than those emerging from supervised learning. Then, we present properties arising for attacks in the latent space. Finally, we evaluate whether three well-known defense strategies can increase adversarial robustness in downstream tasks by only fine-tuning the classification head to provide robustness even in view of limited compute resources. These defense strategies are: Adversarial Training, Ensemble Adversarial Training and Ensemble of Specialized Networks.



## **5. Adversarial Vulnerability of Randomized Ensembles**

cs.LG

Published as a conference paper in ICML 2022

**SubmitDate**: 2022-06-14    [paper-pdf](http://arxiv.org/pdf/2206.06737v1)

**Authors**: Hassan Dbouk, Naresh R. Shanbhag

**Abstracts**: Despite the tremendous success of deep neural networks across various tasks, their vulnerability to imperceptible adversarial perturbations has hindered their deployment in the real world. Recently, works on randomized ensembles have empirically demonstrated significant improvements in adversarial robustness over standard adversarially trained (AT) models with minimal computational overhead, making them a promising solution for safety-critical resource-constrained applications. However, this impressive performance raises the question: Are these robustness gains provided by randomized ensembles real? In this work we address this question both theoretically and empirically. We first establish theoretically that commonly employed robustness evaluation methods such as adaptive PGD provide a false sense of security in this setting. Subsequently, we propose a theoretically-sound and efficient adversarial attack algorithm (ARC) capable of compromising random ensembles even in cases where adaptive PGD fails to do so. We conduct comprehensive experiments across a variety of network architectures, training schemes, datasets, and norms to support our claims, and empirically establish that randomized ensembles are in fact more vulnerable to $\ell_p$-bounded adversarial perturbations than even standard AT models. Our code can be found at https://github.com/hsndbk4/ARC.



## **6. Downlink Power Allocation in Massive MIMO via Deep Learning: Adversarial Attacks and Training**

cs.LG

13 pages, 14 figures, published in IEEE Transactions on Cognitive  Communications and Networking

**SubmitDate**: 2022-06-14    [paper-pdf](http://arxiv.org/pdf/2206.06592v1)

**Authors**: B. R. Manoj, Meysam Sadeghi, Erik G. Larsson

**Abstracts**: The successful emergence of deep learning (DL) in wireless system applications has raised concerns about new security-related challenges. One such security challenge is adversarial attacks. Although there has been much work demonstrating the susceptibility of DL-based classification tasks to adversarial attacks, regression-based problems in the context of a wireless system have not been studied so far from an attack perspective. The aim of this paper is twofold: (i) we consider a regression problem in a wireless setting and show that adversarial attacks can break the DL-based approach and (ii) we analyze the effectiveness of adversarial training as a defensive technique in adversarial settings and show that the robustness of DL-based wireless system against attacks improves significantly. Specifically, the wireless application considered in this paper is the DL-based power allocation in the downlink of a multicell massive multi-input-multi-output system, where the goal of the attack is to yield an infeasible solution by the DL model. We extend the gradient-based adversarial attacks: fast gradient sign method (FGSM), momentum iterative FGSM, and projected gradient descent method to analyze the susceptibility of the considered wireless application with and without adversarial training. We analyze the deep neural network (DNN) models performance against these attacks, where the adversarial perturbations are crafted using both the white-box and black-box attacks.



## **7. Person Re-identification Method Based on Color Attack and Joint Defence**

cs.CV

Accepted by CVPR2022 Workshops  (https://openaccess.thecvf.com/content/CVPR2022W/HCIS/html/Gong_Person_Re-Identification_Method_Based_on_Color_Attack_and_Joint_Defence_CVPRW_2022_paper.html)

**SubmitDate**: 2022-06-14    [paper-pdf](http://arxiv.org/pdf/2111.09571v4)

**Authors**: Yunpeng Gong, Liqing Huang, Lifei Chen

**Abstracts**: The main challenges of ReID is the intra-class variations caused by color deviation under different camera conditions. Simultaneously, we find that most of the existing adversarial metric attacks are realized by interfering with the color characteristics of the sample. Based on this observation, we first propose a local transformation attack (LTA) based on color variation. It uses more obvious color variation to randomly disturb the color of the retrieved image, rather than adding random noise. Experiments show that the performance of the proposed LTA method is better than the advanced attack methods. Furthermore, considering that the contour feature is the main factor of the robustness of adversarial training, and the color feature will directly affect the success rate of attack. Therefore, we further propose joint adversarial defense (JAD) method, which include proactive defense and passive defense. Proactive defense fuse multi-modality images to enhance the contour feature and color feature, and considers local homomorphic transformation to solve the over-fitting problem. Passive defense exploits the invariance of contour feature during image scaling to mitigate the adversarial disturbance on contour feature. Finally, a series of experimental results show that the proposed joint adversarial defense method is more competitive than a state-of-the-art methods.



## **8. Towards Alternative Techniques for Improving Adversarial Robustness: Analysis of Adversarial Training at a Spectrum of Perturbations**

cs.LG

**SubmitDate**: 2022-06-13    [paper-pdf](http://arxiv.org/pdf/2206.06496v1)

**Authors**: Kaustubh Sridhar, Souradeep Dutta, Ramneet Kaur, James Weimer, Oleg Sokolsky, Insup Lee

**Abstracts**: Adversarial training (AT) and its variants have spearheaded progress in improving neural network robustness to adversarial perturbations and common corruptions in the last few years. Algorithm design of AT and its variants are focused on training models at a specified perturbation strength $\epsilon$ and only using the feedback from the performance of that $\epsilon$-robust model to improve the algorithm. In this work, we focus on models, trained on a spectrum of $\epsilon$ values. We analyze three perspectives: model performance, intermediate feature precision and convolution filter sensitivity. In each, we identify alternative improvements to AT that otherwise wouldn't have been apparent at a single $\epsilon$. Specifically, we find that for a PGD attack at some strength $\delta$, there is an AT model at some slightly larger strength $\epsilon$, but no greater, that generalizes best to it. Hence, we propose overdesigning for robustness where we suggest training models at an $\epsilon$ just above $\delta$. Second, we observe (across various $\epsilon$ values) that robustness is highly sensitive to the precision of intermediate features and particularly those after the first and second layer. Thus, we propose adding a simple quantization to defenses that improves accuracy on seen and unseen adaptive attacks. Third, we analyze convolution filters of each layer of models at increasing $\epsilon$ and notice that those of the first and second layer may be solely responsible for amplifying input perturbations. We present our findings and demonstrate our techniques through experiments with ResNet and WideResNet models on the CIFAR-10 and CIFAR-10-C datasets.



## **9. Distributed Adversarial Training to Robustify Deep Neural Networks at Scale**

cs.LG

**SubmitDate**: 2022-06-13    [paper-pdf](http://arxiv.org/pdf/2206.06257v1)

**Authors**: Gaoyuan Zhang, Songtao Lu, Yihua Zhang, Xiangyi Chen, Pin-Yu Chen, Quanfu Fan, Lee Martie, Lior Horesh, Mingyi Hong, Sijia Liu

**Abstracts**: Current deep neural networks (DNNs) are vulnerable to adversarial attacks, where adversarial perturbations to the inputs can change or manipulate classification. To defend against such attacks, an effective and popular approach, known as adversarial training (AT), has been shown to mitigate the negative impact of adversarial attacks by virtue of a min-max robust training method. While effective, it remains unclear whether it can successfully be adapted to the distributed learning context. The power of distributed optimization over multiple machines enables us to scale up robust training over large models and datasets. Spurred by that, we propose distributed adversarial training (DAT), a large-batch adversarial training framework implemented over multiple machines. We show that DAT is general, which supports training over labeled and unlabeled data, multiple types of attack generation methods, and gradient compression operations favored for distributed optimization. Theoretically, we provide, under standard conditions in the optimization theory, the convergence rate of DAT to the first-order stationary points in general non-convex settings. Empirically, we demonstrate that DAT either matches or outperforms state-of-the-art robust accuracies and achieves a graceful training speedup (e.g., on ResNet-50 under ImageNet). Codes are available at https://github.com/dat-2022/dat.



## **10. Adversarial Models Towards Data Availability and Integrity of Distributed State Estimation for Industrial IoT-Based Smart Grid**

cs.CR

11 pages (DC), Journal manuscript

**SubmitDate**: 2022-06-13    [paper-pdf](http://arxiv.org/pdf/2206.06027v1)

**Authors**: Haftu Tasew Reda, Abdun Mahmood, Adnan Anwar, Naveen Chilamkurti

**Abstracts**: Security issue of distributed state estimation (DSE) is an important prospect for the rapidly growing smart grid ecosystem. Any coordinated cyberattack targeting the distributed system of state estimators can cause unrestrained estimation errors and can lead to a myriad of security risks, including failure of power system operation. This article explores the security threats of a smart grid arising from the exploitation of DSE vulnerabilities. To this aim, novel adversarial strategies based on two-stage data availability and integrity attacks are proposed towards a distributed industrial Internet of Things-based smart grid. The former's attack goal is to prevent boundary data exchange among distributed control centers, while the latter's attack goal is to inject a falsified data to cause local and global system unobservability. The proposed framework is evaluated on IEEE standard 14-bus system and benchmarked against the state-of-the-art research. Experimental results show that the proposed two-stage cyberattack results in an estimated error of approximately 34.74% compared to an error of the order of 10^-3 under normal operating conditions.



## **11. Universal, transferable and targeted adversarial attacks**

cs.LG

**SubmitDate**: 2022-06-13    [paper-pdf](http://arxiv.org/pdf/1908.11332v4)

**Authors**: Junde Wu, Rao Fu

**Abstracts**: Deep Neural Networks have been found vulnerable re-cently. A kind of well-designed inputs, which called adver-sarial examples, can lead the networks to make incorrectpredictions. Depending on the different scenarios, goalsand capabilities, the difficulties of the attacks are different.For example, a targeted attack is more difficult than a non-targeted attack, a universal attack is more difficult than anon-universal attack, a transferable attack is more difficultthan a nontransferable one. The question is: Is there existan attack that can meet all these requirements? In this pa-per, we answer this question by producing a kind of attacksunder these conditions. We learn a universal mapping tomap the sources to the adversarial examples. These exam-ples can fool classification networks to classify all of theminto one targeted class, and also have strong transferability.Our code is released at: xxxxx.



## **12. Revisiting and Advancing Fast Adversarial Training Through The Lens of Bi-Level Optimization**

cs.LG

**SubmitDate**: 2022-06-13    [paper-pdf](http://arxiv.org/pdf/2112.12376v5)

**Authors**: Yihua Zhang, Guanhua Zhang, Prashant Khanduri, Mingyi Hong, Shiyu Chang, Sijia Liu

**Abstracts**: Adversarial training (AT) is a widely recognized defense mechanism to gain the robustness of deep neural networks against adversarial attacks. It is built on min-max optimization (MMO), where the minimizer (i.e., defender) seeks a robust model to minimize the worst-case training loss in the presence of adversarial examples crafted by the maximizer (i.e., attacker). However, the conventional MMO method makes AT hard to scale. Thus, Fast-AT (Wong et al., 2020) and other recent algorithms attempt to simplify MMO by replacing its maximization step with the single gradient sign-based attack generation step. Although easy to implement, Fast-AT lacks theoretical guarantees, and its empirical performance is unsatisfactory due to the issue of robust catastrophic overfitting when training with strong adversaries. In this paper, we advance Fast-AT from the fresh perspective of bi-level optimization (BLO). We first show that the commonly-used Fast-AT is equivalent to using a stochastic gradient algorithm to solve a linearized BLO problem involving a sign operation. However, the discrete nature of the sign operation makes it difficult to understand the algorithm performance. Inspired by BLO, we design and analyze a new set of robust training algorithms termed Fast Bi-level AT (Fast-BAT), which effectively defends sign-based projected gradient descent (PGD) attacks without using any gradient sign method or explicit robust regularization. In practice, we show our method yields substantial robustness improvements over baselines across multiple models and datasets. Codes are available at https://github.com/OPTML-Group/Fast-BAT.



## **13. Deploying Convolutional Networks on Untrusted Platforms Using 2D Holographic Reduced Representations**

cs.LG

To appear in the Proceedings of the 39 th International Conference on  Machine Learning, Baltimore, Maryland, USA, PMLR 162, 2022

**SubmitDate**: 2022-06-13    [paper-pdf](http://arxiv.org/pdf/2206.05893v1)

**Authors**: Mohammad Mahmudul Alam, Edward Raff, Tim Oates, James Holt

**Abstracts**: Due to the computational cost of running inference for a neural network, the need to deploy the inferential steps on a third party's compute environment or hardware is common. If the third party is not fully trusted, it is desirable to obfuscate the nature of the inputs and outputs, so that the third party can not easily determine what specific task is being performed. Provably secure protocols for leveraging an untrusted party exist but are too computational demanding to run in practice. We instead explore a different strategy of fast, heuristic security that we call Connectionist Symbolic Pseudo Secrets. By leveraging Holographic Reduced Representations (HRR), we create a neural network with a pseudo-encryption style defense that empirically shows robustness to attack, even under threat models that unrealistically favor the adversary.



## **14. InBiaseD: Inductive Bias Distillation to Improve Generalization and Robustness through Shape-awareness**

cs.CV

Accepted at 1st Conference on Lifelong Learning Agents (CoLLAs 2022)

**SubmitDate**: 2022-06-12    [paper-pdf](http://arxiv.org/pdf/2206.05846v1)

**Authors**: Shruthi Gowda, Bahram Zonooz, Elahe Arani

**Abstracts**: Humans rely less on spurious correlations and trivial cues, such as texture, compared to deep neural networks which lead to better generalization and robustness. It can be attributed to the prior knowledge or the high-level cognitive inductive bias present in the brain. Therefore, introducing meaningful inductive bias to neural networks can help learn more generic and high-level representations and alleviate some of the shortcomings. We propose InBiaseD to distill inductive bias and bring shape-awareness to the neural networks. Our method includes a bias alignment objective that enforces the networks to learn more generic representations that are less vulnerable to unintended cues in the data which results in improved generalization performance. InBiaseD is less susceptible to shortcut learning and also exhibits lower texture bias. The better representations also aid in improving robustness to adversarial attacks and we hence plugin InBiaseD seamlessly into the existing adversarial training schemes to show a better trade-off between generalization and robustness.



## **15. Consistent Attack: Universal Adversarial Perturbation on Embodied Vision Navigation**

cs.LG

**SubmitDate**: 2022-06-12    [paper-pdf](http://arxiv.org/pdf/2206.05751v1)

**Authors**: You Qiaoben, Chengyang Ying, Xinning Zhou, Hang Su, Jun Zhu, Bo Zhang

**Abstracts**: Embodied agents in vision navigation coupled with deep neural networks have attracted increasing attention. However, deep neural networks are vulnerable to malicious adversarial noises, which may potentially cause catastrophic failures in Embodied Vision Navigation. Among these adversarial noises, universal adversarial perturbations (UAP), i.e., the image-agnostic perturbation applied on each frame received by the agent, are more critical for Embodied Vision Navigation since they are computation-efficient and application-practical during the attack. However, existing UAP methods do not consider the system dynamics of Embodied Vision Navigation. For extending UAP in the sequential decision setting, we formulate the disturbed environment under the universal noise $\delta$, as a $\delta$-disturbed Markov Decision Process ($\delta$-MDP). Based on the formulation, we analyze the properties of $\delta$-MDP and propose two novel Consistent Attack methods for attacking Embodied agents, which first consider the dynamic of the MDP by estimating the disturbed Q function and the disturbed distribution. In spite of victim models, our Consistent Attack can cause a significant drop in the performance for the Goalpoint task in habitat. Extensive experimental results indicate that there exist potential risks for applying Embodied Vision Navigation methods to the real world.



## **16. Darknet Traffic Classification and Adversarial Attacks**

cs.LG

**SubmitDate**: 2022-06-12    [paper-pdf](http://arxiv.org/pdf/2206.06371v1)

**Authors**: Nhien Rust-Nguyen, Mark Stamp

**Abstracts**: The anonymous nature of darknets is commonly exploited for illegal activities. Previous research has employed machine learning and deep learning techniques to automate the detection of darknet traffic in an attempt to block these criminal activities. This research aims to improve darknet traffic detection by assessing Support Vector Machines (SVM), Random Forest (RF), Convolutional Neural Networks (CNN), and Auxiliary-Classifier Generative Adversarial Networks (AC-GAN) for classification of such traffic and the underlying application types. We find that our RF model outperforms the state-of-the-art machine learning techniques used in prior work with the CIC-Darknet2020 dataset. To evaluate the robustness of our RF classifier, we obfuscate select application type classes to simulate realistic adversarial attack scenarios. We demonstrate that our best-performing classifier can be defeated by such attacks, and we consider ways to deal with such adversarial attacks.



## **17. Security of Machine Learning-Based Anomaly Detection in Cyber Physical Systems**

cs.DC

**SubmitDate**: 2022-06-12    [paper-pdf](http://arxiv.org/pdf/2206.05678v1)

**Authors**: Zahra Jadidi, Shantanu Pal, Nithesh Nayak K, Arawinkumaar Selvakkumar, Chih-Chia Chang, Maedeh Beheshti, Alireza Jolfaei

**Abstracts**: In this study, we focus on the impact of adversarial attacks on deep learning-based anomaly detection in CPS networks and implement a mitigation approach against the attack by retraining models using adversarial samples. We use the Bot-IoT and Modbus IoT datasets to represent the two CPS networks. We train deep learning models and generate adversarial samples using these datasets. These datasets are captured from IoT and Industrial IoT (IIoT) networks. They both provide samples of normal and attack activities. The deep learning model trained with these datasets showed high accuracy in detecting attacks. An Artificial Neural Network (ANN) is adopted with one input layer, four intermediate layers, and one output layer. The output layer has two nodes representing the binary classification results. To generate adversarial samples for the experiment, we used a function called the `fast_gradient_method' from the Cleverhans library. The experimental result demonstrates the influence of FGSM adversarial samples on the accuracy of the predictions and proves the effectiveness of using the retrained model to defend against adversarial attacks.



## **18. An Efficient Method for Sample Adversarial Perturbations against Nonlinear Support Vector Machines**

cs.LG

**SubmitDate**: 2022-06-12    [paper-pdf](http://arxiv.org/pdf/2206.05664v1)

**Authors**: Wen Su, Qingna Li

**Abstracts**: Adversarial perturbations have drawn great attentions in various machine learning models. In this paper, we investigate the sample adversarial perturbations for nonlinear support vector machines (SVMs). Due to the implicit form of the nonlinear functions mapping data to the feature space, it is difficult to obtain the explicit form of the adversarial perturbations. By exploring the special property of nonlinear SVMs, we transform the optimization problem of attacking nonlinear SVMs into a nonlinear KKT system. Such a system can be solved by various numerical methods. Numerical results show that our method is efficient in computing adversarial perturbations.



## **19. Reward Poisoning Attacks on Offline Multi-Agent Reinforcement Learning**

cs.LG

**SubmitDate**: 2022-06-11    [paper-pdf](http://arxiv.org/pdf/2206.01888v2)

**Authors**: Young Wu, Jeremey McMahan, Xiaojin Zhu, Qiaomin Xie

**Abstracts**: We expose the danger of reward poisoning in offline multi-agent reinforcement learning (MARL), whereby an attacker can modify the reward vectors to different learners in an offline data set while incurring a poisoning cost. Based on the poisoned data set, all rational learners using some confidence-bound-based MARL algorithm will infer that a target policy - chosen by the attacker and not necessarily a solution concept originally - is the Markov perfect dominant strategy equilibrium for the underlying Markov Game, hence they will adopt this potentially damaging target policy in the future. We characterize the exact conditions under which the attacker can install a target policy. We further show how the attacker can formulate a linear program to minimize its poisoning cost. Our work shows the need for robust MARL against adversarial attacks.



## **20. SemAttack: Natural Textual Attacks via Different Semantic Spaces**

cs.CL

Published at Findings of NAACL 2022

**SubmitDate**: 2022-06-11    [paper-pdf](http://arxiv.org/pdf/2205.01287v3)

**Authors**: Boxin Wang, Chejian Xu, Xiangyu Liu, Yu Cheng, Bo Li

**Abstracts**: Recent studies show that pre-trained language models (LMs) are vulnerable to textual adversarial attacks. However, existing attack methods either suffer from low attack success rates or fail to search efficiently in the exponentially large perturbation space. We propose an efficient and effective framework SemAttack to generate natural adversarial text by constructing different semantic perturbation functions. In particular, SemAttack optimizes the generated perturbations constrained on generic semantic spaces, including typo space, knowledge space (e.g., WordNet), contextualized semantic space (e.g., the embedding space of BERT clusterings), or the combination of these spaces. Thus, the generated adversarial texts are more semantically close to the original inputs. Extensive experiments reveal that state-of-the-art (SOTA) large-scale LMs (e.g., DeBERTa-v2) and defense strategies (e.g., FreeLB) are still vulnerable to SemAttack. We further demonstrate that SemAttack is general and able to generate natural adversarial texts for different languages (e.g., English and Chinese) with high attack success rates. Human evaluations also confirm that our generated adversarial texts are natural and barely affect human performance. Our code is publicly available at https://github.com/AI-secure/SemAttack.



## **21. How does Heterophily Impact Robustness of Graph Neural Networks? Theoretical Connections and Practical Implications**

cs.LG

Accepted to KDD 2022; complete version with full appendix; 21 pages,  2 figures

**SubmitDate**: 2022-06-11    [paper-pdf](http://arxiv.org/pdf/2106.07767v3)

**Authors**: Jiong Zhu, Junchen Jin, Donald Loveland, Michael T. Schaub, Danai Koutra

**Abstracts**: We bridge two research directions on graph neural networks (GNNs), by formalizing the relation between heterophily of node labels (i.e., connected nodes tend to have dissimilar labels) and the robustness of GNNs to adversarial attacks. Our theoretical and empirical analyses show that for homophilous graph data, impactful structural attacks always lead to reduced homophily, while for heterophilous graph data the change in the homophily level depends on the node degrees. These insights have practical implications for defending against attacks on real-world graphs: we deduce that separate aggregators for ego- and neighbor-embeddings, a design principle which has been identified to significantly improve prediction for heterophilous graph data, can also offer increased robustness to GNNs. Our comprehensive experiments show that GNNs merely adopting this design achieve improved empirical and certifiable robustness compared to the best-performing unvaccinated model. Additionally, combining this design with explicit defense mechanisms against adversarial attacks leads to an improved robustness with up to 18.33% performance increase under attacks compared to the best-performing vaccinated model.



## **22. Game-Theoretic Neyman-Pearson Detection to Combat Strategic Evasion**

cs.CR

**SubmitDate**: 2022-06-10    [paper-pdf](http://arxiv.org/pdf/2206.05276v1)

**Authors**: Yinan Hu, Quanyan Zhu

**Abstracts**: The security in networked systems depends greatly on recognizing and identifying adversarial behaviors. Traditional detection methods focus on specific categories of attacks and have become inadequate for increasingly stealthy and deceptive attacks that are designed to bypass detection strategically. This work aims to develop a holistic theory to countermeasure such evasive attacks. We focus on extending a fundamental class of statistical-based detection methods based on Neyman-Pearson's (NP) hypothesis testing formulation. We propose game-theoretic frameworks to capture the conflicting relationship between a strategic evasive attacker and an evasion-aware NP detector. By analyzing both the equilibrium behaviors of the attacker and the NP detector, we characterize their performance using Equilibrium Receiver-Operational-Characteristic (EROC) curves. We show that the evasion-aware NP detectors outperform the passive ones in the way that the former can act strategically against the attacker's behavior and adaptively modify their decision rules based on the received messages. In addition, we extend our framework to a sequential setting where the user sends out identically distributed messages. We corroborate the analytical results with a case study of anomaly detection.



## **23. Blades: A Simulator for Attacks and Defenses in Federated Learning**

cs.CR

**SubmitDate**: 2022-06-10    [paper-pdf](http://arxiv.org/pdf/2206.05359v1)

**Authors**: Shenghui Li, Li Ju, Tianru Zhang, Edith Ngai, Thiemo Voigt

**Abstracts**: Federated learning enables distributed training across a set of clients, without requiring any of the participants to reveal their private training data to a centralized entity or each other. Due to the nature of decentralized execution, federated learning is vulnerable to attacks from adversarial (Byzantine) clients by modifying the local updates to their desires. Therefore, it is important to develop robust federated learning algorithms that can defend Byzantine clients without losing model convergence and performance. In the study of robustness problems, a simulator can simplify and accelerate the implementation and evaluation of attack and defense strategies. However, there is a lack of open-source simulators to meet such needs. Herein, we present Blades, a scalable, extensible, and easily configurable simulator to assist researchers and developers in efficiently implementing and validating novel strategies against baseline algorithms in robust federated learning. Blades is built upon a versatile distributed framework Ray, making it effortless to parallelize single machine code from a single CPU to multi-core, multi-GPU, or multi-node with minimal configurations. Blades contains built-in implementations of representative attack and defense strategies and provides user-friendly interfaces to easily incorporate new ideas. We maintain the source code and documents at https://github.com/bladesteam/blades.



## **24. Hierarchical Federated Learning with Privacy**

cs.LG

**SubmitDate**: 2022-06-10    [paper-pdf](http://arxiv.org/pdf/2206.05209v1)

**Authors**: Varun Chandrasekaran, Suman Banerjee, Diego Perino, Nicolas Kourtellis

**Abstracts**: Federated learning (FL), where data remains at the federated clients, and where only gradient updates are shared with a central aggregator, was assumed to be private. Recent work demonstrates that adversaries with gradient-level access can mount successful inference and reconstruction attacks. In such settings, differentially private (DP) learning is known to provide resilience. However, approaches used in the status quo (\ie central and local DP) introduce disparate utility vs. privacy trade-offs. In this work, we take the first step towards mitigating such trade-offs through {\em hierarchical FL (HFL)}. We demonstrate that by the introduction of a new intermediary level where calibrated DP noise can be added, better privacy vs. utility trade-offs can be obtained; we term this {\em hierarchical DP (HDP)}. Our experiments with 3 different datasets (commonly used as benchmarks for FL) suggest that HDP produces models as accurate as those obtained using central DP, where noise is added at a central aggregator. Such an approach also provides comparable benefit against inference adversaries as in the local DP case, where noise is added at the federated clients.



## **25. Localized adversarial artifacts for compressed sensing MRI**

eess.IV

14 pages, 7 figures

**SubmitDate**: 2022-06-10    [paper-pdf](http://arxiv.org/pdf/2206.05289v1)

**Authors**: Rima Alaifari, Giovanni S. Alberti, Tandri Gauksson

**Abstracts**: As interest in deep neural networks (DNNs) for image reconstruction tasks grows, their reliability has been called into question (Antun et al., 2020; Gottschling et al., 2020). However, recent work has shown that compared to total variation (TV) minimization, they show similar robustness to adversarial noise in terms of $\ell^2$-reconstruction error (Genzel et al., 2022). We consider a different notion of robustness, using the $\ell^\infty$-norm, and argue that localized reconstruction artifacts are a more relevant defect than the $\ell^2$-error. We create adversarial perturbations to undersampled MRI measurements which induce severe localized artifacts in the TV-regularized reconstruction. The same attack method is not as effective against DNN based reconstruction. Finally, we show that this phenomenon is inherent to reconstruction methods for which exact recovery can be guaranteed, as with compressed sensing reconstructions with $\ell^1$- or TV-minimization.



## **26. SERVFAIL: The Unintended Consequences of Algorithm Agility in DNSSEC**

cs.CR

Withdrawn on request of one of the persons listed as authors

**SubmitDate**: 2022-06-10    [paper-pdf](http://arxiv.org/pdf/2205.10608v2)

**Authors**: Elias Heftrig, Jean-Pierre Seifert, Haya Shulman, Peter Thomassen, Michael Waidner, Nils Wisiol

**Abstracts**: Cryptographic algorithm agility is an important property for DNSSEC: it allows easy deployment of new algorithms if the existing ones are no longer secure. Significant operational and research efforts are dedicated to pushing the deployment of new algorithms in DNSSEC forward. Recent research shows that DNSSEC is gradually achieving algorithm agility: most DNSSEC supporting resolvers can validate a number of different algorithms and domains are increasingly signed with cryptographically strong ciphers.   In this work we show for the first time that the cryptographic agility in DNSSEC, although critical for making DNS secure with strong cryptography, also introduces a severe vulnerability. We find that under certain conditions, when new algorithms are listed in signed DNS responses, the resolvers do not validate DNSSEC. As a result, domains that deploy new ciphers, risk exposing the validating resolvers to cache poisoning attacks.   We use this to develop DNSSEC-downgrade attacks and show that in some situations these attacks can be launched even by off-path adversaries. We experimentally and ethically evaluate our attacks against popular DNS resolver implementations, public DNS providers, and DNS services used by web clients worldwide. We validate the success of DNSSEC-downgrade attacks by poisoning the resolvers: we inject fake records, in signed domains, into the caches of validating resolvers. We find that major DNS providers, such as Google Public DNS and Cloudflare, as well as 70% of DNS resolvers used by web clients are vulnerable to our attacks.   We trace the factors that led to this situation and provide recommendations.



## **27. Enhancing Clean Label Backdoor Attack with Two-phase Specific Triggers**

cs.CR

**SubmitDate**: 2022-06-10    [paper-pdf](http://arxiv.org/pdf/2206.04881v1)

**Authors**: Nan Luo, Yuanzhang Li, Yajie Wang, Shangbo Wu, Yu-an Tan, Quanxin Zhang

**Abstracts**: Backdoor attacks threaten Deep Neural Networks (DNNs). Towards stealthiness, researchers propose clean-label backdoor attacks, which require the adversaries not to alter the labels of the poisoned training datasets. Clean-label settings make the attack more stealthy due to the correct image-label pairs, but some problems still exist: first, traditional methods for poisoning training data are ineffective; second, traditional triggers are not stealthy which are still perceptible. To solve these problems, we propose a two-phase and image-specific triggers generation method to enhance clean-label backdoor attacks. Our methods are (1) powerful: our triggers can both promote the two phases (i.e., the backdoor implantation and activation phase) in backdoor attacks simultaneously; (2) stealthy: our triggers are generated from each image. They are image-specific instead of fixed triggers. Extensive experiments demonstrate that our approach can achieve a fantastic attack success rate~(98.98%) with low poisoning rate~(5%), high stealthiness under many evaluation metrics and is resistant to backdoor defense methods.



## **28. ReFace: Real-time Adversarial Attacks on Face Recognition Systems**

cs.CV

**SubmitDate**: 2022-06-09    [paper-pdf](http://arxiv.org/pdf/2206.04783v1)

**Authors**: Shehzeen Hussain, Todd Huster, Chris Mesterharm, Paarth Neekhara, Kevin An, Malhar Jere, Harshvardhan Sikka, Farinaz Koushanfar

**Abstracts**: Deep neural network based face recognition models have been shown to be vulnerable to adversarial examples. However, many of the past attacks require the adversary to solve an input-dependent optimization problem using gradient descent which makes the attack impractical in real-time. These adversarial examples are also tightly coupled to the attacked model and are not as successful in transferring to different models. In this work, we propose ReFace, a real-time, highly-transferable attack on face recognition models based on Adversarial Transformation Networks (ATNs). ATNs model adversarial example generation as a feed-forward neural network. We find that the white-box attack success rate of a pure U-Net ATN falls substantially short of gradient-based attacks like PGD on large face recognition datasets. We therefore propose a new architecture for ATNs that closes this gap while maintaining a 10000x speedup over PGD. Furthermore, we find that at a given perturbation magnitude, our ATN adversarial perturbations are more effective in transferring to new face recognition models than PGD. ReFace attacks can successfully deceive commercial face recognition services in a transfer attack setting and reduce face identification accuracy from 82% to 16.4% for AWS SearchFaces API and Azure face verification accuracy from 91% to 50.1%.



## **29. Network insensitivity to parameter noise via adversarial regularization**

cs.LG

**SubmitDate**: 2022-06-09    [paper-pdf](http://arxiv.org/pdf/2106.05009v3)

**Authors**: Julian Büchel, Fynn Faber, Dylan R. Muir

**Abstracts**: Neuromorphic neural network processors, in the form of compute-in-memory crossbar arrays of memristors, or in the form of subthreshold analog and mixed-signal ASICs, promise enormous advantages in compute density and energy efficiency for NN-based ML tasks. However, these technologies are prone to computational non-idealities, due to process variation and intrinsic device physics. This degrades the task performance of networks deployed to the processor, by introducing parameter noise into the deployed model. While it is possible to calibrate each device, or train networks individually for each processor, these approaches are expensive and impractical for commercial deployment. Alternative methods are therefore needed to train networks that are inherently robust against parameter variation, as a consequence of network architecture and parameters. We present a new adversarial network optimisation algorithm that attacks network parameters during training, and promotes robust performance during inference in the face of parameter variation. Our approach introduces a regularization term penalising the susceptibility of a network to weight perturbation. We compare against previous approaches for producing parameter insensitivity such as dropout, weight smoothing and introducing parameter noise during training. We show that our approach produces models that are more robust to targeted parameter variation, and equally robust to random parameter variation. Our approach finds minima in flatter locations in the weight-loss landscape compared with other approaches, highlighting that the networks found by our technique are less sensitive to parameter perturbation. Our work provides an approach to deploy neural network architectures to inference devices that suffer from computational non-idealities, with minimal loss of performance. ...



## **30. Unlearning Protected User Attributes in Recommendations with Adversarial Training**

cs.IR

Accepted at SIGIR 2022

**SubmitDate**: 2022-06-09    [paper-pdf](http://arxiv.org/pdf/2206.04500v1)

**Authors**: Christian Ganhör, David Penz, Navid Rekabsaz, Oleg Lesota, Markus Schedl

**Abstracts**: Collaborative filtering algorithms capture underlying consumption patterns, including the ones specific to particular demographics or protected information of users, e.g. gender, race, and location. These encoded biases can influence the decision of a recommendation system (RS) towards further separation of the contents provided to various demographic subgroups, and raise privacy concerns regarding the disclosure of users' protected attributes. In this work, we investigate the possibility and challenges of removing specific protected information of users from the learned interaction representations of a RS algorithm, while maintaining its effectiveness. Specifically, we incorporate adversarial training into the state-of-the-art MultVAE architecture, resulting in a novel model, Adversarial Variational Auto-Encoder with Multinomial Likelihood (Adv-MultVAE), which aims at removing the implicit information of protected attributes while preserving recommendation performance. We conduct experiments on the MovieLens-1M and LFM-2b-DemoBias datasets, and evaluate the effectiveness of the bias mitigation method based on the inability of external attackers in revealing the users' gender information from the model. Comparing with baseline MultVAE, the results show that Adv-MultVAE, with marginal deterioration in performance (w.r.t. NDCG and recall), largely mitigates inherent biases in the model on both datasets.



## **31. Subfield Algorithms for Ideal- and Module-SVP Based on the Decomposition Group**

cs.CR

29 pages plus appendix, to appear in Banach Center Publications

**SubmitDate**: 2022-06-09    [paper-pdf](http://arxiv.org/pdf/2105.03219v3)

**Authors**: Christian Porter, Andrew Mendelsohn, Cong Ling

**Abstracts**: Whilst lattice-based cryptosystems are believed to be resistant to quantum attack, they are often forced to pay for that security with inefficiencies in implementation. This problem is overcome by ring- and module-based schemes such as Ring-LWE or Module-LWE, whose keysize can be reduced by exploiting its algebraic structure, allowing for faster computations. Many rings may be chosen to define such cryptoschemes, but cyclotomic rings, due to their cyclic nature allowing for easy multiplication, are the community standard. However, there is still much uncertainty as to whether this structure may be exploited to an adversary's benefit. In this paper, we show that the decomposition group of a cyclotomic ring of arbitrary conductor can be utilised to significantly decrease the dimension of the ideal (or module) lattice required to solve a given instance of SVP. Moreover, we show that there exist a large number of rational primes for which, if the prime ideal factors of an ideal lie over primes of this form, give rise to an "easy" instance of SVP. It is important to note that the work on ideal SVP does not break Ring-LWE, since its security reduction is from worst case ideal SVP to average case Ring-LWE, and is one way.



## **32. CARLA-GeAR: a Dataset Generator for a Systematic Evaluation of Adversarial Robustness of Vision Models**

cs.CV

**SubmitDate**: 2022-06-09    [paper-pdf](http://arxiv.org/pdf/2206.04365v1)

**Authors**: Federico Nesti, Giulio Rossolini, Gianluca D'Amico, Alessandro Biondi, Giorgio Buttazzo

**Abstracts**: Adversarial examples represent a serious threat for deep neural networks in several application domains and a huge amount of work has been produced to investigate them and mitigate their effects. Nevertheless, no much work has been devoted to the generation of datasets specifically designed to evaluate the adversarial robustness of neural models. This paper presents CARLA-GeAR, a tool for the automatic generation of photo-realistic synthetic datasets that can be used for a systematic evaluation of the adversarial robustness of neural models against physical adversarial patches, as well as for comparing the performance of different adversarial defense/detection methods. The tool is built on the CARLA simulator, using its Python API, and allows the generation of datasets for several vision tasks in the context of autonomous driving. The adversarial patches included in the generated datasets are attached to billboards or the back of a truck and are crafted by using state-of-the-art white-box attack strategies to maximize the prediction error of the model under test. Finally, the paper presents an experimental study to evaluate the performance of some defense methods against such attacks, showing how the datasets generated with CARLA-GeAR might be used in future work as a benchmark for adversarial defense in the real world. All the code and datasets used in this paper are available at http://carlagear.retis.santannapisa.it.



## **33. Plug & Play Attacks: Towards Robust and Flexible Model Inversion Attacks**

cs.LG

Accepted by ICML 2022

**SubmitDate**: 2022-06-09    [paper-pdf](http://arxiv.org/pdf/2201.12179v4)

**Authors**: Lukas Struppek, Dominik Hintersdorf, Antonio De Almeida Correia, Antonia Adler, Kristian Kersting

**Abstracts**: Model inversion attacks (MIAs) aim to create synthetic images that reflect the class-wise characteristics from a target classifier's private training data by exploiting the model's learned knowledge. Previous research has developed generative MIAs that use generative adversarial networks (GANs) as image priors tailored to a specific target model. This makes the attacks time- and resource-consuming, inflexible, and susceptible to distributional shifts between datasets. To overcome these drawbacks, we present Plug & Play Attacks, which relax the dependency between the target model and image prior, and enable the use of a single GAN to attack a wide range of targets, requiring only minor adjustments to the attack. Moreover, we show that powerful MIAs are possible even with publicly available pre-trained GANs and under strong distributional shifts, for which previous approaches fail to produce meaningful results. Our extensive evaluation confirms the improved robustness and flexibility of Plug & Play Attacks and their ability to create high-quality images revealing sensitive class characteristics.



## **34. Learning to Break Deep Perceptual Hashing: The Use Case NeuralHash**

cs.LG

Accepted by ACM FAccT 2022 as Oral

**SubmitDate**: 2022-06-09    [paper-pdf](http://arxiv.org/pdf/2111.06628v4)

**Authors**: Lukas Struppek, Dominik Hintersdorf, Daniel Neider, Kristian Kersting

**Abstracts**: Apple recently revealed its deep perceptual hashing system NeuralHash to detect child sexual abuse material (CSAM) on user devices before files are uploaded to its iCloud service. Public criticism quickly arose regarding the protection of user privacy and the system's reliability. In this paper, we present the first comprehensive empirical analysis of deep perceptual hashing based on NeuralHash. Specifically, we show that current deep perceptual hashing may not be robust. An adversary can manipulate the hash values by applying slight changes in images, either induced by gradient-based approaches or simply by performing standard image transformations, forcing or preventing hash collisions. Such attacks permit malicious actors easily to exploit the detection system: from hiding abusive material to framing innocent users, everything is possible. Moreover, using the hash values, inferences can still be made about the data stored on user devices. In our view, based on our results, deep perceptual hashing in its current form is generally not ready for robust client-side scanning and should not be used from a privacy perspective.



## **35. Bounding Training Data Reconstruction in Private (Deep) Learning**

cs.LG

**SubmitDate**: 2022-06-09    [paper-pdf](http://arxiv.org/pdf/2201.12383v3)

**Authors**: Chuan Guo, Brian Karrer, Kamalika Chaudhuri, Laurens van der Maaten

**Abstracts**: Differential privacy is widely accepted as the de facto method for preventing data leakage in ML, and conventional wisdom suggests that it offers strong protection against privacy attacks. However, existing semantic guarantees for DP focus on membership inference, which may overestimate the adversary's capabilities and is not applicable when membership status itself is non-sensitive. In this paper, we derive the first semantic guarantees for DP mechanisms against training data reconstruction attacks under a formal threat model. We show that two distinct privacy accounting methods -- Renyi differential privacy and Fisher information leakage -- both offer strong semantic protection against data reconstruction attacks.



## **36. Blacklight: Scalable Defense for Neural Networks against Query-Based Black-Box Attacks**

cs.CR

**SubmitDate**: 2022-06-09    [paper-pdf](http://arxiv.org/pdf/2006.14042v3)

**Authors**: Huiying Li, Shawn Shan, Emily Wenger, Jiayun Zhang, Haitao Zheng, Ben Y. Zhao

**Abstracts**: Deep learning systems are known to be vulnerable to adversarial examples. In particular, query-based black-box attacks do not require knowledge of the deep learning model, but can compute adversarial examples over the network by submitting queries and inspecting returns. Recent work largely improves the efficiency of those attacks, demonstrating their practicality on today's ML-as-a-service platforms.   We propose Blacklight, a new defense against query-based black-box adversarial attacks. The fundamental insight driving our design is that, to compute adversarial examples, these attacks perform iterative optimization over the network, producing image queries highly similar in the input space. Blacklight detects query-based black-box attacks by detecting highly similar queries, using an efficient similarity engine operating on probabilistic content fingerprints. We evaluate Blacklight against eight state-of-the-art attacks, across a variety of models and image classification tasks. Blacklight identifies them all, often after only a handful of queries. By rejecting all detected queries, Blacklight prevents any attack to complete, even when attackers persist to submit queries after account ban or query rejection. Blacklight is also robust against several powerful countermeasures, including an optimal black-box attack that approximates white-box attacks in efficiency. Finally, we illustrate how Blacklight generalizes to other domains like text classification.



## **37. Adversarial Text Normalization**

cs.CL

**SubmitDate**: 2022-06-08    [paper-pdf](http://arxiv.org/pdf/2206.04137v1)

**Authors**: Joanna Bitton, Maya Pavlova, Ivan Evtimov

**Abstracts**: Text-based adversarial attacks are becoming more commonplace and accessible to general internet users. As these attacks proliferate, the need to address the gap in model robustness becomes imminent. While retraining on adversarial data may increase performance, there remains an additional class of character-level attacks on which these models falter. Additionally, the process to retrain a model is time and resource intensive, creating a need for a lightweight, reusable defense. In this work, we propose the Adversarial Text Normalizer, a novel method that restores baseline performance on attacked content with low computational overhead. We evaluate the efficacy of the normalizer on two problem areas prone to adversarial attacks, i.e. Hate Speech and Natural Language Inference. We find that text normalization provides a task-agnostic defense against character-level attacks that can be implemented supplementary to adversarial retraining solutions, which are more suited for semantic alterations.



## **38. PrivHAR: Recognizing Human Actions From Privacy-preserving Lens**

cs.CV

**SubmitDate**: 2022-06-08    [paper-pdf](http://arxiv.org/pdf/2206.03891v1)

**Authors**: Carlos Hinojosa, Miguel Marquez, Henry Arguello, Ehsan Adeli, Li Fei-Fei, Juan Carlos Niebles

**Abstracts**: The accelerated use of digital cameras prompts an increasing concern about privacy and security, particularly in applications such as action recognition. In this paper, we propose an optimizing framework to provide robust visual privacy protection along the human action recognition pipeline. Our framework parameterizes the camera lens to successfully degrade the quality of the videos to inhibit privacy attributes and protect against adversarial attacks while maintaining relevant features for activity recognition. We validate our approach with extensive simulations and hardware experiments.



## **39. Standalone Neural ODEs with Sensitivity Analysis**

cs.LG

25 pages, 15 figures; typos corrected

**SubmitDate**: 2022-06-08    [paper-pdf](http://arxiv.org/pdf/2205.13933v2)

**Authors**: Rym Jaroudi, Lukáš Malý, Gabriel Eilertsen, B. Tomas Johansson, Jonas Unger, George Baravdish

**Abstracts**: This paper presents the Standalone Neural ODE (sNODE), a continuous-depth neural ODE model capable of describing a full deep neural network. This uses a novel nonlinear conjugate gradient (NCG) descent optimization scheme for training, where the Sobolev gradient can be incorporated to improve smoothness of model weights. We also present a general formulation of the neural sensitivity problem and show how it is used in the NCG training. The sensitivity analysis provides a reliable measure of uncertainty propagation throughout a network, and can be used to study model robustness and to generate adversarial attacks. Our evaluations demonstrate that our novel formulations lead to increased robustness and performance as compared to ResNet models, and that it opens up for new opportunities for designing and developing machine learning with improved explainability.



## **40. Wavelet Regularization Benefits Adversarial Training**

cs.CV

Preprint version

**SubmitDate**: 2022-06-08    [paper-pdf](http://arxiv.org/pdf/2206.03727v1)

**Authors**: Jun Yan, Huilin Yin, Xiaoyang Deng, Ziming Zhao, Wancheng Ge, Hao Zhang, Gerhard Rigoll

**Abstracts**: Adversarial training methods are state-of-the-art (SOTA) empirical defense methods against adversarial examples. Many regularization methods have been proven to be effective with the combination of adversarial training. Nevertheless, such regularization methods are implemented in the time domain. Since adversarial vulnerability can be regarded as a high-frequency phenomenon, it is essential to regulate the adversarially-trained neural network models in the frequency domain. Faced with these challenges, we make a theoretical analysis on the regularization property of wavelets which can enhance adversarial training. We propose a wavelet regularization method based on the Haar wavelet decomposition which is named Wavelet Average Pooling. This wavelet regularization module is integrated into the wide residual neural network so that a new WideWaveletResNet model is formed. On the datasets of CIFAR-10 and CIFAR-100, our proposed Adversarial Wavelet Training method realizes considerable robustness under different types of attacks. It verifies the assumption that our wavelet regularization method can enhance adversarial robustness especially in the deep wide neural networks. The visualization experiments of the Frequency Principle (F-Principle) and interpretability are implemented to show the effectiveness of our method. A detailed comparison based on different wavelet base functions is presented. The code is available at the repository: \url{https://github.com/momo1986/AdversarialWaveletTraining}.



## **41. PRADA: Practical Black-Box Adversarial Attacks against Neural Ranking Models**

cs.IR

**SubmitDate**: 2022-06-08    [paper-pdf](http://arxiv.org/pdf/2204.01321v3)

**Authors**: Chen Wu, Ruqing Zhang, Jiafeng Guo, Maarten de Rijke, Yixing Fan, Xueqi Cheng

**Abstracts**: Neural ranking models (NRMs) have shown remarkable success in recent years, especially with pre-trained language models. However, deep neural models are notorious for their vulnerability to adversarial examples. Adversarial attacks may become a new type of web spamming technique given our increased reliance on neural information retrieval models. Therefore, it is important to study potential adversarial attacks to identify vulnerabilities of NRMs before they are deployed. In this paper, we introduce the Word Substitution Ranking Attack (WSRA) task against NRMs, which aims to promote a target document in rankings by adding adversarial perturbations to its text. We focus on the decision-based black-box attack setting, where the attackers cannot directly get access to the model information, but can only query the target model to obtain the rank positions of the partial retrieved list. This attack setting is realistic in real-world search engines. We propose a novel Pseudo Relevance-based ADversarial ranking Attack method (PRADA) that learns a surrogate model based on Pseudo Relevance Feedback (PRF) to generate gradients for finding the adversarial perturbations. Experiments on two web search benchmark datasets show that PRADA can outperform existing attack strategies and successfully fool the NRM with small indiscernible perturbations of text.



## **42. Latent Boundary-guided Adversarial Training**

cs.LG

To appear in Machine Learning

**SubmitDate**: 2022-06-08    [paper-pdf](http://arxiv.org/pdf/2206.03717v1)

**Authors**: Xiaowei Zhou, Ivor W. Tsang, Jie Yin

**Abstracts**: Deep Neural Networks (DNNs) have recently achieved great success in many classification tasks. Unfortunately, they are vulnerable to adversarial attacks that generate adversarial examples with a small perturbation to fool DNN models, especially in model sharing scenarios. Adversarial training is proved to be the most effective strategy that injects adversarial examples into model training to improve the robustness of DNN models to adversarial attacks. However, adversarial training based on the existing adversarial examples fails to generalize well to standard, unperturbed test data. To achieve a better trade-off between standard accuracy and adversarial robustness, we propose a novel adversarial training framework called LAtent bounDary-guided aDvErsarial tRaining (LADDER) that adversarially trains DNN models on latent boundary-guided adversarial examples. As opposed to most of the existing methods that generate adversarial examples in the input space, LADDER generates a myriad of high-quality adversarial examples through adding perturbations to latent features. The perturbations are made along the normal of the decision boundary constructed by an SVM with an attention mechanism. We analyze the merits of our generated boundary-guided adversarial examples from a boundary field perspective and visualization view. Extensive experiments and detailed analysis on MNIST, SVHN, CelebA, and CIFAR-10 validate the effectiveness of LADDER in achieving a better trade-off between standard accuracy and adversarial robustness as compared with vanilla DNNs and competitive baselines.



## **43. Autoregressive Perturbations for Data Poisoning**

cs.LG

21 pages, 13 figures. Code available at  https://github.com/psandovalsegura/autoregressive-poisoning

**SubmitDate**: 2022-06-08    [paper-pdf](http://arxiv.org/pdf/2206.03693v1)

**Authors**: Pedro Sandoval-Segura, Vasu Singla, Jonas Geiping, Micah Goldblum, Tom Goldstein, David W. Jacobs

**Abstracts**: The prevalence of data scraping from social media as a means to obtain datasets has led to growing concerns regarding unauthorized use of data. Data poisoning attacks have been proposed as a bulwark against scraping, as they make data "unlearnable" by adding small, imperceptible perturbations. Unfortunately, existing methods require knowledge of both the target architecture and the complete dataset so that a surrogate network can be trained, the parameters of which are used to generate the attack. In this work, we introduce autoregressive (AR) poisoning, a method that can generate poisoned data without access to the broader dataset. The proposed AR perturbations are generic, can be applied across different datasets, and can poison different architectures. Compared to existing unlearnable methods, our AR poisons are more resistant against common defenses such as adversarial training and strong data augmentations. Our analysis further provides insight into what makes an effective data poison.



## **44. SHORTSTACK: Distributed, Fault-tolerant, Oblivious Data Access**

cs.CR

Full version of USENIX OSDI'22 paper

**SubmitDate**: 2022-06-08    [paper-pdf](http://arxiv.org/pdf/2205.14281v2)

**Authors**: Midhul Vuppalapati, Kushal Babel, Anurag Khandelwal, Rachit Agarwal

**Abstracts**: Many applications that benefit from data offload to cloud services operate on private data. A now-long line of work has shown that, even when data is offloaded in an encrypted form, an adversary can learn sensitive information by analyzing data access patterns. Existing techniques for oblivious data access-that protect against access pattern attacks-require a centralized and stateful trusted proxy to orchestrate data accesses from applications to cloud services. We show that, in failure-prone deployments, such a centralized and stateful proxy results in violation of oblivious data access security guarantees and/or system unavailability. We thus initiate the study of distributed, fault-tolerant, oblivious data access.   We present SHORTSTACK, a distributed proxy architecture for oblivious data access in failure-prone deployments. SHORTSTACK achieves the classical obliviousness guarantee--access patterns observed by the adversary being independent of the input--even under a powerful passive persistent adversary that can force failure of arbitrary (bounded-sized) subset of proxy servers at arbitrary times. We also introduce a security model that enables studying oblivious data access with distributed, failure-prone, servers. We provide a formal proof that SHORTSTACK enables oblivious data access under this model, and show empirically that SHORTSTACK performance scales near-linearly with number of distributed proxy servers.



## **45. Dap-FL: Federated Learning flourishes by adaptive tuning and secure aggregation**

cs.CR

**SubmitDate**: 2022-06-08    [paper-pdf](http://arxiv.org/pdf/2206.03623v1)

**Authors**: Qian Chen, Zilong Wang, Jiawei Chen, Haonan Yan, Xiaodong Lin

**Abstracts**: Federated learning (FL), an attractive and promising distributed machine learning paradigm, has sparked extensive interest in exploiting tremendous data stored on ubiquitous mobile devices. However, conventional FL suffers severely from resource heterogeneity, as clients with weak computational and communication capability may be unable to complete local training using the same local training hyper-parameters. In this paper, we propose Dap-FL, a deep deterministic policy gradient (DDPG)-assisted adaptive FL system, in which local learning rates and local training epochs are adaptively adjusted by all resource-heterogeneous clients through locally deployed DDPG-assisted adaptive hyper-parameter selection schemes. Particularly, the rationality of the proposed hyper-parameter selection scheme is confirmed through rigorous mathematical proof. Besides, due to the thoughtlessness of security consideration of adaptive FL systems in previous studies, we introduce the Paillier cryptosystem to aggregate local models in a secure and privacy-preserving manner. Rigorous analyses show that the proposed Dap-FL system could guarantee the security of clients' private local models against chosen-plaintext attacks and chosen-message attacks in a widely used honest-but-curious participants and active adversaries security model. In addition, through ingenious and extensive experiments, the proposed Dap-FL achieves higher global model prediction accuracy and faster convergence rates than conventional FL, and the comprehensiveness of the adjusted local training hyper-parameters is validated. More importantly, experimental results also show that the proposed Dap-FL achieves higher model prediction accuracy than two state-of-the-art RL-assisted FL methods, i.e., 6.03% higher than DDPG-based FL and 7.85% higher than DQN-based FL.



## **46. Random and Adversarial Bit Error Robustness: Energy-Efficient and Secure DNN Accelerators**

cs.LG

**SubmitDate**: 2022-06-07    [paper-pdf](http://arxiv.org/pdf/2104.08323v2)

**Authors**: David Stutz, Nandhini Chandramoorthy, Matthias Hein, Bernt Schiele

**Abstracts**: Deep neural network (DNN) accelerators received considerable attention in recent years due to the potential to save energy compared to mainstream hardware. Low-voltage operation of DNN accelerators allows to further reduce energy consumption, however, causes bit-level failures in the memory storing the quantized weights. Furthermore, DNN accelerators are vulnerable to adversarial attacks on voltage controllers or individual bits. In this paper, we show that a combination of robust fixed-point quantization, weight clipping, as well as random bit error training (RandBET) or adversarial bit error training (AdvBET) improves robustness against random or adversarial bit errors in quantized DNN weights significantly. This leads not only to high energy savings for low-voltage operation as well as low-precision quantization, but also improves security of DNN accelerators. In contrast to related work, our approach generalizes across operating voltages and accelerators and does not require hardware changes. Moreover, we present a novel adversarial bit error attack and are able to obtain robustness against both targeted and untargeted bit-level attacks. Without losing more than 0.8%/2% in test accuracy, we can reduce energy consumption on CIFAR10 by 20%/30% for 8/4-bit quantization. Allowing up to 320 adversarial bit errors, we reduce test error from above 90% (chance level) to 26.22%.



## **47. Optimal Clock Synchronization with Signatures**

cs.DC

**SubmitDate**: 2022-06-07    [paper-pdf](http://arxiv.org/pdf/2203.02553v2)

**Authors**: Christoph Lenzen, Julian Loss

**Abstracts**: Cryptographic signatures can be used to increase the resilience of distributed systems against adversarial attacks, by increasing the number of faulty parties that can be tolerated. While this is well-studied for consensus, it has been underexplored in the context of fault-tolerant clock synchronization, even in fully connected systems. Here, the honest parties of an $n$-node system are required to compute output clocks of small skew (i.e., maximum phase offset) despite local clock rates varying between $1$ and $\vartheta>1$, end-to-end communication delays varying between $d-u$ and $d$, and the interference from malicious parties. So far, it is only known that clock pulses of skew $d$ can be generated with (trivially optimal) resilience of $\lceil n/2\rceil-1$ (PODC `19), improving over the tight bound of $\lceil n/3\rceil-1$ holding without signatures for \emph{any} skew bound (STOC `84, PODC `85). Since typically $d\gg u$ and $\vartheta-1\ll 1$, this is far from the lower bound of $u+(\vartheta-1)d$ that applies even in the fault-free case (IPL `01).   We prove matching upper and lower bounds of $\Theta(u+(\vartheta-1)d)$ on the skew for the resilience range from $\lceil n/3\rceil$ to $\lceil n/2\rceil-1$. The algorithm showing the upper bound is, under the assumption that the adversary cannot forge signatures, deterministic. The lower bound holds even if clocks are initially perfectly synchronized, message delays between honest nodes are known, $\vartheta$ is arbitrarily close to one, and the synchronization algorithm is randomized. This has crucial implications for network designers that seek to leverage signatures for providing more robust time. In contrast to the setting without signatures, they must ensure that an attacker cannot easily bypass the lower bound on the delay on links with a faulty endpoint.



## **48. Towards Understanding and Mitigating Audio Adversarial Examples for Speaker Recognition**

cs.SD

**SubmitDate**: 2022-06-07    [paper-pdf](http://arxiv.org/pdf/2206.03393v1)

**Authors**: Guangke Chen, Zhe Zhao, Fu Song, Sen Chen, Lingling Fan, Feng Wang, Jiashui Wang

**Abstracts**: Speaker recognition systems (SRSs) have recently been shown to be vulnerable to adversarial attacks, raising significant security concerns. In this work, we systematically investigate transformation and adversarial training based defenses for securing SRSs. According to the characteristic of SRSs, we present 22 diverse transformations and thoroughly evaluate them using 7 recent promising adversarial attacks (4 white-box and 3 black-box) on speaker recognition. With careful regard for best practices in defense evaluations, we analyze the strength of transformations to withstand adaptive attacks. We also evaluate and understand their effectiveness against adaptive attacks when combined with adversarial training. Our study provides lots of useful insights and findings, many of them are new or inconsistent with the conclusions in the image and speech recognition domains, e.g., variable and constant bit rate speech compressions have different performance, and some non-differentiable transformations remain effective against current promising evasion techniques which often work well in the image domain. We demonstrate that the proposed novel feature-level transformation combined with adversarial training is rather effective compared to the sole adversarial training in a complete white-box setting, e.g., increasing the accuracy by 13.62% and attack cost by two orders of magnitude, while other transformations do not necessarily improve the overall defense capability. This work sheds further light on the research directions in this field. We also release our evaluation platform SPEAKERGUARD to foster further research.



## **49. Building Robust Ensembles via Margin Boosting**

cs.LG

Accepted by ICML 2022

**SubmitDate**: 2022-06-07    [paper-pdf](http://arxiv.org/pdf/2206.03362v1)

**Authors**: Dinghuai Zhang, Hongyang Zhang, Aaron Courville, Yoshua Bengio, Pradeep Ravikumar, Arun Sai Suggala

**Abstracts**: In the context of adversarial robustness, a single model does not usually have enough power to defend against all possible adversarial attacks, and as a result, has sub-optimal robustness. Consequently, an emerging line of work has focused on learning an ensemble of neural networks to defend against adversarial attacks. In this work, we take a principled approach towards building robust ensembles. We view this problem from the perspective of margin-boosting and develop an algorithm for learning an ensemble with maximum margin. Through extensive empirical evaluation on benchmark datasets, we show that our algorithm not only outperforms existing ensembling techniques, but also large models trained in an end-to-end fashion. An important byproduct of our work is a margin-maximizing cross-entropy (MCE) loss, which is a better alternative to the standard cross-entropy (CE) loss. Empirically, we show that replacing the CE loss in state-of-the-art adversarial training techniques with our MCE loss leads to significant performance improvement.



## **50. Adaptive Regularization for Adversarial Training**

stat.ML

**SubmitDate**: 2022-06-07    [paper-pdf](http://arxiv.org/pdf/2206.03353v1)

**Authors**: Dongyoon Yang, Insung Kong, Yongdai Kim

**Abstracts**: Adversarial training, which is to enhance robustness against adversarial attacks, has received much attention because it is easy to generate human-imperceptible perturbations of data to deceive a given deep neural network. In this paper, we propose a new adversarial training algorithm that is theoretically well motivated and empirically superior to other existing algorithms. A novel feature of the proposed algorithm is to use a data-adaptive regularization for robustifying a prediction model. We apply more regularization to data which are more vulnerable to adversarial attacks and vice versa. Even though the idea of data-adaptive regularization is not new, our data-adaptive regularization has a firm theoretical base of reducing an upper bound of the robust risk. Numerical experiments illustrate that our proposed algorithm improves the generalization (accuracy on clean samples) and robustness (accuracy on adversarial attacks) simultaneously to achieve the state-of-the-art performance.



