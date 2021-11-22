# New Adversarial Attack Papers
**update at 2021-11-22 23:56:49**

[中文版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. Rethinking Clustering for Robustness**

cs.LG

Accepted to the 32nd British Machine Vision Conference (BMVC'21)

**SubmitDate**: 2021-11-19    [paper-pdf](http://arxiv.org/pdf/2006.07682v3)

**Authors**: Motasem Alfarra, Juan C. Pérez, Adel Bibi, Ali Thabet, Pablo Arbeláez, Bernard Ghanem

**Abstracts**: This paper studies how encouraging semantically-aligned features during deep neural network training can increase network robustness. Recent works observed that Adversarial Training leads to robust models, whose learnt features appear to correlate with human perception. Inspired by this connection from robustness to semantics, we study the complementary connection: from semantics to robustness. To do so, we provide a robustness certificate for distance-based classification models (clustering-based classifiers). Moreover, we show that this certificate is tight, and we leverage it to propose ClusTR (Clustering Training for Robustness), a clustering-based and adversary-free training framework to learn robust models. Interestingly, \textit{ClusTR} outperforms adversarially-trained networks by up to $4\%$ under strong PGD attacks.



## **2. Meta Adversarial Perturbations**

cs.LG

**SubmitDate**: 2021-11-19    [paper-pdf](http://arxiv.org/pdf/2111.10291v1)

**Authors**: Chia-Hung Yuan, Pin-Yu Chen, Chia-Mu Yu

**Abstracts**: A plethora of attack methods have been proposed to generate adversarial examples, among which the iterative methods have been demonstrated the ability to find a strong attack. However, the computation of an adversarial perturbation for a new data point requires solving a time-consuming optimization problem from scratch. To generate a stronger attack, it normally requires updating a data point with more iterations. In this paper, we show the existence of a meta adversarial perturbation (MAP), a better initialization that causes natural images to be misclassified with high probability after being updated through only a one-step gradient ascent update, and propose an algorithm for computing such perturbations. We conduct extensive experiments, and the empirical results demonstrate that state-of-the-art deep neural networks are vulnerable to meta perturbations. We further show that these perturbations are not only image-agnostic, but also model-agnostic, as a single perturbation generalizes well across unseen data points and different neural network architectures.



## **3. Resilience from Diversity: Population-based approach to harden models against adversarial attacks**

cs.LG

10 pages, 6 figures, 5 tables

**SubmitDate**: 2021-11-19    [paper-pdf](http://arxiv.org/pdf/2111.10272v1)

**Authors**: Jasser Jasser, Ivan Garibay

**Abstracts**: Traditional deep learning models exhibit intriguing vulnerabilities that allow an attacker to force them to fail at their task. Notorious attacks such as the Fast Gradient Sign Method (FGSM) and the more powerful Projected Gradient Descent (PGD) generate adversarial examples by adding a magnitude of perturbation $\epsilon$ to the input's computed gradient, resulting in a deterioration of the effectiveness of the model's classification. This work introduces a model that is resilient to adversarial attacks. Our model leverages a well established principle from biological sciences: population diversity produces resilience against environmental changes. More precisely, our model consists of a population of $n$ diverse submodels, each one of them trained to individually obtain a high accuracy for the task at hand, while forced to maintain meaningful differences in their weight tensors. Each time our model receives a classification query, it selects a submodel from its population at random to answer the query. To introduce and maintain diversity in population of submodels, we introduce the concept of counter linking weights. A Counter-Linked Model (CLM) consists of submodels of the same architecture where a periodic random similarity examination is conducted during the simultaneous training to guarantee diversity while maintaining accuracy. In our testing, CLM robustness got enhanced by around 20% when tested on the MNIST dataset and at least 15% when tested on the CIFAR-10 dataset. When implemented with adversarially trained submodels, this methodology achieves state-of-the-art robustness. On the MNIST dataset with $\epsilon=0.3$, it achieved 94.34% against FGSM and 91% against PGD. On the CIFAR-10 dataset with $\epsilon=8/255$, it achieved 62.97% against FGSM and 59.16% against PGD.



## **4. Fast Minimum-norm Adversarial Attacks through Adaptive Norm Constraints**

cs.LG

Accepted at NeurIPS'21

**SubmitDate**: 2021-11-19    [paper-pdf](http://arxiv.org/pdf/2102.12827v3)

**Authors**: Maura Pintor, Fabio Roli, Wieland Brendel, Battista Biggio

**Abstracts**: Evaluating adversarial robustness amounts to finding the minimum perturbation needed to have an input sample misclassified. The inherent complexity of the underlying optimization requires current gradient-based attacks to be carefully tuned, initialized, and possibly executed for many computationally-demanding iterations, even if specialized to a given perturbation model. In this work, we overcome these limitations by proposing a fast minimum-norm (FMN) attack that works with different $\ell_p$-norm perturbation models ($p=0, 1, 2, \infty$), is robust to hyperparameter choices, does not require adversarial starting points, and converges within few lightweight steps. It works by iteratively finding the sample misclassified with maximum confidence within an $\ell_p$-norm constraint of size $\epsilon$, while adapting $\epsilon$ to minimize the distance of the current sample to the decision boundary. Extensive experiments show that FMN significantly outperforms existing attacks in terms of convergence speed and computation time, while reporting comparable or even smaller perturbation sizes.



## **5. Federated Learning for Malware Detection in IoT Devices**

cs.CR

**SubmitDate**: 2021-11-19    [paper-pdf](http://arxiv.org/pdf/2104.09994v3)

**Authors**: Valerian Rey, Pedro Miguel Sánchez Sánchez, Alberto Huertas Celdrán, Gérôme Bovet, Martin Jaggi

**Abstracts**: This work investigates the possibilities enabled by federated learning concerning IoT malware detection and studies security issues inherent to this new learning paradigm. In this context, a framework that uses federated learning to detect malware affecting IoT devices is presented. N-BaIoT, a dataset modeling network traffic of several real IoT devices while affected by malware, has been used to evaluate the proposed framework. Both supervised and unsupervised federated models (multi-layer perceptron and autoencoder) able to detect malware affecting seen and unseen IoT devices of N-BaIoT have been trained and evaluated. Furthermore, their performance has been compared to two traditional approaches. The first one lets each participant locally train a model using only its own data, while the second consists of making the participants share their data with a central entity in charge of training a global model. This comparison has shown that the use of more diverse and large data, as done in the federated and centralized methods, has a considerable positive impact on the model performance. Besides, the federated models, while preserving the participant's privacy, show similar results as the centralized ones. As an additional contribution and to measure the robustness of the federated approach, an adversarial setup with several malicious participants poisoning the federated model has been considered. The baseline model aggregation averaging step used in most federated learning algorithms appears highly vulnerable to different attacks, even with a single adversary. The performance of other model aggregation functions acting as countermeasures is thus evaluated under the same attack scenarios. These functions provide a significant improvement against malicious participants, but more efforts are still needed to make federated approaches robust.



## **6. Fooling Adversarial Training with Inducing Noise**

cs.LG

**SubmitDate**: 2021-11-19    [paper-pdf](http://arxiv.org/pdf/2111.10130v1)

**Authors**: Zhirui Wang, Yifei Wang, Yisen Wang

**Abstracts**: Adversarial training is widely believed to be a reliable approach to improve model robustness against adversarial attack. However, in this paper, we show that when trained on one type of poisoned data, adversarial training can also be fooled to have catastrophic behavior, e.g., $<1\%$ robust test accuracy with $>90\%$ robust training accuracy on CIFAR-10 dataset. Previously, there are other types of noise poisoned in the training data that have successfully fooled standard training ($15.8\%$ standard test accuracy with $99.9\%$ standard training accuracy on CIFAR-10 dataset), but their poisonings can be easily removed when adopting adversarial training. Therefore, we aim to design a new type of inducing noise, named ADVIN, which is an irremovable poisoning of training data. ADVIN can not only degrade the robustness of adversarial training by a large margin, for example, from $51.7\%$ to $0.57\%$ on CIFAR-10 dataset, but also be effective for fooling standard training ($13.1\%$ standard test accuracy with $100\%$ standard training accuracy). Additionally, ADVIN can be applied to preventing personal data (like selfies) from being exploited without authorization under whether standard or adversarial training.



## **7. Exposing Weaknesses of Malware Detectors with Explainability-Guided Evasion Attacks**

cs.CR

**SubmitDate**: 2021-11-19    [paper-pdf](http://arxiv.org/pdf/2111.10085v1)

**Authors**: Wei Wang, Ruoxi Sun, Tian Dong, Shaofeng Li, Minhui Xue, Gareth Tyson, Haojin Zhu

**Abstracts**: Numerous open-source and commercial malware detectors are available. However, the efficacy of these tools has been threatened by new adversarial attacks, whereby malware attempts to evade detection using, for example, machine learning techniques. In this work, we design an adversarial evasion attack that relies on both feature-space and problem-space manipulation. It uses explainability-guided feature selection to maximize evasion by identifying the most critical features that impact detection. We then use this attack as a benchmark to evaluate several state-of-the-art malware detectors. We find that (i) state-of-the-art malware detectors are vulnerable to even simple evasion strategies, and they can easily be tricked using off-the-shelf techniques; (ii) feature-space manipulation and problem-space obfuscation can be combined to enable evasion without needing white-box understanding of the detector; (iii) we can use explainability approaches (e.g., SHAP) to guide the feature manipulation and explain how attacks can transfer across multiple detectors. Our findings shed light on the weaknesses of current malware detectors, as well as how they can be improved.



## **8. Enhanced countering adversarial attacks via input denoising and feature restoring**

cs.CV

**SubmitDate**: 2021-11-19    [paper-pdf](http://arxiv.org/pdf/2111.10075v1)

**Authors**: Yanni Li, Wenhui Zhang, Jiawei Liu, Xiaoli Kou, Hui Li, Jiangtao Cui

**Abstracts**: Despite the fact that deep neural networks (DNNs) have achieved prominent performance in various applications, it is well known that DNNs are vulnerable to adversarial examples/samples (AEs) with imperceptible perturbations in clean/original samples. To overcome the weakness of the existing defense methods against adversarial attacks, which damages the information on the original samples, leading to the decrease of the target classifier accuracy, this paper presents an enhanced countering adversarial attack method IDFR (via Input Denoising and Feature Restoring). The proposed IDFR is made up of an enhanced input denoiser (ID) and a hidden lossy feature restorer (FR) based on the convex hull optimization. Extensive experiments conducted on benchmark datasets show that the proposed IDFR outperforms the various state-of-the-art defense methods, and is highly effective for protecting target models against various adversarial black-box or white-box attacks. \footnote{Souce code is released at: \href{https://github.com/ID-FR/IDFR}{https://github.com/ID-FR/IDFR}}



## **9. Towards Efficiently Evaluating the Robustness of Deep Neural Networks in IoT Systems: A GAN-based Method**

cs.LG

arXiv admin note: text overlap with arXiv:2002.02196

**SubmitDate**: 2021-11-19    [paper-pdf](http://arxiv.org/pdf/2111.10055v1)

**Authors**: Tao Bai, Jun Zhao, Jinlin Zhu, Shoudong Han, Jiefeng Chen, Bo Li, Alex Kot

**Abstracts**: Intelligent Internet of Things (IoT) systems based on deep neural networks (DNNs) have been widely deployed in the real world. However, DNNs are found to be vulnerable to adversarial examples, which raises people's concerns about intelligent IoT systems' reliability and security. Testing and evaluating the robustness of IoT systems becomes necessary and essential. Recently various attacks and strategies have been proposed, but the efficiency problem remains unsolved properly. Existing methods are either computationally extensive or time-consuming, which is not applicable in practice. In this paper, we propose a novel framework called Attack-Inspired GAN (AI-GAN) to generate adversarial examples conditionally. Once trained, it can generate adversarial perturbations efficiently given input images and target classes. We apply AI-GAN on different datasets in white-box settings, black-box settings and targeted models protected by state-of-the-art defenses. Through extensive experiments, AI-GAN achieves high attack success rates, outperforming existing methods, and reduces generation time significantly. Moreover, for the first time, AI-GAN successfully scales to complex datasets e.g. CIFAR-100 and ImageNet, with about $90\%$ success rates among all classes.



## **10. Generating Unrestricted 3D Adversarial Point Clouds**

cs.CV

**SubmitDate**: 2021-11-19    [paper-pdf](http://arxiv.org/pdf/2111.08973v2)

**Authors**: Xuelong Dai, Yanjie Li, Hua Dai, Bin Xiao

**Abstracts**: Utilizing 3D point cloud data has become an urgent need for the deployment of artificial intelligence in many areas like facial recognition and self-driving. However, deep learning for 3D point clouds is still vulnerable to adversarial attacks, e.g., iterative attacks, point transformation attacks, and generative attacks. These attacks need to restrict perturbations of adversarial examples within a strict bound, leading to the unrealistic adversarial 3D point clouds. In this paper, we propose an Adversarial Graph-Convolutional Generative Adversarial Network (AdvGCGAN) to generate visually realistic adversarial 3D point clouds from scratch. Specifically, we use a graph convolutional generator and a discriminator with an auxiliary classifier to generate realistic point clouds, which learn the latent distribution from the real 3D data. The unrestricted adversarial attack loss is incorporated in the special adversarial training of GAN, which enables the generator to generate the adversarial examples to spoof the target network. Compared with the existing state-of-art attack methods, the experiment results demonstrate the effectiveness of our unrestricted adversarial attack methods with a higher attack success rate and visual quality. Additionally, the proposed AdvGCGAN can achieve better performance against defense models and better transferability than existing attack methods with strong camouflage.



## **11. Arbitrarily Fast Switched Distributed Stabilization of Partially Unknown Interconnected Multiagent Systems: A Proactive Cyber Defense Perspective**

cs.SY

**SubmitDate**: 2021-11-19    [paper-pdf](http://arxiv.org/pdf/2110.14199v2)

**Authors**: Vahid Rezaei, Jafar Haadi Jafarian, Douglas C. Sicker

**Abstracts**: A design framework recently has been developed to stabilize interconnected multiagent systems in a distributed manner, and systematically capture the architectural aspect of cyber-physical systems. Such a control theoretic framework, however, results in a stabilization protocol which is passive with respect to the cyber attacks and conservative regarding the guaranteed level of resiliency. We treat the control layer topology and stabilization gains as the degrees of freedom, and develop a mixed control and cybersecurity design framework to address the above concerns. From a control perspective, despite the agent layer modeling uncertainties and perturbations, we propose a new step-by-step procedure to design a set of control sublayers for an arbitrarily fast switching of the control layer topology. From a proactive cyber defense perspective, we propose a satisfiability modulo theory formulation to obtain a set of control sublayer structures with security considerations, and offer a frequent and fast mutation of these sublayers such that the control layer topology will remain unpredictable for the adversaries. We prove the robust input-to-state stability of the two-layer interconnected multiagent system, and validate the proposed ideas in simulation.



## **12. TnT Attacks! Universal Naturalistic Adversarial Patches Against Deep Neural Network Systems**

cs.CV

We demonstrate physical deployments in multiple videos at  https://tntattacks.github.io/

**SubmitDate**: 2021-11-19    [paper-pdf](http://arxiv.org/pdf/2111.09999v1)

**Authors**: Bao Gia Doan, Minhui Xue, Shiqing Ma, Ehsan Abbasnejad, Damith C. Ranasinghe

**Abstracts**: Deep neural networks are vulnerable to attacks from adversarial inputs and, more recently, Trojans to misguide or hijack the decision of the model. We expose the existence of an intriguing class of bounded adversarial examples -- Universal NaTuralistic adversarial paTches -- we call TnTs, by exploring the superset of the bounded adversarial example space and the natural input space within generative adversarial networks. Now, an adversary can arm themselves with a patch that is naturalistic, less malicious-looking, physically realizable, highly effective -- achieving high attack success rates, and universal. A TnT is universal because any input image captured with a TnT in the scene will: i) misguide a network (untargeted attack); or ii) force the network to make a malicious decision (targeted attack). Interestingly, now, an adversarial patch attacker has the potential to exert a greater level of control -- the ability to choose a location independent, natural-looking patch as a trigger in contrast to being constrained to noisy perturbations -- an ability is thus far shown to be only possible with Trojan attack methods needing to interfere with the model building processes to embed a backdoor at the risk discovery; but, still realize a patch deployable in the physical world. Through extensive experiments on the large-scale visual classification task, ImageNet with evaluations across its entire validation set of 50,000 images, we demonstrate the realistic threat from TnTs and the robustness of the attack. We show a generalization of the attack to create patches achieving higher attack success rates than existing state-of-the-art methods. Our results show the generalizability of the attack to different visual classification tasks (CIFAR-10, GTSRB, PubFig) and multiple state-of-the-art deep neural networks such as WideResnet50, Inception-V3 and VGG-16.



## **13. Combinatorial Bandits under Strategic Manipulations**

cs.LG

**SubmitDate**: 2021-11-19    [paper-pdf](http://arxiv.org/pdf/2102.12722v4)

**Authors**: Jing Dong, Ke Li, Shuai Li, Baoxiang Wang

**Abstracts**: Strategic behavior against sequential learning methods, such as "click framing" in real recommendation systems, have been widely observed. Motivated by such behavior we study the problem of combinatorial multi-armed bandits (CMAB) under strategic manipulations of rewards, where each arm can modify the emitted reward signals for its own interest. This characterization of the adversarial behavior is a relaxation of previously well-studied settings such as adversarial attacks and adversarial corruption. We propose a strategic variant of the combinatorial UCB algorithm, which has a regret of at most $O(m\log T + m B_{max})$ under strategic manipulations, where $T$ is the time horizon, $m$ is the number of arms, and $B_{max}$ is the maximum budget of an arm. We provide lower bounds on the budget for arms to incur certain regret of the bandit algorithm. Extensive experiments on online worker selection for crowdsourcing systems, online influence maximization and online recommendations with both synthetic and real datasets corroborate our theoretical findings on robustness and regret bounds, in a variety of regimes of manipulation budgets.



## **14. A Review of Adversarial Attack and Defense for Classification Methods**

cs.CR

**SubmitDate**: 2021-11-18    [paper-pdf](http://arxiv.org/pdf/2111.09961v1)

**Authors**: Yao Li, Minhao Cheng, Cho-Jui Hsieh, Thomas C. M. Lee

**Abstracts**: Despite the efficiency and scalability of machine learning systems, recent studies have demonstrated that many classification methods, especially deep neural networks (DNNs), are vulnerable to adversarial examples; i.e., examples that are carefully crafted to fool a well-trained classification model while being indistinguishable from natural data to human. This makes it potentially unsafe to apply DNNs or related methods in security-critical areas. Since this issue was first identified by Biggio et al. (2013) and Szegedy et al.(2014), much work has been done in this field, including the development of attack methods to generate adversarial examples and the construction of defense techniques to guard against such examples. This paper aims to introduce this topic and its latest developments to the statistical community, primarily focusing on the generation and guarding of adversarial examples. Computing codes (in python and R) used in the numerical experiments are publicly available for readers to explore the surveyed methods. It is the hope of the authors that this paper will encourage more statisticians to work on this important and exciting field of generating and defending against adversarial examples.



## **15. Resilient Consensus-based Multi-agent Reinforcement Learning with Function Approximation**

cs.LG

**SubmitDate**: 2021-11-18    [paper-pdf](http://arxiv.org/pdf/2111.06776v2)

**Authors**: Martin Figura, Yixuan Lin, Ji Liu, Vijay Gupta

**Abstracts**: Adversarial attacks during training can strongly influence the performance of multi-agent reinforcement learning algorithms. It is, thus, highly desirable to augment existing algorithms such that the impact of adversarial attacks on cooperative networks is eliminated, or at least bounded. In this work, we consider a fully decentralized network, where each agent receives a local reward and observes the global state and action. We propose a resilient consensus-based actor-critic algorithm, whereby each agent estimates the team-average reward and value function, and communicates the associated parameter vectors to its immediate neighbors. We show that in the presence of Byzantine agents, whose estimation and communication strategies are completely arbitrary, the estimates of the cooperative agents converge to a bounded consensus value with probability one, provided that there are at most $H$ Byzantine agents in the neighborhood of each cooperative agent and the network is $(2H+1)$-robust. Furthermore, we prove that the policy of the cooperative agents converges with probability one to a bounded neighborhood around a local maximizer of their team-average objective function under the assumption that the policies of the adversarial agents asymptotically become stationary.



## **16. Robust Person Re-identification with Multi-Modal Joint Defence**

cs.CV

**SubmitDate**: 2021-11-18    [paper-pdf](http://arxiv.org/pdf/2111.09571v1)

**Authors**: Yunpeng Gong, Lifei Chen

**Abstracts**: The Person Re-identification (ReID) system based on metric learning has been proved to inherit the vulnerability of deep neural networks (DNNs), which are easy to be fooled by adversarail metric attacks. Existing work mainly relies on adversarial training for metric defense, and more methods have not been fully studied. By exploring the impact of attacks on the underlying features, we propose targeted methods for metric attacks and defence methods. In terms of metric attack, we use the local color deviation to construct the intra-class variation of the input to attack color features. In terms of metric defenses, we propose a joint defense method which includes two parts of proactive defense and passive defense. Proactive defense helps to enhance the robustness of the model to color variations and the learning of structure relations across multiple modalities by constructing different inputs from multimodal images, and passive defense exploits the invariance of structural features in a changing pixel space by circuitous scaling to preserve structural features while eliminating some of the adversarial noise. Extensive experiments demonstrate that the proposed joint defense compared with the existing adversarial metric defense methods which not only against multiple attacks at the same time but also has not significantly reduced the generalization capacity of the model. The code is available at https://github.com/finger-monkey/multi-modal_joint_defence.



## **17. DPA: Learning Robust Physical Adversarial Camouflages for Object Detectors**

cs.CV

**SubmitDate**: 2021-11-18    [paper-pdf](http://arxiv.org/pdf/2109.00124v2)

**Authors**: Yexin Duan, Jialin Chen, Xingyu Zhou, Junhua Zou, Zhengyun He, Wu Zhang, Jin Zhang, Zhisong Pan

**Abstracts**: Adversarial attacks are feasible in the real world for object detection. However, most of the previous works have tried to learn local "patches" applied to an object to fool detectors, which become less effective in squint view angles. To address this issue, we propose the Dense Proposals Attack (DPA) to learn one-piece, physical, and targeted adversarial camouflages for detectors. The camouflages are one-piece because they are generated as a whole for an object, physical because they remain adversarial when filmed under arbitrary viewpoints and different illumination conditions, and targeted because they can cause detectors to misidentify an object as a specific target class. In order to make the generated camouflages robust in the physical world, we introduce a combination of transformations to model the physical phenomena. In addition, to improve the attacks, DPA simultaneously attacks all the classifications in the fixed proposals. Moreover, we build a virtual 3D scene using the Unity simulation engine to fairly and reproducibly evaluate different physical attacks. Extensive experiments demonstrate that DPA outperforms the state-of-the-art methods, and it is generic for any object and generalized well to the real world, posing a potential threat to the security-critical computer vision systems.



## **18. Adversarial attacks on voter model dynamics in complex networks**

physics.soc-ph

6 pages, 4 figures

**SubmitDate**: 2021-11-18    [paper-pdf](http://arxiv.org/pdf/2111.09561v1)

**Authors**: Katsumi Chiyomaru, Kazuhiro Takemoto

**Abstracts**: This study investigates adversarial attacks conducted to distort the voter model dynamics in complex networks. Specifically, a simple adversarial attack method is proposed for holding the state of an individual's opinions closer to the target state in the voter model dynamics; the method shows that even when one opinion is the majority, the vote outcome can be inverted (i.e., the outcome can lean toward the other opinion) by adding extremely small (hard-to-detect) perturbations strategically generated in social networks. Adversarial attacks are relatively more effective for complex (large and dense) networks. The results indicate that opinion dynamics can be unknowingly distorted.



## **19. ZeBRA: Precisely Destroying Neural Networks with Zero-Data Based Repeated Bit Flip Attack**

cs.LG

14 pages, 3 figures, 5 tables, Accepted at British Machine Vision  Conference (BMVC) 2021

**SubmitDate**: 2021-11-18    [paper-pdf](http://arxiv.org/pdf/2111.01080v2)

**Authors**: Dahoon Park, Kon-Woo Kwon, Sunghoon Im, Jaeha Kung

**Abstracts**: In this paper, we present Zero-data Based Repeated bit flip Attack (ZeBRA) that precisely destroys deep neural networks (DNNs) by synthesizing its own attack datasets. Many prior works on adversarial weight attack require not only the weight parameters, but also the training or test dataset in searching vulnerable bits to be attacked. We propose to synthesize the attack dataset, named distilled target data, by utilizing the statistics of batch normalization layers in the victim DNN model. Equipped with the distilled target data, our ZeBRA algorithm can search vulnerable bits in the model without accessing training or test dataset. Thus, our approach makes the adversarial weight attack more fatal to the security of DNNs. Our experimental results show that 2.0x (CIFAR-10) and 1.6x (ImageNet) less number of bit flips are required on average to destroy DNNs compared to the previous attack method. Our code is available at https://github. com/pdh930105/ZeBRA.



## **20. Finding Optimal Tangent Points for Reducing Distortions of Hard-label Attacks**

cs.CV

accepted at NeurIPS 2021, including the appendix

**SubmitDate**: 2021-11-18    [paper-pdf](http://arxiv.org/pdf/2111.07492v2)

**Authors**: Chen Ma, Xiangyu Guo, Li Chen, Jun-Hai Yong, Yisen Wang

**Abstracts**: One major problem in black-box adversarial attacks is the high query complexity in the hard-label attack setting, where only the top-1 predicted label is available. In this paper, we propose a novel geometric-based approach called Tangent Attack (TA), which identifies an optimal tangent point of a virtual hemisphere located on the decision boundary to reduce the distortion of the attack. Assuming the decision boundary is locally flat, we theoretically prove that the minimum $\ell_2$ distortion can be obtained by reaching the decision boundary along the tangent line passing through such tangent point in each iteration. To improve the robustness of our method, we further propose a generalized method which replaces the hemisphere with a semi-ellipsoid to adapt to curved decision boundaries. Our approach is free of hyperparameters and pre-training. Extensive experiments conducted on the ImageNet and CIFAR-10 datasets demonstrate that our approach can consume only a small number of queries to achieve the low-magnitude distortion. The implementation source code is released online at https://github.com/machanic/TangentAttack.



## **21. Attacking Deep Learning AI Hardware with Universal Adversarial Perturbation**

cs.CR

**SubmitDate**: 2021-11-18    [paper-pdf](http://arxiv.org/pdf/2111.09488v1)

**Authors**: Mehdi Sadi, B. M. S. Bahar Talukder, Kaniz Mishty, Md Tauhidur Rahman

**Abstracts**: Universal Adversarial Perturbations are image-agnostic and model-independent noise that when added with any image can mislead the trained Deep Convolutional Neural Networks into the wrong prediction. Since these Universal Adversarial Perturbations can seriously jeopardize the security and integrity of practical Deep Learning applications, existing techniques use additional neural networks to detect the existence of these noises at the input image source. In this paper, we demonstrate an attack strategy that when activated by rogue means (e.g., malware, trojan) can bypass these existing countermeasures by augmenting the adversarial noise at the AI hardware accelerator stage. We demonstrate the accelerator-level universal adversarial noise attack on several deep Learning models using co-simulation of the software kernel of Conv2D function and the Verilog RTL model of the hardware under the FuseSoC environment.



## **22. Cortical Features for Defense Against Adversarial Audio Attacks**

cs.SD

Co-author legal name changed

**SubmitDate**: 2021-11-17    [paper-pdf](http://arxiv.org/pdf/2102.00313v2)

**Authors**: Ilya Kavalerov, Ruijie Zheng, Wojciech Czaja, Rama Chellappa

**Abstracts**: We propose using a computational model of the auditory cortex as a defense against adversarial attacks on audio. We apply several white-box iterative optimization-based adversarial attacks to an implementation of Amazon Alexa's HW network, and a modified version of this network with an integrated cortical representation, and show that the cortical features help defend against universal adversarial examples. At the same level of distortion, the adversarial noises found for the cortical network are always less effective for universal audio attacks. We make our code publicly available at https://github.com/ilyakava/py3fst.



## **23. Address Behaviour Vulnerabilities in the Next Generation of Autonomous Robots**

cs.RO

preprint and extended version of Nature Machine Intelligence, Vol 3,  November 2021, Pag 927-928

**SubmitDate**: 2021-11-17    [paper-pdf](http://arxiv.org/pdf/2103.13268v2)

**Authors**: Michele Colledanchise

**Abstracts**: Robots applications in our daily life increase at an unprecedented pace. As robots will soon operate "out in the wild", we must identify the safety and security vulnerabilities they will face. Robotics researchers and manufacturers focus their attention on new, cheaper, and more reliable applications. Still, they often disregard the operability in adversarial environments where a trusted or untrusted user can jeopardize or even alter the robot's task.   In this paper, we identify a new paradigm of security threats in the next generation of robots. These threats fall beyond the known hardware or network-based ones, and we must find new solutions to address them. These new threats include malicious use of the robot's privileged access, tampering with the robot sensors system, and tricking the robot's deliberation into harmful behaviors. We provide a taxonomy of attacks that exploit these vulnerabilities with realistic examples, and we outline effective countermeasures to prevent better, detect, and mitigate them.



## **24. Do Not Trust Prediction Scores for Membership Inference Attacks**

cs.LG

15 pages, 9 figures, 9 tables

**SubmitDate**: 2021-11-17    [paper-pdf](http://arxiv.org/pdf/2111.09076v1)

**Authors**: Dominik Hintersdorf, Lukas Struppek, Kristian Kersting

**Abstracts**: Membership inference attacks (MIAs) aim to determine whether a specific sample was used to train a predictive model. Knowing this may indeed lead to a privacy breach. Arguably, most MIAs, however, make use of the model's prediction scores - the probability of each output given some input - following the intuition that the trained model tends to behave differently on its training data. We argue that this is a fallacy for many modern deep network architectures, e.g., ReLU type neural networks produce almost always high prediction scores far away from the training data. Consequently, MIAs will miserably fail since this behavior leads to high false-positive rates not only on known domains but also on out-of-distribution data and implicitly acts as a defense against MIAs. Specifically, using generative adversarial networks, we are able to produce a potentially infinite number of samples falsely classified as part of the training data. In other words, the threat of MIAs is overestimated and less information is leaked than previously assumed. Moreover, there is actually a trade-off between the overconfidence of classifiers and their susceptibility to MIAs: the more classifiers know when they do not know, making low confidence predictions far away from the training data, the more they reveal the training data.



## **25. TraSw: Tracklet-Switch Adversarial Attacks against Multi-Object Tracking**

cs.CV

**SubmitDate**: 2021-11-17    [paper-pdf](http://arxiv.org/pdf/2111.08954v1)

**Authors**: Delv Lin, Qi Chen, Chengyu Zhou, Kun He

**Abstracts**: Benefiting from the development of Deep Neural Networks, Multi-Object Tracking (MOT) has achieved aggressive progress. Currently, the real-time Joint-Detection-Tracking (JDT) based MOT trackers gain increasing attention and derive many excellent models. However, the robustness of JDT trackers is rarely studied, and it is challenging to attack the MOT system since its mature association algorithms are designed to be robust against errors during tracking. In this work, we analyze the weakness of JDT trackers and propose a novel adversarial attack method, called Tracklet-Switch (TraSw), against the complete tracking pipeline of MOT. Specifically, a push-pull loss and a center leaping optimization are designed to generate adversarial examples for both re-ID feature and object detection. TraSw can fool the tracker to fail to track the targets in the subsequent frames by attacking very few frames. We evaluate our method on the advanced deep trackers (i.e., FairMOT, JDE, ByteTrack) using the MOT-Challenge datasets (i.e., 2DMOT15, MOT17, and MOT20). Experiments show that TraSw can achieve a high success rate of over 95% by attacking only five frames on average for the single-target attack and a reasonably high success rate of over 80% for the multiple-target attack. The code is available at https://github.com/DerryHub/FairMOT-attack .



## **26. Turning Your Strength against You: Detecting and Mitigating Robust and Universal Adversarial Patch Attacks**

cs.CR

**SubmitDate**: 2021-11-17    [paper-pdf](http://arxiv.org/pdf/2108.05075v2)

**Authors**: Zitao Chen, Pritam Dash, Karthik Pattabiraman

**Abstracts**: Adversarial patch attacks against image classification deep neural networks (DNNs), which inject arbitrary distortions within a bounded region of an image, can generate adversarial perturbations that are robust (i.e., remain adversarial in physical world) and universal (i.e., remain adversarial on any input). Such attacks can lead to severe consequences in real-world DNN-based systems.   This work proposes Jujutsu, a technique to detect and mitigate robust and universal adversarial patch attacks. For detection, Jujutsu exploits the attacks' universal property - Jujutsu first locates the region of the potential adversarial patch, and then strategically transfers it to a dedicated region in a new image to determine whether it is truly malicious. For attack mitigation, Jujutsu leverages the attacks' localized nature via image inpainting to synthesize the semantic contents in the pixels that are corrupted by the attacks, and reconstruct the ``clean'' image.   We evaluate Jujutsu on four diverse datasets (ImageNet, ImageNette, CelebA and Place365), and show that Jujutsu achieves superior performance and significantly outperforms existing techniques. We find that Jujutsu can further defend against different variants of the basic attack, including 1) physical-world attack; 2) attacks that target diverse classes; 3) attacks that construct patches in different shapes and 4) adaptive attacks.



## **27. Detecting AutoAttack Perturbations in the Frequency Domain**

cs.CV

**SubmitDate**: 2021-11-16    [paper-pdf](http://arxiv.org/pdf/2111.08785v1)

**Authors**: Peter Lorenz, Paula Harder, Dominik Strassel, Margret Keuper, Janis Keuper

**Abstracts**: Recently, adversarial attacks on image classification networks by the AutoAttack (Croce and Hein, 2020b) framework have drawn a lot of attention. While AutoAttack has shown a very high attack success rate, most defense approaches are focusing on network hardening and robustness enhancements, like adversarial training. This way, the currently best-reported method can withstand about 66% of adversarial examples on CIFAR10. In this paper, we investigate the spatial and frequency domain properties of AutoAttack and propose an alternative defense. Instead of hardening a network, we detect adversarial attacks during inference, rejecting manipulated inputs. Based on a rather simple and fast analysis in the frequency domain, we introduce two different detection algorithms. First, a black box detector that only operates on the input images and achieves a detection accuracy of 100% on the AutoAttack CIFAR10 benchmark and 99.3% on ImageNet, for epsilon = 8/255 in both cases. Second, a whitebox detector using an analysis of CNN feature maps, leading to a detection rate of also 100% and 98.7% on the same benchmarks.



## **28. Robustness of Bayesian Neural Networks to White-Box Adversarial Attacks**

cs.LG

Accepted at the fourth IEEE International Conference on Artificial  Intelligence and Knowledge Engineering (AIKE 2021)

**SubmitDate**: 2021-11-16    [paper-pdf](http://arxiv.org/pdf/2111.08591v1)

**Authors**: Adaku Uchendu, Daniel Campoy, Christopher Menart, Alexandra Hildenbrandt

**Abstracts**: Bayesian Neural Networks (BNNs), unlike Traditional Neural Networks (TNNs) are robust and adept at handling adversarial attacks by incorporating randomness. This randomness improves the estimation of uncertainty, a feature lacking in TNNs. Thus, we investigate the robustness of BNNs to white-box attacks using multiple Bayesian neural architectures. Furthermore, we create our BNN model, called BNN-DenseNet, by fusing Bayesian inference (i.e., variational Bayes) to the DenseNet architecture, and BDAV, by combining this intervention with adversarial training. Experiments are conducted on the CIFAR-10 and FGVC-Aircraft datasets. We attack our models with strong white-box attacks ($l_\infty$-FGSM, $l_\infty$-PGD, $l_2$-PGD, EOT $l_\infty$-FGSM, and EOT $l_\infty$-PGD). In all experiments, at least one BNN outperforms traditional neural networks during adversarial attack scenarios. An adversarially-trained BNN outperforms its non-Bayesian, adversarially-trained counterpart in most experiments, and often by significant margins. Lastly, we investigate network calibration and find that BNNs do not make overconfident predictions, providing evidence that BNNs are also better at measuring uncertainty.



## **29. Improving the robustness and accuracy of biomedical language models through adversarial training**

cs.CL

**SubmitDate**: 2021-11-16    [paper-pdf](http://arxiv.org/pdf/2111.08529v1)

**Authors**: Milad Moradi, Matthias Samwald

**Abstracts**: Deep transformer neural network models have improved the predictive accuracy of intelligent text processing systems in the biomedical domain. They have obtained state-of-the-art performance scores on a wide variety of biomedical and clinical Natural Language Processing (NLP) benchmarks. However, the robustness and reliability of these models has been less explored so far. Neural NLP models can be easily fooled by adversarial samples, i.e. minor changes to input that preserve the meaning and understandability of the text but force the NLP system to make erroneous decisions. This raises serious concerns about the security and trust-worthiness of biomedical NLP systems, especially when they are intended to be deployed in real-world use cases. We investigated the robustness of several transformer neural language models, i.e. BioBERT, SciBERT, BioMed-RoBERTa, and Bio-ClinicalBERT, on a wide range of biomedical and clinical text processing tasks. We implemented various adversarial attack methods to test the NLP systems in different attack scenarios. Experimental results showed that the biomedical NLP models are sensitive to adversarial samples; their performance dropped in average by 21 and 18.9 absolute percent on character-level and word-level adversarial noise, respectively. Conducting extensive adversarial training experiments, we fine-tuned the NLP models on a mixture of clean samples and adversarial inputs. Results showed that adversarial training is an effective defense mechanism against adversarial noise; the models robustness improved in average by 11.3 absolute percent. In addition, the models performance on clean data increased in average by 2.4 absolute present, demonstrating that adversarial training can boost generalization abilities of biomedical NLP systems.



## **30. Consistent Semantic Attacks on Optical Flow**

cs.CV

Paper and supplementary material

**SubmitDate**: 2021-11-16    [paper-pdf](http://arxiv.org/pdf/2111.08485v1)

**Authors**: Tom Koren, Lior Talker, Michael Dinerstein, Roy J Jevnisek

**Abstracts**: We present a novel approach for semantically targeted adversarial attacks on Optical Flow. In such attacks the goal is to corrupt the flow predictions of a specific object category or instance. Usually, an attacker seeks to hide the adversarial perturbations in the input. However, a quick scan of the output reveals the attack. In contrast, our method helps to hide the attackers intent in the output as well. We achieve this thanks to a regularization term that encourages off-target consistency. We perform extensive tests on leading optical flow models to demonstrate the benefits of our approach in both white-box and black-box settings. Also, we demonstrate the effectiveness of our attack on subsequent tasks that depend on the optical flow.



## **31. Meta-Learning the Search Distribution of Black-Box Random Search Based Adversarial Attacks**

cs.LG

accepted at NeurIPS 2021; updated the numbers in Table 5 and added  references

**SubmitDate**: 2021-11-16    [paper-pdf](http://arxiv.org/pdf/2111.01714v2)

**Authors**: Maksym Yatsura, Jan Hendrik Metzen, Matthias Hein

**Abstracts**: Adversarial attacks based on randomized search schemes have obtained state-of-the-art results in black-box robustness evaluation recently. However, as we demonstrate in this work, their efficiency in different query budget regimes depends on manual design and heuristic tuning of the underlying proposal distributions. We study how this issue can be addressed by adapting the proposal distribution online based on the information obtained during the attack. We consider Square Attack, which is a state-of-the-art score-based black-box attack, and demonstrate how its performance can be improved by a learned controller that adjusts the parameters of the proposal distribution online during the attack. We train the controller using gradient-based end-to-end training on a CIFAR10 model with white box access. We demonstrate that plugging the learned controller into the attack consistently improves its black-box robustness estimate in different query regimes by up to 20% for a wide range of different models with black-box access. We further show that the learned adaptation principle transfers well to the other data distributions such as CIFAR100 or ImageNet and to the targeted attack setting.



## **32. Bridge the Gap Between CV and NLP! A Gradient-based Textual Adversarial Attack Framework**

cs.CL

Work on progress

**SubmitDate**: 2021-11-16    [paper-pdf](http://arxiv.org/pdf/2110.15317v2)

**Authors**: Lifan Yuan, Yichi Zhang, Yangyi Chen, Wei Wei

**Abstracts**: Despite great success on many machine learning tasks, deep neural networks are still vulnerable to adversarial samples. While gradient-based adversarial attack methods are well-explored in the field of computer vision, it is impractical to directly apply them in natural language processing due to the discrete nature of text. To bridge this gap, we propose a general framework to adapt existing gradient-based methods to craft textual adversarial samples. In this framework, gradient-based continuous perturbations are added to the embedding layer and are amplified in the forward propagation process. Then the final perturbed latent representations are decoded with a mask language model head to obtain potential adversarial samples. In this paper, we instantiate our framework with \textbf{T}extual \textbf{P}rojected \textbf{G}radient \textbf{D}escent (\textbf{TPGD}). We conduct comprehensive experiments to evaluate our framework by performing transfer black-box attacks on BERT, RoBERTa and ALBERT on three benchmark datasets. Experimental results demonstrate our method achieves an overall better performance and produces more fluent and grammatical adversarial samples compared to strong baseline methods. All the code and data will be made public.



## **33. InFlow: Robust outlier detection utilizing Normalizing Flows**

cs.LG

**SubmitDate**: 2021-11-16    [paper-pdf](http://arxiv.org/pdf/2106.12894v2)

**Authors**: Nishant Kumar, Pia Hanfeld, Michael Hecht, Michael Bussmann, Stefan Gumhold, Nico Hoffmann

**Abstracts**: Normalizing flows are prominent deep generative models that provide tractable probability distributions and efficient density estimation. However, they are well known to fail while detecting Out-of-Distribution (OOD) inputs as they directly encode the local features of the input representations in their latent space. In this paper, we solve this overconfidence issue of normalizing flows by demonstrating that flows, if extended by an attention mechanism, can reliably detect outliers including adversarial attacks. Our approach does not require outlier data for training and we showcase the efficiency of our method for OOD detection by reporting state-of-the-art performance in diverse experimental settings. Code available at https://github.com/ComputationalRadiationPhysics/InFlow .



## **34. TSS: Transformation-Specific Smoothing for Robustness Certification**

cs.LG

2021 ACM SIGSAC Conference on Computer and Communications Security  (CCS '21)

**SubmitDate**: 2021-11-16    [paper-pdf](http://arxiv.org/pdf/2002.12398v5)

**Authors**: Linyi Li, Maurice Weber, Xiaojun Xu, Luka Rimanic, Bhavya Kailkhura, Tao Xie, Ce Zhang, Bo Li

**Abstracts**: As machine learning (ML) systems become pervasive, safeguarding their security is critical. However, recently it has been demonstrated that motivated adversaries are able to mislead ML systems by perturbing test data using semantic transformations. While there exists a rich body of research providing provable robustness guarantees for ML models against $\ell_p$ norm bounded adversarial perturbations, guarantees against semantic perturbations remain largely underexplored. In this paper, we provide TSS -- a unified framework for certifying ML robustness against general adversarial semantic transformations. First, depending on the properties of each transformation, we divide common transformations into two categories, namely resolvable (e.g., Gaussian blur) and differentially resolvable (e.g., rotation) transformations. For the former, we propose transformation-specific randomized smoothing strategies and obtain strong robustness certification. The latter category covers transformations that involve interpolation errors, and we propose a novel approach based on stratified sampling to certify the robustness. Our framework TSS leverages these certification strategies and combines with consistency-enhanced training to provide rigorous certification of robustness. We conduct extensive experiments on over ten types of challenging semantic transformations and show that TSS significantly outperforms the state of the art. Moreover, to the best of our knowledge, TSS is the first approach that achieves nontrivial certified robustness on the large-scale ImageNet dataset. For instance, our framework achieves 30.4% certified robust accuracy against rotation attack (within $\pm 30^\circ$) on ImageNet. Moreover, to consider a broader range of transformations, we show TSS is also robust against adaptive attacks and unforeseen image corruptions such as CIFAR-10-C and ImageNet-C.



## **35. Android HIV: A Study of Repackaging Malware for Evading Machine-Learning Detection**

cs.CR

14 pages, 11 figures

**SubmitDate**: 2021-11-16    [paper-pdf](http://arxiv.org/pdf/1808.04218v4)

**Authors**: Xiao Chen, Chaoran Li, Derui Wang, Sheng Wen, Jun Zhang, Surya Nepal, Yang Xiang, Kui Ren

**Abstracts**: Machine learning based solutions have been successfully employed for automatic detection of malware on Android. However, machine learning models lack robustness to adversarial examples, which are crafted by adding carefully chosen perturbations to the normal inputs. So far, the adversarial examples can only deceive detectors that rely on syntactic features (e.g., requested permissions, API calls, etc), and the perturbations can only be implemented by simply modifying application's manifest. While recent Android malware detectors rely more on semantic features from Dalvik bytecode rather than manifest, existing attacking/defending methods are no longer effective. In this paper, we introduce a new attacking method that generates adversarial examples of Android malware and evades being detected by the current models. To this end, we propose a method of applying optimal perturbations onto Android APK that can successfully deceive the machine learning detectors. We develop an automated tool to generate the adversarial examples without human intervention. In contrast to existing works, the adversarial examples crafted by our method can also deceive recent machine learning based detectors that rely on semantic features such as control-flow-graph. The perturbations can also be implemented directly onto APK's Dalvik bytecode rather than Android manifest to evade from recent detectors. We demonstrate our attack on two state-of-the-art Android malware detection schemes, MaMaDroid and Drebin. Our results show that the malware detection rates decreased from 96% to 0% in MaMaDroid, and from 97% to 0% in Drebin, with just a small number of codes to be inserted into the APK.



## **36. A Survey on Adversarial Attacks for Malware Analysis**

cs.CR

42 Pages, 31 Figures, 11 Tables

**SubmitDate**: 2021-11-16    [paper-pdf](http://arxiv.org/pdf/2111.08223v1)

**Authors**: Kshitiz Aryal, Maanak Gupta, Mahmoud Abdelsalam

**Abstracts**: Machine learning has witnessed tremendous growth in its adoption and advancement in the last decade. The evolution of machine learning from traditional algorithms to modern deep learning architectures has shaped the way today's technology functions. Its unprecedented ability to discover knowledge/patterns from unstructured data and automate the decision-making process led to its application in wide domains. High flying machine learning arena has been recently pegged back by the introduction of adversarial attacks. Adversaries are able to modify data, maximizing the classification error of the models. The discovery of blind spots in machine learning models has been exploited by adversarial attackers by generating subtle intentional perturbations in test samples. Increasing dependency on data has paved the blueprint for ever-high incentives to camouflage machine learning models. To cope with probable catastrophic consequences in the future, continuous research is required to find vulnerabilities in form of adversarial and design remedies in systems. This survey aims at providing the encyclopedic introduction to adversarial attacks that are carried out against malware detection systems. The paper will introduce various machine learning techniques used to generate adversarial and explain the structure of target files. The survey will also model the threat posed by the adversary and followed by brief descriptions of widely accepted adversarial algorithms. Work will provide a taxonomy of adversarial evasion attacks on the basis of attack domain and adversarial generation techniques. Adversarial evasion attacks carried out against malware detectors will be discussed briefly under each taxonomical headings and compared with concomitant researches. Analyzing the current research challenges in an adversarial generation, the survey will conclude by pinpointing the open future research directions.



## **37. FedCG: Leverage Conditional GAN for Protecting Privacy and Maintaining Competitive Performance in Federated Learning**

cs.LG

**SubmitDate**: 2021-11-16    [paper-pdf](http://arxiv.org/pdf/2111.08211v1)

**Authors**: Yuezhou Wu, Yan Kang, Jiahuan Luo, Yuanqin He, Qiang Yang

**Abstracts**: Federated learning (FL) aims to protect data privacy by enabling clients to collaboratively build machine learning models without sharing their private data. However, recent works demonstrate that FL is vulnerable to gradient-based data recovery attacks. Varieties of privacy-preserving technologies have been leveraged to further enhance the privacy of FL. Nonetheless, they either are computational or communication expensive (e.g., homomorphic encryption) or suffer from precision loss (e.g., differential privacy). In this work, we propose \textsc{FedCG}, a novel \underline{fed}erated learning method that leverages \underline{c}onditional \underline{g}enerative adversarial networks to achieve high-level privacy protection while still maintaining competitive model performance. More specifically, \textsc{FedCG} decomposes each client's local network into a private extractor and a public classifier and keeps the extractor local to protect privacy. Instead of exposing extractors which is the culprit of privacy leakage, \textsc{FedCG} shares clients' generators with the server for aggregating common knowledge aiming to enhance the performance of clients' local networks. Extensive experiments demonstrate that \textsc{FedCG} can achieve competitive model performance compared with baseline FL methods, and numerical privacy analysis shows that \textsc{FedCG} has high-level privacy-preserving capability.



## **38. 3D Adversarial Attacks Beyond Point Cloud**

cs.CV

8 pages, 6 figs

**SubmitDate**: 2021-11-16    [paper-pdf](http://arxiv.org/pdf/2104.12146v3)

**Authors**: Jinlai Zhang, Lyujie Chen, Binbin Liu, Bo Ouyang, Qizhi Xie, Jihong Zhu, Weiming Li, Yanmei Meng

**Abstracts**: Recently, 3D deep learning models have been shown to be susceptible to adversarial attacks like their 2D counterparts. Most of the state-of-the-art (SOTA) 3D adversarial attacks perform perturbation to 3D point clouds. To reproduce these attacks in the physical scenario, a generated adversarial 3D point cloud need to be reconstructed to mesh, which leads to a significant drop in its adversarial effect. In this paper, we propose a strong 3D adversarial attack named Mesh Attack to address this problem by directly performing perturbation on mesh of a 3D object. In order to take advantage of the most effective gradient-based attack, a differentiable sample module that back-propagate the gradient of point cloud to mesh is introduced. To further ensure the adversarial mesh examples without outlier and 3D printable, three mesh losses are adopted. Extensive experiments demonstrate that the proposed scheme outperforms SOTA 3D attacks by a significant margin. We also achieved SOTA performance under various defenses. Our code is available at: https://github.com/cuge1995/Mesh-Attack.



## **39. Augmenting Zero Trust Architecture to Endpoints Using Blockchain: A State-of-The-Art Review**

cs.CR

(1) Fixed the reference numbering (2) Fixed syntax errors,  improvements (3) document re-structured

**SubmitDate**: 2021-11-15    [paper-pdf](http://arxiv.org/pdf/2104.00460v4)

**Authors**: Lampis Alevizos, Vinh Thong Ta, Max Hashem Eiza

**Abstracts**: With the purpose of defending against lateral movement in today's borderless networks, Zero Trust Architecture (ZTA) adoption is gaining momentum. With a full scale ZTA implementation, it is unlikely that adversaries will be able to spread through the network starting from a compromised endpoint. However, the already authenticated and authorised session of a compromised endpoint can be leveraged to perform limited, though malicious, activities ultimately rendering the endpoints the Achilles heel of ZTA. To effectively detect such attacks, distributed collaborative intrusion detection systems with an attack scenario-based approach have been developed. Nonetheless, Advanced Persistent Threats (APTs) have demonstrated their ability to bypass this approach with a high success ratio. As a result, adversaries can pass undetected or potentially alter the detection logging mechanisms to achieve a stealthy presence. Recently, blockchain technology has demonstrated solid use cases in the cyber security domain. In this paper, motivated by the convergence of ZTA and blockchain-based intrusion detection and prevention, we examine how ZTA can be augmented onto endpoints. Namely, we perform a state-of-the-art review of ZTA models, real-world architectures with a focus on endpoints, and blockchain-based intrusion detection systems. We discuss the potential of blockchain's immutability fortifying the detection process and identify open challenges as well as potential solutions and future directions.



## **40. NNoculation: Catching BadNets in the Wild**

cs.CR

**SubmitDate**: 2021-11-15    [paper-pdf](http://arxiv.org/pdf/2002.08313v2)

**Authors**: Akshaj Kumar Veldanda, Kang Liu, Benjamin Tan, Prashanth Krishnamurthy, Farshad Khorrami, Ramesh Karri, Brendan Dolan-Gavitt, Siddharth Garg

**Abstracts**: This paper proposes a novel two-stage defense (NNoculation) against backdoored neural networks (BadNets) that, repairs a BadNet both pre-deployment and online in response to backdoored test inputs encountered in the field. In the pre-deployment stage, NNoculation retrains the BadNet with random perturbations of clean validation inputs to partially reduce the adversarial impact of a backdoor. Post-deployment, NNoculation detects and quarantines backdoored test inputs by recording disagreements between the original and pre-deployment patched networks. A CycleGAN is then trained to learn transformations between clean validation and quarantined inputs; i.e., it learns to add triggers to clean validation images. Backdoored validation images along with their correct labels are used to further retrain the pre-deployment patched network, yielding our final defense. Empirical evaluation on a comprehensive suite of backdoor attacks show that NNoculation outperforms all state-of-the-art defenses that make restrictive assumptions and only work on specific backdoor attacks, or fail on adaptive attacks. In contrast, NNoculation makes minimal assumptions and provides an effective defense, even under settings where existing defenses are ineffective due to attackers circumventing their restrictive assumptions.



## **41. Generative Dynamic Patch Attack**

cs.CV

Published as a conference paper at BMVC 2021

**SubmitDate**: 2021-11-15    [paper-pdf](http://arxiv.org/pdf/2111.04266v2)

**Authors**: Xiang Li, Shihao Ji

**Abstracts**: Adversarial patch attack is a family of attack algorithms that perturb a part of image to fool a deep neural network model. Existing patch attacks mostly consider injecting adversarial patches at input-agnostic locations: either a predefined location or a random location. This attack setup may be sufficient for attack but has considerable limitations when using it for adversarial training. Thus, robust models trained with existing patch attacks cannot effectively defend other adversarial attacks. In this paper, we first propose an end-to-end patch attack algorithm, Generative Dynamic Patch Attack (GDPA), which generates both patch pattern and patch location adversarially for each input image. We show that GDPA is a generic attack framework that can produce dynamic/static and visible/invisible patches with a few configuration changes. Secondly, GDPA can be readily integrated for adversarial training to improve model robustness to various adversarial attacks. Extensive experiments on VGGFace, Traffic Sign and ImageNet show that GDPA achieves higher attack success rates than state-of-the-art patch attacks, while adversarially trained model with GDPA demonstrates superior robustness to adversarial patch attacks than competing methods. Our source code can be found at https://github.com/lxuniverse/gdpa.



## **42. Website fingerprinting on early QUIC traffic**

cs.CR

This work has been accepted by Elsevier Computer Networks for  publication

**SubmitDate**: 2021-11-15    [paper-pdf](http://arxiv.org/pdf/2101.11871v2)

**Authors**: Pengwei Zhan, Liming Wang, Yi Tang

**Abstracts**: Cryptographic protocols have been widely used to protect the user's privacy and avoid exposing private information. QUIC (Quick UDP Internet Connections), including the version originally designed by Google (GQUIC) and the version standardized by IETF (IQUIC), as alternatives to the traditional HTTP, demonstrate their unique transmission characteristics: based on UDP for encrypted resource transmitting, accelerating web page rendering. However, existing encrypted transmission schemes based on TCP are vulnerable to website fingerprinting (WFP) attacks, allowing adversaries to infer the users' visited websites by eavesdropping on the transmission channel. Whether GQUIC and IQUIC can effectively resist such attacks is worth investigating. In this paper, we study the vulnerabilities of GQUIC, IQUIC, and HTTPS to WFP attacks from the perspective of traffic analysis. Extensive experiments show that, in the early traffic scenario, GQUIC is the most vulnerable to WFP attacks among GQUIC, IQUIC, and HTTPS, while IQUIC is more vulnerable than HTTPS, but the vulnerability of the three protocols is similar in the normal full traffic scenario. Features transferring analysis shows that most features are transferable between protocols when on normal full traffic scenario. However, combining with the qualitative analysis of latent feature representation, we find that the transferring is inefficient when on early traffic, as GQUIC, IQUIC, and HTTPS show the significantly different magnitude of variation in the traffic distribution on early traffic. By upgrading the one-time WFP attacks to multiple WFP Top-a attacks, we find that the attack accuracy on GQUIC and IQUIC reach 95.4% and 95.5%, respectively, with only 40 packets and just using simple features, whereas reach only 60.7% when on HTTPS. We also demonstrate that the vulnerability of IQUIC is only slightly dependent on the network environment.



## **43. Adversarial Detection Avoidance Attacks: Evaluating the robustness of perceptual hashing-based client-side scanning**

cs.CR

**SubmitDate**: 2021-11-15    [paper-pdf](http://arxiv.org/pdf/2106.09820v2)

**Authors**: Shubham Jain, Ana-Maria Cretu, Yves-Alexandre de Montjoye

**Abstracts**: End-to-end encryption (E2EE) by messaging platforms enable people to securely and privately communicate with one another. Its widespread adoption however raised concerns that illegal content might now be shared undetected. Following the global pushback against key escrow systems, client-side scanning based on perceptual hashing has been recently proposed by tech companies, governments and researchers to detect illegal content in E2EE communications. We here propose the first framework to evaluate the robustness of perceptual hashing-based client-side scanning to detection avoidance attacks and show current systems to not be robust. More specifically, we propose three adversarial attacks--a general black-box attack and two white-box attacks for discrete cosine transform-based algorithms--against perceptual hashing algorithms. In a large-scale evaluation, we show perceptual hashing-based client-side scanning mechanisms to be highly vulnerable to detection avoidance attacks in a black-box setting, with more than 99.9\% of images successfully attacked while preserving the content of the image. We furthermore show our attack to generate diverse perturbations, strongly suggesting that straightforward mitigation strategies would be ineffective. Finally, we show that the larger thresholds necessary to make the attack harder would probably require more than one billion images to be flagged and decrypted daily, raising strong privacy concerns. Taken together, our results shed serious doubts on the robustness of perceptual hashing-based client-side scanning mechanisms currently proposed by governments, organizations, and researchers around the world.



## **44. Property Inference Attacks Against GANs**

cs.CR

To Appear in NDSS 2022

**SubmitDate**: 2021-11-15    [paper-pdf](http://arxiv.org/pdf/2111.07608v1)

**Authors**: Junhao Zhou, Yufei Chen, Chao Shen, Yang Zhang

**Abstracts**: While machine learning (ML) has made tremendous progress during the past decade, recent research has shown that ML models are vulnerable to various security and privacy attacks. So far, most of the attacks in this field focus on discriminative models, represented by classifiers. Meanwhile, little attention has been paid to the security and privacy risks of generative models, such as generative adversarial networks (GANs). In this paper, we propose the first set of training dataset property inference attacks against GANs. Concretely, the adversary aims to infer the macro-level training dataset property, i.e., the proportion of samples used to train a target GAN with respect to a certain attribute. A successful property inference attack can allow the adversary to gain extra knowledge of the target GAN's training dataset, thereby directly violating the intellectual property of the target model owner. Also, it can be used as a fairness auditor to check whether the target GAN is trained with a biased dataset. Besides, property inference can serve as a building block for other advanced attacks, such as membership inference. We propose a general attack pipeline that can be tailored to two attack scenarios, including the full black-box setting and partial black-box setting. For the latter, we introduce a novel optimization framework to increase the attack efficacy. Extensive experiments over four representative GAN models on five property inference tasks show that our attacks achieve strong performance. In addition, we show that our attacks can be used to enhance the performance of membership inference against GANs.



## **45. Towards Interpretability of Speech Pause in Dementia Detection using Adversarial Learning**

cs.CL

**SubmitDate**: 2021-11-14    [paper-pdf](http://arxiv.org/pdf/2111.07454v1)

**Authors**: Youxiang Zhu, Bang Tran, Xiaohui Liang, John A. Batsis, Robert M. Roth

**Abstracts**: Speech pause is an effective biomarker in dementia detection. Recent deep learning models have exploited speech pauses to achieve highly accurate dementia detection, but have not exploited the interpretability of speech pauses, i.e., what and how positions and lengths of speech pauses affect the result of dementia detection. In this paper, we will study the positions and lengths of dementia-sensitive pauses using adversarial learning approaches. Specifically, we first utilize an adversarial attack approach by adding the perturbation to the speech pauses of the testing samples, aiming to reduce the confidence levels of the detection model. Then, we apply an adversarial training approach to evaluate the impact of the perturbation in training samples on the detection model. We examine the interpretability from the perspectives of model accuracy, pause context, and pause length. We found that some pauses are more sensitive to dementia than other pauses from the model's perspective, e.g., speech pauses near to the verb "is". Increasing lengths of sensitive pauses or adding sensitive pauses leads the model inference to Alzheimer's Disease, while decreasing the lengths of sensitive pauses or deleting sensitive pauses leads to non-AD.



## **46. Generating Band-Limited Adversarial Surfaces Using Neural Networks**

cs.CV

**SubmitDate**: 2021-11-14    [paper-pdf](http://arxiv.org/pdf/2111.07424v1)

**Authors**: Roee Ben Shlomo, Yevgeniy Men, Ido Imanuel

**Abstracts**: Generating adversarial examples is the art of creating a noise that is added to an input signal of a classifying neural network, and thus changing the network's classification, while keeping the noise as tenuous as possible. While the subject is well-researched in the 2D regime, it is lagging behind in the 3D regime, i.e. attacking a classifying network that works on 3D point-clouds or meshes and, for example, classifies the pose of people's 3D scans. As of now, the vast majority of papers that describe adversarial attacks in this regime work by methods of optimization. In this technical report we suggest a neural network that generates the attacks. This network utilizes PointNet's architecture with some alterations. While the previous articles on which we based our work on have to optimize each shape separately, i.e. tailor an attack from scratch for each individual input without any learning, we attempt to create a unified model that can deduce the needed adversarial example with a single forward run.



## **47. Measuring the Contribution of Multiple Model Representations in Detecting Adversarial Instances**

cs.LG

**SubmitDate**: 2021-11-13    [paper-pdf](http://arxiv.org/pdf/2111.07035v1)

**Authors**: Daniel Steinberg, Paul Munro

**Abstracts**: Deep learning models have been used for a wide variety of tasks. They are prevalent in computer vision, natural language processing, speech recognition, and other areas. While these models have worked well under many scenarios, it has been shown that they are vulnerable to adversarial attacks. This has led to a proliferation of research into ways that such attacks could be identified and/or defended against. Our goal is to explore the contribution that can be attributed to using multiple underlying models for the purpose of adversarial instance detection. Our paper describes two approaches that incorporate representations from multiple models for detecting adversarial examples. We devise controlled experiments for measuring the detection impact of incrementally utilizing additional models. For many of the scenarios we consider, the results show that performance increases with the number of underlying models used for extracting representations.



## **48. Adversarially Robust Learning for Security-Constrained Optimal Power Flow**

math.OC

Accepted at Neural Information Processing Systems (NeurIPS) 2021

**SubmitDate**: 2021-11-12    [paper-pdf](http://arxiv.org/pdf/2111.06961v1)

**Authors**: Priya L. Donti, Aayushya Agarwal, Neeraj Vijay Bedmutha, Larry Pileggi, J. Zico Kolter

**Abstracts**: In recent years, the ML community has seen surges of interest in both adversarially robust learning and implicit layers, but connections between these two areas have seldom been explored. In this work, we combine innovations from these areas to tackle the problem of N-k security-constrained optimal power flow (SCOPF). N-k SCOPF is a core problem for the operation of electrical grids, and aims to schedule power generation in a manner that is robust to potentially k simultaneous equipment outages. Inspired by methods in adversarially robust training, we frame N-k SCOPF as a minimax optimization problem - viewing power generation settings as adjustable parameters and equipment outages as (adversarial) attacks - and solve this problem via gradient-based techniques. The loss function of this minimax problem involves resolving implicit equations representing grid physics and operational decisions, which we differentiate through via the implicit function theorem. We demonstrate the efficacy of our framework in solving N-3 SCOPF, which has traditionally been considered as prohibitively expensive to solve given that the problem size depends combinatorially on the number of potential outages.



## **49. Learning to Break Deep Perceptual Hashing: The Use Case NeuralHash**

cs.LG

22 pages, 15 figures, 5 tables

**SubmitDate**: 2021-11-12    [paper-pdf](http://arxiv.org/pdf/2111.06628v1)

**Authors**: Lukas Struppek, Dominik Hintersdorf, Daniel Neider, Kristian Kersting

**Abstracts**: Apple recently revealed its deep perceptual hashing system NeuralHash to detect child sexual abuse material (CSAM) on user devices before files are uploaded to its iCloud service. Public criticism quickly arose regarding the protection of user privacy and the system's reliability. In this paper, we present the first comprehensive empirical analysis of deep perceptual hashing based on NeuralHash. Specifically, we show that current deep perceptual hashing may not be robust. An adversary can manipulate the hash values by applying slight changes in images, either induced by gradient-based approaches or simply by performing standard image transformations, forcing or preventing hash collisions. Such attacks permit malicious actors easily to exploit the detection system: from hiding abusive material to framing innocent users, everything is possible. Moreover, using the hash values, inferences can still be made about the data stored on user devices. In our view, based on our results, deep perceptual hashing in its current form is generally not ready for robust client-side scanning and should not be used from a privacy perspective.



## **50. Characterizing and Improving the Robustness of Self-Supervised Learning through Background Augmentations**

cs.CV

Technical Report; Additional Results

**SubmitDate**: 2021-11-12    [paper-pdf](http://arxiv.org/pdf/2103.12719v2)

**Authors**: Chaitanya K. Ryali, David J. Schwab, Ari S. Morcos

**Abstracts**: Recent progress in self-supervised learning has demonstrated promising results in multiple visual tasks. An important ingredient in high-performing self-supervised methods is the use of data augmentation by training models to place different augmented views of the same image nearby in embedding space. However, commonly used augmentation pipelines treat images holistically, ignoring the semantic relevance of parts of an image-e.g. a subject vs. a background-which can lead to the learning of spurious correlations. Our work addresses this problem by investigating a class of simple, yet highly effective "background augmentations", which encourage models to focus on semantically-relevant content by discouraging them from focusing on image backgrounds. Through a systematic investigation, we show that background augmentations lead to substantial improvements in performance across a spectrum of state-of-the-art self-supervised methods (MoCo-v2, BYOL, SwAV) on a variety of tasks, e.g. $\sim$+1-2% gains on ImageNet, enabling performance on par with the supervised baseline. Further, we find the improvement in limited-labels settings is even larger (up to 4.2%). Background augmentations also improve robustness to a number of distribution shifts, including natural adversarial examples, ImageNet-9, adversarial attacks, ImageNet-Renditions. We also make progress in completely unsupervised saliency detection, in the process of generating saliency masks used for background augmentations.



