# Latest Adversarial Attack Papers
**update at 2022-10-14 17:17:52**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. Pikachu: Securing PoS Blockchains from Long-Range Attacks by Checkpointing into Bitcoin PoW using Taproot**

cs.CR

To appear at ConsensusDay 22 (ACM CCS 2022 Workshop)

**SubmitDate**: 2022-10-13    [abs](http://arxiv.org/abs/2208.05408v2) [paper-pdf](http://arxiv.org/pdf/2208.05408v2)

**Authors**: Sarah Azouvi, Marko Vukolić

**Abstract**: Blockchain systems based on a reusable resource, such as proof-of-stake (PoS), provide weaker security guarantees than those based on proof-of-work. Specifically, they are vulnerable to long-range attacks, where an adversary can corrupt prior participants in order to rewrite the full history of the chain. To prevent this attack on a PoS chain, we propose a protocol that checkpoints the state of the PoS chain to a proof-of-work blockchain such as Bitcoin. Our checkpointing protocol hence does not rely on any central authority. Our work uses Schnorr signatures and leverages Bitcoin recent Taproot upgrade, allowing us to create a checkpointing transaction of constant size. We argue for the security of our protocol and present an open-source implementation that was tested on the Bitcoin testnet.



## **2. AccelAT: A Framework for Accelerating the Adversarial Training of Deep Neural Networks through Accuracy Gradient**

cs.LG

12 pages

**SubmitDate**: 2022-10-13    [abs](http://arxiv.org/abs/2210.06888v1) [paper-pdf](http://arxiv.org/pdf/2210.06888v1)

**Authors**: Farzad Nikfam, Alberto Marchisio, Maurizio Martina, Muhammad Shafique

**Abstract**: Adversarial training is exploited to develop a robust Deep Neural Network (DNN) model against the malicious altered data. These attacks may have catastrophic effects on DNN models but are indistinguishable for a human being. For example, an external attack can modify an image adding noises invisible for a human eye, but a DNN model misclassified the image. A key objective for developing robust DNN models is to use a learning algorithm that is fast but can also give model that is robust against different types of adversarial attacks. Especially for adversarial training, enormously long training times are needed for obtaining high accuracy under many different types of adversarial samples generated using different adversarial attack techniques.   This paper aims at accelerating the adversarial training to enable fast development of robust DNN models against adversarial attacks. The general method for improving the training performance is the hyperparameters fine-tuning, where the learning rate is one of the most crucial hyperparameters. By modifying its shape (the value over time) and value during the training, we can obtain a model robust to adversarial attacks faster than standard training.   First, we conduct experiments on two different datasets (CIFAR10, CIFAR100), exploring various techniques. Then, this analysis is leveraged to develop a novel fast training methodology, AccelAT, which automatically adjusts the learning rate for different epochs based on the accuracy gradient. The experiments show comparable results with the related works, and in several experiments, the adversarial training of DNNs using our AccelAT framework is conducted up to 2 times faster than the existing techniques. Thus, our findings boost the speed of adversarial training in an era in which security and performance are fundamental optimization objectives in DNN-based applications.



## **3. Adv-Attribute: Inconspicuous and Transferable Adversarial Attack on Face Recognition**

cs.CV

Accepted by NeurIPS2022

**SubmitDate**: 2022-10-13    [abs](http://arxiv.org/abs/2210.06871v1) [paper-pdf](http://arxiv.org/pdf/2210.06871v1)

**Authors**: Shuai Jia, Bangjie Yin, Taiping Yao, Shouhong Ding, Chunhua Shen, Xiaokang Yang, Chao Ma

**Abstract**: Deep learning models have shown their vulnerability when dealing with adversarial attacks. Existing attacks almost perform on low-level instances, such as pixels and super-pixels, and rarely exploit semantic clues. For face recognition attacks, existing methods typically generate the l_p-norm perturbations on pixels, however, resulting in low attack transferability and high vulnerability to denoising defense models. In this work, instead of performing perturbations on the low-level pixels, we propose to generate attacks through perturbing on the high-level semantics to improve attack transferability. Specifically, a unified flexible framework, Adversarial Attributes (Adv-Attribute), is designed to generate inconspicuous and transferable attacks on face recognition, which crafts the adversarial noise and adds it into different attributes based on the guidance of the difference in face recognition features from the target. Moreover, the importance-aware attribute selection and the multi-objective optimization strategy are introduced to further ensure the balance of stealthiness and attacking strength. Extensive experiments on the FFHQ and CelebA-HQ datasets show that the proposed Adv-Attribute method achieves the state-of-the-art attacking success rates while maintaining better visual effects against recent attack methods.



## **4. Federated Learning for Tabular Data: Exploring Potential Risk to Privacy**

cs.CR

In the proceedings of The 33rd IEEE International Symposium on  Software Reliability Engineering (ISSRE), November 2022

**SubmitDate**: 2022-10-13    [abs](http://arxiv.org/abs/2210.06856v1) [paper-pdf](http://arxiv.org/pdf/2210.06856v1)

**Authors**: Han Wu, Zilong Zhao, Lydia Y. Chen, Aad van Moorsel

**Abstract**: Federated Learning (FL) has emerged as a potentially powerful privacy-preserving machine learning methodology, since it avoids exchanging data between participants, but instead exchanges model parameters. FL has traditionally been applied to image, voice and similar data, but recently it has started to draw attention from domains including financial services where the data is predominantly tabular. However, the work on tabular data has not yet considered potential attacks, in particular attacks using Generative Adversarial Networks (GANs), which have been successfully applied to FL for non-tabular data. This paper is the first to explore leakage of private data in Federated Learning systems that process tabular data. We design a Generative Adversarial Networks (GANs)-based attack model which can be deployed on a malicious client to reconstruct data and its properties from other participants. As a side-effect of considering tabular data, we are able to statistically assess the efficacy of the attack (without relying on human observation such as done for FL for images). We implement our attack model in a recently developed generic FL software framework for tabular data processing. The experimental results demonstrate the effectiveness of the proposed attack model, thus suggesting that further research is required to counter GAN-based privacy attacks.



## **5. Observed Adversaries in Deep Reinforcement Learning**

cs.LG

**SubmitDate**: 2022-10-13    [abs](http://arxiv.org/abs/2210.06787v1) [paper-pdf](http://arxiv.org/pdf/2210.06787v1)

**Authors**: Eugene Lim, Harold Soh

**Abstract**: In this work, we point out the problem of observed adversaries for deep policies. Specifically, recent work has shown that deep reinforcement learning is susceptible to adversarial attacks where an observed adversary acts under environmental constraints to invoke natural but adversarial observations. This setting is particularly relevant for HRI since HRI-related robots are expected to perform their tasks around and with other agents. In this work, we demonstrate that this effect persists even with low-dimensional observations. We further show that these adversarial attacks transfer across victims, which potentially allows malicious attackers to train an adversary without access to the target victim.



## **6. COLLIDER: A Robust Training Framework for Backdoor Data**

cs.LG

Accepted to the 16th Asian Conference on Computer Vision (ACCV 2022)

**SubmitDate**: 2022-10-13    [abs](http://arxiv.org/abs/2210.06704v1) [paper-pdf](http://arxiv.org/pdf/2210.06704v1)

**Authors**: Hadi M. Dolatabadi, Sarah Erfani, Christopher Leckie

**Abstract**: Deep neural network (DNN) classifiers are vulnerable to backdoor attacks. An adversary poisons some of the training data in such attacks by installing a trigger. The goal is to make the trained DNN output the attacker's desired class whenever the trigger is activated while performing as usual for clean data. Various approaches have recently been proposed to detect malicious backdoored DNNs. However, a robust, end-to-end training approach, like adversarial training, is yet to be discovered for backdoor poisoned data. In this paper, we take the first step toward such methods by developing a robust training framework, COLLIDER, that selects the most prominent samples by exploiting the underlying geometric structures of the data. Specifically, we effectively filter out candidate poisoned data at each training epoch by solving a geometrical coreset selection objective. We first argue how clean data samples exhibit (1) gradients similar to the clean majority of data and (2) low local intrinsic dimensionality (LID). Based on these criteria, we define a novel coreset selection objective to find such samples, which are used for training a DNN. We show the effectiveness of the proposed method for robust training of DNNs on various poisoned datasets, reducing the backdoor success rate significantly.



## **7. A Game Theoretical vulnerability analysis of Adversarial Attack**

cs.GT

Accepted in 17th International Symposium on Visual Computing,2022

**SubmitDate**: 2022-10-13    [abs](http://arxiv.org/abs/2210.06670v1) [paper-pdf](http://arxiv.org/pdf/2210.06670v1)

**Authors**: Khondker Fariha Hossain, Alireza Tavakkoli, Shamik Sengupta

**Abstract**: In recent times deep learning has been widely used for automating various security tasks in Cyber Domains. However, adversaries manipulate data in many situations and diminish the deployed deep learning model's accuracy. One notable example is fooling CAPTCHA data to access the CAPTCHA-based Classifier leading to the critical system being vulnerable to cybersecurity attacks. To alleviate this, we propose a computational framework of game theory to analyze the CAPTCHA-based Classifier's vulnerability, strategy, and outcomes by forming a simultaneous two-player game. We apply the Fast Gradient Symbol Method (FGSM) and One Pixel Attack on CAPTCHA Data to imitate real-life scenarios of possible cyber-attack. Subsequently, to interpret this scenario from a Game theoretical perspective, we represent the interaction in the Stackelberg Game in Kuhn tree to study players' possible behaviors and actions by applying our Classifier's actual predicted values. Thus, we interpret potential attacks in deep learning applications while representing viable defense strategies in the game theory prospect.



## **8. Understanding Impacts of Task Similarity on Backdoor Attack and Detection**

cs.CR

**SubmitDate**: 2022-10-12    [abs](http://arxiv.org/abs/2210.06509v1) [paper-pdf](http://arxiv.org/pdf/2210.06509v1)

**Authors**: Di Tang, Rui Zhu, XiaoFeng Wang, Haixu Tang, Yi Chen

**Abstract**: With extensive studies on backdoor attack and detection, still fundamental questions are left unanswered regarding the limits in the adversary's capability to attack and the defender's capability to detect. We believe that answers to these questions can be found through an in-depth understanding of the relations between the primary task that a benign model is supposed to accomplish and the backdoor task that a backdoored model actually performs. For this purpose, we leverage similarity metrics in multi-task learning to formally define the backdoor distance (similarity) between the primary task and the backdoor task, and analyze existing stealthy backdoor attacks, revealing that most of them fail to effectively reduce the backdoor distance and even for those that do, still much room is left to further improve their stealthiness. So we further design a new method, called TSA attack, to automatically generate a backdoor model under a given distance constraint, and demonstrate that our new attack indeed outperforms existing attacks, making a step closer to understanding the attacker's limits. Most importantly, we provide both theoretic results and experimental evidence on various datasets for the positive correlation between the backdoor distance and backdoor detectability, demonstrating that indeed our task similarity analysis help us better understand backdoor risks and has the potential to identify more effective mitigations.



## **9. On Attacking Out-Domain Uncertainty Estimation in Deep Neural Networks**

cs.LG

**SubmitDate**: 2022-10-12    [abs](http://arxiv.org/abs/2210.02191v2) [paper-pdf](http://arxiv.org/pdf/2210.02191v2)

**Authors**: Huimin Zeng, Zhenrui Yue, Yang Zhang, Ziyi Kou, Lanyu Shang, Dong Wang

**Abstract**: In many applications with real-world consequences, it is crucial to develop reliable uncertainty estimation for the predictions made by the AI decision systems. Targeting at the goal of estimating uncertainty, various deep neural network (DNN) based uncertainty estimation algorithms have been proposed. However, the robustness of the uncertainty returned by these algorithms has not been systematically explored. In this work, to raise the awareness of the research community on robust uncertainty estimation, we show that state-of-the-art uncertainty estimation algorithms could fail catastrophically under our proposed adversarial attack despite their impressive performance on uncertainty estimation. In particular, we aim at attacking the out-domain uncertainty estimation: under our attack, the uncertainty model would be fooled to make high-confident predictions for the out-domain data, which they originally would have rejected. Extensive experimental results on various benchmark image datasets show that the uncertainty estimated by state-of-the-art methods could be easily corrupted by our attack.



## **10. On Optimal Learning Under Targeted Data Poisoning**

cs.LG

**SubmitDate**: 2022-10-12    [abs](http://arxiv.org/abs/2210.02713v2) [paper-pdf](http://arxiv.org/pdf/2210.02713v2)

**Authors**: Steve Hanneke, Amin Karbasi, Mohammad Mahmoody, Idan Mehalel, Shay Moran

**Abstract**: Consider the task of learning a hypothesis class $\mathcal{H}$ in the presence of an adversary that can replace up to an $\eta$ fraction of the examples in the training set with arbitrary adversarial examples. The adversary aims to fail the learner on a particular target test point $x$ which is known to the adversary but not to the learner. In this work we aim to characterize the smallest achievable error $\epsilon=\epsilon(\eta)$ by the learner in the presence of such an adversary in both realizable and agnostic settings. We fully achieve this in the realizable setting, proving that $\epsilon=\Theta(\mathtt{VC}(\mathcal{H})\cdot \eta)$, where $\mathtt{VC}(\mathcal{H})$ is the VC dimension of $\mathcal{H}$. Remarkably, we show that the upper bound can be attained by a deterministic learner. In the agnostic setting we reveal a more elaborate landscape: we devise a deterministic learner with a multiplicative regret guarantee of $\epsilon \leq C\cdot\mathtt{OPT} + O(\mathtt{VC}(\mathcal{H})\cdot \eta)$, where $C > 1$ is a universal numerical constant. We complement this by showing that for any deterministic learner there is an attack which worsens its error to at least $2\cdot \mathtt{OPT}$. This implies that a multiplicative deterioration in the regret is unavoidable in this case. Finally, the algorithms we develop for achieving the optimal rates are inherently improper. Nevertheless, we show that for a variety of natural concept classes, such as linear classifiers, it is possible to retain the dependence $\epsilon=\Theta_{\mathcal{H}}(\eta)$ by a proper algorithm in the realizable setting. Here $\Theta_{\mathcal{H}}$ conceals a polynomial dependence on $\mathtt{VC}(\mathcal{H})$.



## **11. Alleviating Adversarial Attacks on Variational Autoencoders with MCMC**

cs.LG

**SubmitDate**: 2022-10-12    [abs](http://arxiv.org/abs/2203.09940v2) [paper-pdf](http://arxiv.org/pdf/2203.09940v2)

**Authors**: Anna Kuzina, Max Welling, Jakub M. Tomczak

**Abstract**: Variational autoencoders (VAEs) are latent variable models that can generate complex objects and provide meaningful latent representations. Moreover, they could be further used in downstream tasks such as classification. As previous work has shown, one can easily fool VAEs to produce unexpected latent representations and reconstructions for a visually slightly modified input. Here, we examine several objective functions for adversarial attack construction proposed previously and present a solution to alleviate the effect of these attacks. Our method utilizes the Markov Chain Monte Carlo (MCMC) technique in the inference step that we motivate with a theoretical analysis. Thus, we do not incorporate any extra costs during training, and the performance on non-attacked inputs is not decreased. We validate our approach on a variety of datasets (MNIST, Fashion MNIST, Color MNIST, CelebA) and VAE configurations ($\beta$-VAE, NVAE, $\beta$-TCVAE), and show that our approach consistently improves the model robustness to adversarial attacks.



## **12. Visual Prompting for Adversarial Robustness**

cs.CV

6 pages, 4 figures, 3 tables

**SubmitDate**: 2022-10-12    [abs](http://arxiv.org/abs/2210.06284v1) [paper-pdf](http://arxiv.org/pdf/2210.06284v1)

**Authors**: Aochuan Chen, Peter Lorenz, Yuguang Yao, Pin-Yu Chen, Sijia Liu

**Abstract**: In this work, we leverage visual prompting (VP) to improve adversarial robustness of a fixed, pre-trained model at testing time. Compared to conventional adversarial defenses, VP allows us to design universal (i.e., data-agnostic) input prompting templates, which have plug-and-play capabilities at testing time to achieve desired model performance without introducing much computation overhead. Although VP has been successfully applied to improving model generalization, it remains elusive whether and how it can be used to defend against adversarial attacks. We investigate this problem and show that the vanilla VP approach is not effective in adversarial defense since a universal input prompt lacks the capacity for robust learning against sample-specific adversarial perturbations. To circumvent it, we propose a new VP method, termed Class-wise Adversarial Visual Prompting (C-AVP), to generate class-wise visual prompts so as to not only leverage the strengths of ensemble prompts but also optimize their interrelations to improve model robustness. Our experiments show that C-AVP outperforms the conventional VP method, with 2.1X standard accuracy gain and 2X robust accuracy gain. Compared to classical test-time defenses, C-AVP also yields a 42X inference time speedup.



## **13. A Characterization of Semi-Supervised Adversarially-Robust PAC Learnability**

cs.LG

NeurIPS 2022 camera-ready

**SubmitDate**: 2022-10-12    [abs](http://arxiv.org/abs/2202.05420v2) [paper-pdf](http://arxiv.org/pdf/2202.05420v2)

**Authors**: Idan Attias, Steve Hanneke, Yishay Mansour

**Abstract**: We study the problem of learning an adversarially robust predictor to test time attacks in the semi-supervised PAC model. We address the question of how many labeled and unlabeled examples are required to ensure learning. We show that having enough unlabeled data (the size of a labeled sample that a fully-supervised method would require), the labeled sample complexity can be arbitrarily smaller compared to previous works, and is sharply characterized by a different complexity measure. We prove nearly matching upper and lower bounds on this sample complexity. This shows that there is a significant benefit in semi-supervised robust learning even in the worst-case distribution-free model, and establishes a gap between the supervised and semi-supervised label complexities which is known not to hold in standard non-robust PAC learning.



## **14. Double Bubble, Toil and Trouble: Enhancing Certified Robustness through Transitivity**

cs.LG

Accepted for Neurips`22, 19 pages, 14 figures, for associated code  see https://github.com/andrew-cullen/DoubleBubble

**SubmitDate**: 2022-10-12    [abs](http://arxiv.org/abs/2210.06077v1) [paper-pdf](http://arxiv.org/pdf/2210.06077v1)

**Authors**: Andrew C. Cullen, Paul Montague, Shijie Liu, Sarah M. Erfani, Benjamin I. P. Rubinstein

**Abstract**: In response to subtle adversarial examples flipping classifications of neural network models, recent research has promoted certified robustness as a solution. There, invariance of predictions to all norm-bounded attacks is achieved through randomised smoothing of network inputs. Today's state-of-the-art certifications make optimal use of the class output scores at the input instance under test: no better radius of certification (under the $L_2$ norm) is possible given only these score. However, it is an open question as to whether such lower bounds can be improved using local information around the instance under test. In this work, we demonstrate how today's "optimal" certificates can be improved by exploiting both the transitivity of certifications, and the geometry of the input space, giving rise to what we term Geometrically-Informed Certified Robustness. By considering the smallest distance to points on the boundary of a set of certifications this approach improves certifications for more than $80\%$ of Tiny-Imagenet instances, yielding an on average $5 \%$ increase in the associated certification. When incorporating training time processes that enhance the certified radius, our technique shows even more promising results, with a uniform $4$ percentage point increase in the achieved certified radius.



## **15. SA: Sliding attack for synthetic speech detection with resistance to clipping and self-splicing**

cs.SD

Updated description and formula

**SubmitDate**: 2022-10-12    [abs](http://arxiv.org/abs/2208.13066v2) [paper-pdf](http://arxiv.org/pdf/2208.13066v2)

**Authors**: Deng JiaCheng, Dong Li, Yan Diqun, Wang Rangding, Zeng Jiaming

**Abstract**: Deep neural networks are vulnerable to adversarial examples that mislead models with imperceptible perturbations. In audio, although adversarial examples have achieved incredible attack success rates on white-box settings and black-box settings, most existing adversarial attacks are constrained by the input length. A More practical scenario is that the adversarial examples must be clipped or self-spliced and input into the black-box model. Therefore, it is necessary to explore how to improve transferability in different input length settings. In this paper, we take the synthetic speech detection task as an example and consider two representative SOTA models. We observe that the gradients of fragments with the same sample value are similar in different models via analyzing the gradients obtained by feeding samples into the model after cropping or self-splicing. Inspired by the above observation, we propose a new adversarial attack method termed sliding attack. Specifically, we make each sampling point aware of gradients at different locations, which can simulate the situation where adversarial examples are input to black-box models with varying input lengths. Therefore, instead of using the current gradient directly in each iteration of the gradient calculation, we go through the following three steps. First, we extract subsegments of different lengths using sliding windows. We then augment the subsegments with data from the adjacent domains. Finally, we feed the sub-segments into different models to obtain aggregate gradients to update adversarial examples. Empirical results demonstrate that our method could significantly improve the transferability of adversarial examples after clipping or self-splicing. Besides, our method could also enhance the transferability between models based on different features.



## **16. Boosting the Transferability of Adversarial Attacks with Reverse Adversarial Perturbation**

cs.CV

NeurIPS 2022 conference paper

**SubmitDate**: 2022-10-12    [abs](http://arxiv.org/abs/2210.05968v1) [paper-pdf](http://arxiv.org/pdf/2210.05968v1)

**Authors**: Zeyu Qin, Yanbo Fan, Yi Liu, Li Shen, Yong Zhang, Jue Wang, Baoyuan Wu

**Abstract**: Deep neural networks (DNNs) have been shown to be vulnerable to adversarial examples, which can produce erroneous predictions by injecting imperceptible perturbations. In this work, we study the transferability of adversarial examples, which is significant due to its threat to real-world applications where model architecture or parameters are usually unknown. Many existing works reveal that the adversarial examples are likely to overfit the surrogate model that they are generated from, limiting its transfer attack performance against different target models. To mitigate the overfitting of the surrogate model, we propose a novel attack method, dubbed reverse adversarial perturbation (RAP). Specifically, instead of minimizing the loss of a single adversarial point, we advocate seeking adversarial example located at a region with unified low loss value, by injecting the worst-case perturbation (the reverse adversarial perturbation) for each step of the optimization procedure. The adversarial attack with RAP is formulated as a min-max bi-level optimization problem. By integrating RAP into the iterative process for attacks, our method can find more stable adversarial examples which are less sensitive to the changes of decision boundary, mitigating the overfitting of the surrogate model. Comprehensive experimental comparisons demonstrate that RAP can significantly boost adversarial transferability. Furthermore, RAP can be naturally combined with many existing black-box attack techniques, to further boost the transferability. When attacking a real-world image recognition system, Google Cloud Vision API, we obtain 22% performance improvement of targeted attacks over the compared method. Our codes are available at https://github.com/SCLBD/Transfer_attack_RAP.



## **17. Robust Models are less Over-Confident**

cs.CV

accepted at NeuRips 2022

**SubmitDate**: 2022-10-12    [abs](http://arxiv.org/abs/2210.05938v1) [paper-pdf](http://arxiv.org/pdf/2210.05938v1)

**Authors**: Julia Grabinski, Paul Gavrikov, Janis Keuper, Margret Keuper

**Abstract**: Despite the success of convolutional neural networks (CNNs) in many academic benchmarks for computer vision tasks, their application in the real-world is still facing fundamental challenges. One of these open problems is the inherent lack of robustness, unveiled by the striking effectiveness of adversarial attacks. Current attack methods are able to manipulate the network's prediction by adding specific but small amounts of noise to the input. In turn, adversarial training (AT) aims to achieve robustness against such attacks and ideally a better model generalization ability by including adversarial samples in the trainingset. However, an in-depth analysis of the resulting robust models beyond adversarial robustness is still pending. In this paper, we empirically analyze a variety of adversarially trained models that achieve high robust accuracies when facing state-of-the-art attacks and we show that AT has an interesting side-effect: it leads to models that are significantly less overconfident with their decisions, even on clean data than non-robust models. Further, our analysis of robust models shows that not only AT but also the model's building blocks (like activation functions and pooling) have a strong influence on the models' prediction confidences. Data & Project website: https://github.com/GeJulia/robustness_confidences_evaluation



## **18. Efficient Adversarial Training without Attacking: Worst-Case-Aware Robust Reinforcement Learning**

cs.LG

36th Conference on Neural Information Processing Systems (NeurIPS  2022)

**SubmitDate**: 2022-10-12    [abs](http://arxiv.org/abs/2210.05927v1) [paper-pdf](http://arxiv.org/pdf/2210.05927v1)

**Authors**: Yongyuan Liang, Yanchao Sun, Ruijie Zheng, Furong Huang

**Abstract**: Recent studies reveal that a well-trained deep reinforcement learning (RL) policy can be particularly vulnerable to adversarial perturbations on input observations. Therefore, it is crucial to train RL agents that are robust against any attacks with a bounded budget. Existing robust training methods in deep RL either treat correlated steps separately, ignoring the robustness of long-term rewards, or train the agents and RL-based attacker together, doubling the computational burden and sample complexity of the training process. In this work, we propose a strong and efficient robust training framework for RL, named Worst-case-aware Robust RL (WocaR-RL) that directly estimates and optimizes the worst-case reward of a policy under bounded l_p attacks without requiring extra samples for learning an attacker. Experiments on multiple environments show that WocaR-RL achieves state-of-the-art performance under various strong attacks, and obtains significantly higher training efficiency than prior state-of-the-art robust training methods. The code of this work is available at https://github.com/umd-huang-lab/WocaR-RL.



## **19. On the Limitations of Stochastic Pre-processing Defenses**

cs.LG

Accepted by Proceedings of the 36th Conference on Neural Information  Processing Systems

**SubmitDate**: 2022-10-11    [abs](http://arxiv.org/abs/2206.09491v3) [paper-pdf](http://arxiv.org/pdf/2206.09491v3)

**Authors**: Yue Gao, Ilia Shumailov, Kassem Fawaz, Nicolas Papernot

**Abstract**: Defending against adversarial examples remains an open problem. A common belief is that randomness at inference increases the cost of finding adversarial inputs. An example of such a defense is to apply a random transformation to inputs prior to feeding them to the model. In this paper, we empirically and theoretically investigate such stochastic pre-processing defenses and demonstrate that they are flawed. First, we show that most stochastic defenses are weaker than previously thought; they lack sufficient randomness to withstand even standard attacks like projected gradient descent. This casts doubt on a long-held assumption that stochastic defenses invalidate attacks designed to evade deterministic defenses and force attackers to integrate the Expectation over Transformation (EOT) concept. Second, we show that stochastic defenses confront a trade-off between adversarial robustness and model invariance; they become less effective as the defended model acquires more invariance to their randomization. Future work will need to decouple these two effects. We also discuss implications and guidance for future research.



## **20. Adversarial Attack Against Image-Based Localization Neural Networks**

cs.CV

13 pages, 10 figures

**SubmitDate**: 2022-10-11    [abs](http://arxiv.org/abs/2210.06589v1) [paper-pdf](http://arxiv.org/pdf/2210.06589v1)

**Authors**: Meir Brand, Itay Naeh, Daniel Teitelman

**Abstract**: In this paper, we present a proof of concept for adversarially attacking the image-based localization module of an autonomous vehicle. This attack aims to cause the vehicle to perform a wrong navigational decisions and prevent it from reaching a desired predefined destination in a simulated urban environment. A database of rendered images allowed us to train a deep neural network that performs a localization task and implement, develop and assess the adversarial pattern. Our tests show that using this adversarial attack we can prevent the vehicle from turning at a given intersection. This is done by manipulating the vehicle's navigational module to falsely estimate its current position and thus fail to initialize the turning procedure until the vehicle misses the last opportunity to perform a safe turn in a given intersection.



## **21. Indicators of Attack Failure: Debugging and Improving Optimization of Adversarial Examples**

cs.LG

Accepted at NeurIPS 2022

**SubmitDate**: 2022-10-11    [abs](http://arxiv.org/abs/2106.09947v3) [paper-pdf](http://arxiv.org/pdf/2106.09947v3)

**Authors**: Maura Pintor, Luca Demetrio, Angelo Sotgiu, Ambra Demontis, Nicholas Carlini, Battista Biggio, Fabio Roli

**Abstract**: Evaluating robustness of machine-learning models to adversarial examples is a challenging problem. Many defenses have been shown to provide a false sense of robustness by causing gradient-based attacks to fail, and they have been broken under more rigorous evaluations. Although guidelines and best practices have been suggested to improve current adversarial robustness evaluations, the lack of automatic testing and debugging tools makes it difficult to apply these recommendations in a systematic manner. In this work, we overcome these limitations by: (i) categorizing attack failures based on how they affect the optimization of gradient-based attacks, while also unveiling two novel failures affecting many popular attack implementations and past evaluations; (ii) proposing six novel indicators of failure, to automatically detect the presence of such failures in the attack optimization process; and (iii) suggesting a systematic protocol to apply the corresponding fixes. Our extensive experimental analysis, involving more than 15 models in 3 distinct application domains, shows that our indicators of failure can be used to debug and improve current adversarial robustness evaluations, thereby providing a first concrete step towards automatizing and systematizing them. Our open-source code is available at: https://github.com/pralab/IndicatorsOfAttackFailure.



## **22. Stable and Efficient Adversarial Training through Local Linearization**

cs.LG

**SubmitDate**: 2022-10-11    [abs](http://arxiv.org/abs/2210.05373v1) [paper-pdf](http://arxiv.org/pdf/2210.05373v1)

**Authors**: Zhuorong Li, Daiwei Yu

**Abstract**: There has been a recent surge in single-step adversarial training as it shows robustness and efficiency. However, a phenomenon referred to as ``catastrophic overfitting" has been observed, which is prevalent in single-step defenses and may frustrate attempts to use FGSM adversarial training. To address this issue, we propose a novel method, Stable and Efficient Adversarial Training (SEAT), which mitigates catastrophic overfitting by harnessing on local properties that distinguish a robust model from that of a catastrophic overfitted model. The proposed SEAT has strong theoretical justifications, in that minimizing the SEAT loss can be shown to favour smooth empirical risk, thereby leading to robustness. Experimental results demonstrate that the proposed method successfully mitigates catastrophic overfitting, yielding superior performance amongst efficient defenses. Our single-step method can reach 51% robust accuracy for CIFAR-10 with $l_\infty$ perturbations of radius $8/255$ under a strong PGD-50 attack, matching the performance of a 10-step iterative adversarial training at merely 3% computational cost.



## **23. Adversarial Robustness of Deep Neural Networks: A Survey from a Formal Verification Perspective**

cs.CR

**SubmitDate**: 2022-10-11    [abs](http://arxiv.org/abs/2206.12227v2) [paper-pdf](http://arxiv.org/pdf/2206.12227v2)

**Authors**: Mark Huasong Meng, Guangdong Bai, Sin Gee Teo, Zhe Hou, Yan Xiao, Yun Lin, Jin Song Dong

**Abstract**: Neural networks have been widely applied in security applications such as spam and phishing detection, intrusion prevention, and malware detection. This black-box method, however, often has uncertainty and poor explainability in applications. Furthermore, neural networks themselves are often vulnerable to adversarial attacks. For those reasons, there is a high demand for trustworthy and rigorous methods to verify the robustness of neural network models. Adversarial robustness, which concerns the reliability of a neural network when dealing with maliciously manipulated inputs, is one of the hottest topics in security and machine learning. In this work, we survey existing literature in adversarial robustness verification for neural networks and collect 39 diversified research works across machine learning, security, and software engineering domains. We systematically analyze their approaches, including how robustness is formulated, what verification techniques are used, and the strengths and limitations of each technique. We provide a taxonomy from a formal verification perspective for a comprehensive understanding of this topic. We classify the existing techniques based on property specification, problem reduction, and reasoning strategies. We also demonstrate representative techniques that have been applied in existing studies with a sample model. Finally, we discuss open questions for future research.



## **24. Zeroth-Order Hard-Thresholding: Gradient Error vs. Expansivity**

cs.LG

Accepted for publication at NeurIPS 2022

**SubmitDate**: 2022-10-11    [abs](http://arxiv.org/abs/2210.05279v1) [paper-pdf](http://arxiv.org/pdf/2210.05279v1)

**Authors**: William de Vazelhes, Hualin Zhang, Huimin Wu, Xiao-Tong Yuan, Bin Gu

**Abstract**: $\ell_0$ constrained optimization is prevalent in machine learning, particularly for high-dimensional problems, because it is a fundamental approach to achieve sparse learning. Hard-thresholding gradient descent is a dominant technique to solve this problem. However, first-order gradients of the objective function may be either unavailable or expensive to calculate in a lot of real-world problems, where zeroth-order (ZO) gradients could be a good surrogate. Unfortunately, whether ZO gradients can work with the hard-thresholding operator is still an unsolved problem. To solve this puzzle, in this paper, we focus on the $\ell_0$ constrained black-box stochastic optimization problems, and propose a new stochastic zeroth-order gradient hard-thresholding (SZOHT) algorithm with a general ZO gradient estimator powered by a novel random support sampling. We provide the convergence analysis of SZOHT under standard assumptions. Importantly, we reveal a conflict between the deviation of ZO estimators and the expansivity of the hard-thresholding operator, and provide a theoretical minimal value of the number of random directions in ZO gradients. In addition, we find that the query complexity of SZOHT is independent or weakly dependent on the dimensionality under different settings. Finally, we illustrate the utility of our method on a portfolio optimization problem as well as black-box adversarial attacks.



## **25. RoHNAS: A Neural Architecture Search Framework with Conjoint Optimization for Adversarial Robustness and Hardware Efficiency of Convolutional and Capsule Networks**

cs.LG

Accepted for publication at IEEE Access

**SubmitDate**: 2022-10-11    [abs](http://arxiv.org/abs/2210.05276v1) [paper-pdf](http://arxiv.org/pdf/2210.05276v1)

**Authors**: Alberto Marchisio, Vojtech Mrazek, Andrea Massa, Beatrice Bussolino, Maurizio Martina, Muhammad Shafique

**Abstract**: Neural Architecture Search (NAS) algorithms aim at finding efficient Deep Neural Network (DNN) architectures for a given application under given system constraints. DNNs are computationally-complex as well as vulnerable to adversarial attacks. In order to address multiple design objectives, we propose RoHNAS, a novel NAS framework that jointly optimizes for adversarial-robustness and hardware-efficiency of DNNs executed on specialized hardware accelerators. Besides the traditional convolutional DNNs, RoHNAS additionally accounts for complex types of DNNs such as Capsule Networks. For reducing the exploration time, RoHNAS analyzes and selects appropriate values of adversarial perturbation for each dataset to employ in the NAS flow. Extensive evaluations on multi - Graphics Processing Unit (GPU) - High Performance Computing (HPC) nodes provide a set of Pareto-optimal solutions, leveraging the tradeoff between the above-discussed design objectives. For example, a Pareto-optimal DNN for the CIFAR-10 dataset exhibits 86.07% accuracy, while having an energy of 38.63 mJ, a memory footprint of 11.85 MiB, and a latency of 4.47 ms.



## **26. Towards Lightweight Black-Box Attacks against Deep Neural Networks**

cs.LG

**SubmitDate**: 2022-10-11    [abs](http://arxiv.org/abs/2209.14826v3) [paper-pdf](http://arxiv.org/pdf/2209.14826v3)

**Authors**: Chenghao Sun, Yonggang Zhang, Wan Chaoqun, Qizhou Wang, Ya Li, Tongliang Liu, Bo Han, Xinmei Tian

**Abstract**: Black-box attacks can generate adversarial examples without accessing the parameters of target model, largely exacerbating the threats of deployed deep neural networks (DNNs). However, previous works state that black-box attacks fail to mislead target models when their training data and outputs are inaccessible. In this work, we argue that black-box attacks can pose practical attacks in this extremely restrictive scenario where only several test samples are available. Specifically, we find that attacking the shallow layers of DNNs trained on a few test samples can generate powerful adversarial examples. As only a few samples are required, we refer to these attacks as lightweight black-box attacks. The main challenge to promoting lightweight attacks is to mitigate the adverse impact caused by the approximation error of shallow layers. As it is hard to mitigate the approximation error with few available samples, we propose Error TransFormer (ETF) for lightweight attacks. Namely, ETF transforms the approximation error in the parameter space into a perturbation in the feature space and alleviates the error by disturbing features. In experiments, lightweight black-box attacks with the proposed ETF achieve surprising results. For example, even if only 1 sample per category available, the attack success rate in lightweight black-box attacks is only about 3% lower than that of the black-box attacks with complete training data.



## **27. Content-Adaptive Pixel Discretization to Improve Model Robustness**

cs.CV

**SubmitDate**: 2022-10-11    [abs](http://arxiv.org/abs/2012.01699v4) [paper-pdf](http://arxiv.org/pdf/2012.01699v4)

**Authors**: Ryan Feng, Wu-chi Feng, Atul Prakash

**Abstract**: Preprocessing defenses such as pixel discretization are appealing to remove adversarial attacks due to their simplicity. However, they have been shown to be ineffective except on simple datasets like MNIST. We hypothesize that existing discretization approaches failed because using a fixed codebook for the entire dataset limits their ability to balance image representation and codeword separability. We first formally prove that adaptive codebooks can provide stronger robustness guarantees than fixed codebooks as a preprocessing defense on some datasets. Based on that insight, we propose a content-adaptive pixel discretization defense called Essential Features, which discretizes the image to a per-image adaptive codebook to reduce the color space. We then find that Essential Features can be further optimized by applying adaptive blurring before the discretization to push perturbed pixel values back to their original value before determining the codebook. Against adaptive attacks, we show that content-adaptive pixel discretization extends the range of datasets that benefit in terms of both L_2 and L_infinity robustness where previously fixed codebooks were found to have failed. Our findings suggest that content-adaptive pixel discretization should be part of the repertoire for making models robust.



## **28. Spinning Sequence-to-Sequence Models with Meta-Backdoors**

cs.CR

Outdated. Superseded by arXiv:2112.05224 and published at IEEE S&P'22  with title: "Spinning Language Models: Risks of Propaganda-As-A-Service and  Countermeasures"

**SubmitDate**: 2022-10-10    [abs](http://arxiv.org/abs/2107.10443v2) [paper-pdf](http://arxiv.org/pdf/2107.10443v2)

**Authors**: Eugene Bagdasaryan, Vitaly Shmatikov

**Abstract**: We investigate a new threat to neural sequence-to-sequence (seq2seq) models: training-time attacks that cause models to "spin" their output and support a certain sentiment when the input contains adversary-chosen trigger words. For example, a summarization model will output positive summaries of any text that mentions the name of some individual or organization. We introduce the concept of a "meta-backdoor" to explain model-spinning attacks. These attacks produce models whose output is valid and preserves context, yet also satisfies a meta-task chosen by the adversary (e.g., positive sentiment). Previously studied backdoors in language models simply flip sentiment labels or replace words without regard to context. Their outputs are incorrect on inputs with the trigger. Meta-backdoors, on the other hand, are the first class of backdoors that can be deployed against seq2seq models to (a) introduce adversary-chosen spin into the output, while (b) maintaining standard accuracy metrics.   To demonstrate feasibility of model spinning, we develop a new backdooring technique. It stacks the adversarial meta-task (e.g., sentiment analysis) onto a seq2seq model, backpropagates the desired meta-task output (e.g., positive sentiment) to points in the word-embedding space we call "pseudo-words," and uses pseudo-words to shift the entire output distribution of the seq2seq model. Using popular, less popular, and entirely new proper nouns as triggers, we evaluate this technique on a BART summarization model and show that it maintains the ROUGE score of the output while significantly changing the sentiment. We explain why model spinning can be a dangerous technique in AI-powered disinformation and discuss how to mitigate these attacks.



## **29. Towards Out-of-Distribution Adversarial Robustness**

cs.LG

Under review ICLR 2023

**SubmitDate**: 2022-10-10    [abs](http://arxiv.org/abs/2210.03150v2) [paper-pdf](http://arxiv.org/pdf/2210.03150v2)

**Authors**: Adam Ibrahim, Charles Guille-Escuret, Ioannis Mitliagkas, Irina Rish, David Krueger, Pouya Bashivan

**Abstract**: Adversarial robustness continues to be a major challenge for deep learning. A core issue is that robustness to one type of attack often fails to transfer to other attacks. While prior work establishes a theoretical trade-off in robustness against different $L_p$ norms, we show that there is potential for improvement against many commonly used attacks by adopting a domain generalisation approach. Concretely, we treat each type of attack as a domain, and apply the Risk Extrapolation method (REx), which promotes similar levels of robustness against all training attacks. Compared to existing methods, we obtain similar or superior worst-case adversarial robustness on attacks seen during training. Moreover, we achieve superior performance on families or tunings of attacks only encountered at test time. On ensembles of attacks, our approach improves the accuracy from 3.4% the best existing baseline to 25.9% on MNIST, and from 16.9% to 23.5% on CIFAR10.



## **30. Sampling without Replacement Leads to Faster Rates in Finite-Sum Minimax Optimization**

math.OC

36th Conference on Neural Information Processing Systems (NeurIPS  2022)

**SubmitDate**: 2022-10-10    [abs](http://arxiv.org/abs/2206.02953v2) [paper-pdf](http://arxiv.org/pdf/2206.02953v2)

**Authors**: Aniket Das, Bernhard Schölkopf, Michael Muehlebach

**Abstract**: We analyze the convergence rates of stochastic gradient algorithms for smooth finite-sum minimax optimization and show that, for many such algorithms, sampling the data points without replacement leads to faster convergence compared to sampling with replacement. For the smooth and strongly convex-strongly concave setting, we consider gradient descent ascent and the proximal point method, and present a unified analysis of two popular without-replacement sampling strategies, namely Random Reshuffling (RR), which shuffles the data every epoch, and Single Shuffling or Shuffle Once (SO), which shuffles only at the beginning. We obtain tight convergence rates for RR and SO and demonstrate that these strategies lead to faster convergence than uniform sampling. Moving beyond convexity, we obtain similar results for smooth nonconvex-nonconcave objectives satisfying a two-sided Polyak-{\L}ojasiewicz inequality. Finally, we demonstrate that our techniques are general enough to analyze the effect of data-ordering attacks, where an adversary manipulates the order in which data points are supplied to the optimizer. Our analysis also recovers tight rates for the incremental gradient method, where the data points are not shuffled at all.



## **31. On The Robustness of Channel Allocation in Joint Radar And Communication Systems: An Auction Approach**

cs.GT

**SubmitDate**: 2022-10-10    [abs](http://arxiv.org/abs/2208.09821v2) [paper-pdf](http://arxiv.org/pdf/2208.09821v2)

**Authors**: Ismail Lotfi, Hongyang Du, Dusit Niyato, Sumei Sun, Dong In Kim

**Abstract**: Joint radar and communication (JRC) is a promising technique for spectrum re-utilization, which enables radar sensing and data transmission to operate on the same frequencies and the same devices. However, due to the multi-objective property of JRC systems, channel allocation to JRC nodes should be carefully designed to maximize system performance. Additionally, because of the broadcast nature of wireless signals, a watchful adversary, i.e., a warden, can detect ongoing transmissions and attack the system. Thus, we develop a covert JRC system that minimizes the detection probability by wardens, in which friendly jammers are deployed to improve the covertness of the JRC nodes during radar sensing and data transmission operations. Furthermore, we propose a robust multi-item auction design for channel allocation for such a JRC system that considers the uncertainty in bids. The proposed auction mechanism achieves the properties of truthfulness, individual rationality, budget feasibility, and computational efficiency. The simulations clearly show the benefits of our design to support covert JRC systems and to provide incentive to the JRC nodes in obtaining spectrum, in which the auction-based channel allocation mechanism is robust against perturbations in the bids, which is highly effective for JRC nodes working in uncertain environments.



## **32. Pruning Adversarially Robust Neural Networks without Adversarial Examples**

cs.LG

Published at ICDM 2022 as a conference paper

**SubmitDate**: 2022-10-09    [abs](http://arxiv.org/abs/2210.04311v1) [paper-pdf](http://arxiv.org/pdf/2210.04311v1)

**Authors**: Tong Jian, Zifeng Wang, Yanzhi Wang, Jennifer Dy, Stratis Ioannidis

**Abstract**: Adversarial pruning compresses models while preserving robustness. Current methods require access to adversarial examples during pruning. This significantly hampers training efficiency. Moreover, as new adversarial attacks and training methods develop at a rapid rate, adversarial pruning methods need to be modified accordingly to keep up. In this work, we propose a novel framework to prune a previously trained robust neural network while maintaining adversarial robustness, without further generating adversarial examples. We leverage concurrent self-distillation and pruning to preserve knowledge in the original model as well as regularizing the pruned model via the Hilbert-Schmidt Information Bottleneck. We comprehensively evaluate our proposed framework and show its superior performance in terms of both adversarial robustness and efficiency when pruning architectures trained on the MNIST, CIFAR-10, and CIFAR-100 datasets against five state-of-the-art attacks. Code is available at https://github.com/neu-spiral/PwoA/.



## **33. Towards Understanding and Boosting Adversarial Transferability from a Distribution Perspective**

cs.CV

\copyright 20XX IEEE. Personal use of this material is permitted.  Permission from IEEE must be obtained for all other uses, in any current or  future media, including reprinting/republishing this material for advertising  or promotional purposes, creating new collective works, for resale or  redistribution to servers or lists, or reuse of any copyrighted component of  this work in other works

**SubmitDate**: 2022-10-09    [abs](http://arxiv.org/abs/2210.04213v1) [paper-pdf](http://arxiv.org/pdf/2210.04213v1)

**Authors**: Yao Zhu, Yuefeng Chen, Xiaodan Li, Kejiang Chen, Yuan He, Xiang Tian, Bolun Zheng, Yaowu Chen, Qingming Huang

**Abstract**: Transferable adversarial attacks against Deep neural networks (DNNs) have received broad attention in recent years. An adversarial example can be crafted by a surrogate model and then attack the unknown target model successfully, which brings a severe threat to DNNs. The exact underlying reasons for the transferability are still not completely understood. Previous work mostly explores the causes from the model perspective, e.g., decision boundary, model architecture, and model capacity. adversarial attacks against Deep neural networks (DNNs) have received broad attention in recent years. An adversarial example can be crafted by a surrogate model and then attack the unknown target model successfully, which brings a severe threat to DNNs. The exact underlying reasons for the transferability are still not completely understood. Previous work mostly explores the causes from the model perspective. Here, we investigate the transferability from the data distribution perspective and hypothesize that pushing the image away from its original distribution can enhance the adversarial transferability. To be specific, moving the image out of its original distribution makes different models hardly classify the image correctly, which benefits the untargeted attack, and dragging the image into the target distribution misleads the models to classify the image as the target class, which benefits the targeted attack. Towards this end, we propose a novel method that crafts adversarial examples by manipulating the distribution of the image. We conduct comprehensive transferable attacks against multiple DNNs to demonstrate the effectiveness of the proposed method. Our method can significantly improve the transferability of the crafted attacks and achieves state-of-the-art performance in both untargeted and targeted scenarios, surpassing the previous best method by up to 40$\%$ in some cases.



## **34. Adversarial Attacks against Windows PE Malware Detection: A Survey of the State-of-the-Art**

cs.CR

**SubmitDate**: 2022-10-09    [abs](http://arxiv.org/abs/2112.12310v2) [paper-pdf](http://arxiv.org/pdf/2112.12310v2)

**Authors**: Xiang Ling, Lingfei Wu, Jiangyu Zhang, Zhenqing Qu, Wei Deng, Xiang Chen, Yaguan Qian, Chunming Wu, Shouling Ji, Tianyue Luo, Jingzheng Wu, Yanjun Wu

**Abstract**: Malware has been one of the most damaging threats to computers that span across multiple operating systems and various file formats. To defend against ever-increasing and ever-evolving malware, tremendous efforts have been made to propose a variety of malware detection that attempt to effectively and efficiently detect malware so as to mitigate possible damages as early as possible. Recent studies have shown that, on the one hand, existing ML and DL techniques enable superior solutions in detecting newly emerging and previously unseen malware. However, on the other hand, ML and DL models are inherently vulnerable to adversarial attacks in the form of adversarial examples. In this paper, we focus on malware with the file format of portable executable (PE) in the family of Windows operating systems, namely Windows PE malware, as a representative case to study the adversarial attack methods in such adversarial settings. To be specific, we start by first outlining the general learning framework of Windows PE malware detection based on ML/DL and subsequently highlighting three unique challenges of performing adversarial attacks in the context of Windows PE malware. Then, we conduct a comprehensive and systematic review to categorize the state-of-the-art adversarial attacks against PE malware detection, as well as corresponding defenses to increase the robustness of Windows PE malware detection. Finally, we conclude the paper by first presenting other related attacks against Windows PE malware detection beyond the adversarial attacks and then shedding light on future research directions and opportunities. In addition, a curated resource list of adversarial attacks and defenses for Windows PE malware detection is also available at https://github.com/ryderling/ adversarial-attacks-and-defenses-for-windows-pe-malware-detection.



## **35. A Zero-Sum Game Framework for Optimal Sensor Placement in Uncertain Networked Control Systems under Cyber-Attacks**

eess.SY

8 pages, 3 figues, Accepted to the 61st Conference on Decision and  Control, Cancun, December 2022

**SubmitDate**: 2022-10-08    [abs](http://arxiv.org/abs/2210.04091v1) [paper-pdf](http://arxiv.org/pdf/2210.04091v1)

**Authors**: Anh Tung Nguyen, Sribalaji C. Anand, André M. H. Teixeira

**Abstract**: This paper proposes a game-theoretic approach to address the problem of optimal sensor placement against an adversary in uncertain networked control systems. The problem is formulated as a zero-sum game with two players, namely a malicious adversary and a detector. Given a protected performance vertex, we consider a detector, with uncertain system knowledge, that selects another vertex on which to place a sensor and monitors its output with the aim of detecting the presence of the adversary. On the other hand, the adversary, also with uncertain system knowledge, chooses a single vertex and conducts a cyber-attack on its input. The purpose of the adversary is to drive the attack vertex as to maximally disrupt the protected performance vertex while remaining undetected by the detector. As our first contribution, the game payoff of the above-defined zero-sum game is formulated in terms of the Value-at-Risk of the adversary's impact. However, this game payoff corresponds to an intractable optimization problem. To tackle the problem, we adopt the scenario approach to approximately compute the game payoff. Then, the optimal monitor selection is determined by analyzing the equilibrium of the zero-sum game. The proposed approach is illustrated via a numerical example of a 10-vertex networked control system.



## **36. Symmetry Subgroup Defense Against Adversarial Attacks**

cs.LG

14 pages

**SubmitDate**: 2022-10-08    [abs](http://arxiv.org/abs/2210.04087v1) [paper-pdf](http://arxiv.org/pdf/2210.04087v1)

**Authors**: Blerta Lindqvist

**Abstract**: Adversarial attacks and defenses disregard the lack of invariance of convolutional neural networks (CNNs), that is, the inability of CNNs to classify samples and their symmetric transformations the same. The lack of invariance of CNNs with respect to symmetry transformations is detrimental when classifying transformed original samples but not necessarily detrimental when classifying transformed adversarial samples. For original images, the lack of invariance means that symmetrically transformed original samples are classified differently from their correct labels. However, for adversarial images, the lack of invariance means that symmetrically transformed adversarial images are classified differently from their incorrect adversarial labels. Might the CNN lack of invariance revert symmetrically transformed adversarial samples to the correct classification? This paper answers this question affirmatively for a threat model that ranges from zero-knowledge adversaries to perfect-knowledge adversaries. We base our defense against perfect-knowledge adversaries on devising a Klein four symmetry subgroup that incorporates an additional artificial symmetry of pixel intensity inversion. The closure property of the subgroup not only provides a framework for the accuracy evaluation but also confines the transformations that an adaptive, perfect-knowledge adversary can apply. We find that by using only symmetry defense, no adversarial samples, and by changing nothing in the model architecture and parameters, we can defend against white-box PGD adversarial attacks, surpassing the PGD adversarial training defense by up to ~50% even against a perfect-knowledge adversary for ImageNet. The proposed defense also maintains and surpasses the classification accuracy for non-adversarial samples.



## **37. Robustness of Unsupervised Representation Learning without Labels**

cs.LG

**SubmitDate**: 2022-10-08    [abs](http://arxiv.org/abs/2210.04076v1) [paper-pdf](http://arxiv.org/pdf/2210.04076v1)

**Authors**: Aleksandar Petrov, Marta Kwiatkowska

**Abstract**: Unsupervised representation learning leverages large unlabeled datasets and is competitive with supervised learning. But non-robust encoders may affect downstream task robustness. Recently, robust representation encoders have become of interest. Still, all prior work evaluates robustness using a downstream classification task. Instead, we propose a family of unsupervised robustness measures, which are model- and task-agnostic and label-free. We benchmark state-of-the-art representation encoders and show that none dominates the rest. We offer unsupervised extensions to the FGSM and PGD attacks. When used in adversarial training, they improve most unsupervised robustness measures, including certified robustness. We validate our results against a linear probe and show that, for MOCOv2, adversarial training results in 3 times higher certified accuracy, a 2-fold decrease in impersonation attack success rate and considerable improvements in certified robustness.



## **38. FedDef: Robust Federated Learning-based Network Intrusion Detection Systems Against Gradient Leakage**

cs.CR

14 pages, 9 figures

**SubmitDate**: 2022-10-08    [abs](http://arxiv.org/abs/2210.04052v1) [paper-pdf](http://arxiv.org/pdf/2210.04052v1)

**Authors**: Jiahui Chen, Yi Zhao, Qi Li, Ke Xu

**Abstract**: Deep learning methods have been widely applied to anomaly-based network intrusion detection systems (NIDS) to detect malicious traffic. To expand the usage scenarios of DL-based methods, the federated learning (FL) framework allows intelligent techniques to jointly train a model by multiple individuals on the basis of respecting individual data privacy. However, it has not yet been systematically evaluated how robust FL-based NIDSs are against existing privacy attacks under existing defenses. To address this issue, in this paper we propose two privacy evaluation metrics designed for FL-based NIDSs, including leveraging two reconstruction attacks to recover the training data to obtain the privacy score for traffic features, followed by Generative Adversarial Network (GAN) based attack that generates adversarial examples with the reconstructed benign traffic to evaluate evasion rate against other NIDSs. We conduct experiments to show that existing defenses provide little protection that the corresponding adversarial traffic can even evade the SOTA NIDS Kitsune. To build a more robust FL-based NIDS, we further propose a novel optimization-based input perturbation defense strategy with theoretical guarantee that achieves both high utility by minimizing the gradient distance and strong privacy protection by maximizing the input distance. We experimentally evaluate four existing defenses on four datasets and show that our defense outperforms all the baselines with strong privacy guarantee while maintaining model accuracy loss within 3% under optimal parameter combination.



## **39. Can Adversarial Training Be Manipulated By Non-Robust Features?**

cs.LG

NeurIPS 2022

**SubmitDate**: 2022-10-08    [abs](http://arxiv.org/abs/2201.13329v4) [paper-pdf](http://arxiv.org/pdf/2201.13329v4)

**Authors**: Lue Tao, Lei Feng, Hongxin Wei, Jinfeng Yi, Sheng-Jun Huang, Songcan Chen

**Abstract**: Adversarial training, originally designed to resist test-time adversarial examples, has shown to be promising in mitigating training-time availability attacks. This defense ability, however, is challenged in this paper. We identify a novel threat model named stability attack, which aims to hinder robust availability by slightly manipulating the training data. Under this threat, we show that adversarial training using a conventional defense budget $\epsilon$ provably fails to provide test robustness in a simple statistical setting, where the non-robust features of the training data can be reinforced by $\epsilon$-bounded perturbation. Further, we analyze the necessity of enlarging the defense budget to counter stability attacks. Finally, comprehensive experiments demonstrate that stability attacks are harmful on benchmark datasets, and thus the adaptive defense is necessary to maintain robustness. Our code is available at https://github.com/TLMichael/Hypocritical-Perturbation.



## **40. BayesImposter: Bayesian Estimation Based .bss Imposter Attack on Industrial Control Systems**

cs.CR

**SubmitDate**: 2022-10-07    [abs](http://arxiv.org/abs/2210.03719v1) [paper-pdf](http://arxiv.org/pdf/2210.03719v1)

**Authors**: Anomadarshi Barua, Lelin Pan, Mohammad Abdullah Al Faruque

**Abstract**: Over the last six years, several papers used memory deduplication to trigger various security issues, such as leaking heap-address and causing bit-flip in the physical memory. The most essential requirement for successful memory deduplication is to provide identical copies of a physical page. Recent works use a brute-force approach to create identical copies of a physical page that is an inaccurate and time-consuming primitive from the attacker's perspective.   Our work begins to fill this gap by providing a domain-specific structured way to duplicate a physical page in cloud settings in the context of industrial control systems (ICSs). Here, we show a new attack primitive - \textit{BayesImposter}, which points out that the attacker can duplicate the .bss section of the target control DLL file of cloud protocols using the \textit{Bayesian estimation} technique. Our approach results in less memory (i.e., 4 KB compared to GB) and time (i.e., 13 minutes compared to hours) compared to the brute-force approach used in recent works. We point out that ICSs can be expressed as state-space models; hence, the \textit{Bayesian estimation} is an ideal choice to be combined with memory deduplication for a successful attack in cloud settings. To demonstrate the strength of \textit{BayesImposter}, we create a real-world automation platform using a scaled-down automated high-bay warehouse and industrial-grade SIMATIC S7-1500 PLC from Siemens as a target ICS. We demonstrate that \textit{BayesImposter} can predictively inject false commands into the PLC that can cause possible equipment damage with machine failure in the target ICS. Moreover, we show that \textit{BayesImposter} is capable of adversarial control over the target ICS resulting in severe consequences, such as killing a person but making it looks like an accident. Therefore, we also provide countermeasures to prevent the attack.



## **41. A Wolf in Sheep's Clothing: Spreading Deadly Pathogens Under the Disguise of Popular Music**

cs.CR

**SubmitDate**: 2022-10-07    [abs](http://arxiv.org/abs/2210.03688v1) [paper-pdf](http://arxiv.org/pdf/2210.03688v1)

**Authors**: Anomadarshi Barua, Yonatan Gizachew Achamyeleh, Mohammad Abdullah Al Faruque

**Abstract**: A Negative Pressure Room (NPR) is an essential requirement by the Bio-Safety Levels (BSLs) in biolabs or infectious-control hospitals to prevent deadly pathogens from being leaked from the facility. An NPR maintains a negative pressure inside with respect to the outside reference space so that microbes are contained inside of an NPR. Nowadays, differential pressure sensors (DPSs) are utilized by the Building Management Systems (BMSs) to control and monitor the negative pressure in an NPR. This paper demonstrates a non-invasive and stealthy attack on NPRs by spoofing a DPS at its resonant frequency. Our contributions are: (1) We show that DPSs used in NPRs typically have resonant frequencies in the audible range. (2) We use this finding to design malicious music to create resonance in DPSs, resulting in an overshooting in the DPS's normal pressure readings. (3) We show how the resonance in DPSs can fool the BMSs so that the NPR turns its negative pressure to a positive one, causing a potential \textit{leak} of deadly microbes from NPRs. We do experiments on 8 DPSs from 5 different manufacturers to evaluate their resonant frequencies considering the sampling tube length and find resonance in 6 DPSs. We can achieve a 2.5 Pa change in negative pressure from a $\sim$7 cm distance when a sampling tube is not present and from a $\sim$2.5 cm distance for a 1 m sampling tube length. We also introduce an interval-time variation approach for an adversarial control over the negative pressure and show that the \textit{forged} pressure can be varied within 12 - 33 Pa. Our attack is also capable of attacking multiple NPRs simultaneously. Moreover, we demonstrate our attack at a real-world NPR located in an anonymous bioresearch facility, which is FDA approved and follows CDC guidelines. We also provide countermeasures to prevent the attack.



## **42. Empowering Graph Representation Learning with Test-Time Graph Transformation**

cs.LG

**SubmitDate**: 2022-10-07    [abs](http://arxiv.org/abs/2210.03561v1) [paper-pdf](http://arxiv.org/pdf/2210.03561v1)

**Authors**: Wei Jin, Tong Zhao, Jiayuan Ding, Yozen Liu, Jiliang Tang, Neil Shah

**Abstract**: As powerful tools for representation learning on graphs, graph neural networks (GNNs) have facilitated various applications from drug discovery to recommender systems. Nevertheless, the effectiveness of GNNs is immensely challenged by issues related to data quality, such as distribution shift, abnormal features and adversarial attacks. Recent efforts have been made on tackling these issues from a modeling perspective which requires additional cost of changing model architectures or re-training model parameters. In this work, we provide a data-centric view to tackle these issues and propose a graph transformation framework named GTrans which adapts and refines graph data at test time to achieve better performance. We provide theoretical analysis on the design of the framework and discuss why adapting graph data works better than adapting the model. Extensive experiments have demonstrated the effectiveness of GTrans on three distinct scenarios for eight benchmark datasets where suboptimal data is presented. Remarkably, GTrans performs the best in most cases with improvements up to 2.8%, 8.2% and 3.8% over the best baselines on three experimental settings.



## **43. A2: Efficient Automated Attacker for Boosting Adversarial Training**

cs.CV

Accepted by NeurIPS2022

**SubmitDate**: 2022-10-07    [abs](http://arxiv.org/abs/2210.03543v1) [paper-pdf](http://arxiv.org/pdf/2210.03543v1)

**Authors**: Zhuoer Xu, Guanghui Zhu, Changhua Meng, Shiwen Cui, Zhenzhe Ying, Weiqiang Wang, Ming GU, Yihua Huang

**Abstract**: Based on the significant improvement of model robustness by AT (Adversarial Training), various variants have been proposed to further boost the performance. Well-recognized methods have focused on different components of AT (e.g., designing loss functions and leveraging additional unlabeled data). It is generally accepted that stronger perturbations yield more robust models. However, how to generate stronger perturbations efficiently is still missed. In this paper, we propose an efficient automated attacker called A2 to boost AT by generating the optimal perturbations on-the-fly during training. A2 is a parameterized automated attacker to search in the attacker space for the best attacker against the defense model and examples. Extensive experiments across different datasets demonstrate that A2 generates stronger perturbations with low extra cost and reliably improves the robustness of various AT methods against different attacks.



## **44. Adversarially Robust Prototypical Few-shot Segmentation with Neural-ODEs**

cs.CV

MICCAI 2022. arXiv admin note: substantial text overlap with  arXiv:2208.12428

**SubmitDate**: 2022-10-07    [abs](http://arxiv.org/abs/2210.03429v1) [paper-pdf](http://arxiv.org/pdf/2210.03429v1)

**Authors**: Prashant Pandey, Aleti Vardhan, Mustafa Chasmai, Tanuj Sur, Brejesh Lall

**Abstract**: Few-shot Learning (FSL) methods are being adopted in settings where data is not abundantly available. This is especially seen in medical domains where the annotations are expensive to obtain. Deep Neural Networks have been shown to be vulnerable to adversarial attacks. This is even more severe in the case of FSL due to the lack of a large number of training examples. In this paper, we provide a framework to make few-shot segmentation models adversarially robust in the medical domain where such attacks can severely impact the decisions made by clinicians who use them. We propose a novel robust few-shot segmentation framework, Prototypical Neural Ordinary Differential Equation (PNODE), that provides defense against gradient-based adversarial attacks. We show that our framework is more robust compared to traditional adversarial defense mechanisms such as adversarial training. Adversarial training involves increased training time and shows robustness to limited types of attacks depending on the type of adversarial examples seen during training. Our proposed framework generalises well to common adversarial attacks like FGSM, PGD and SMIA while having the model parameters comparable to the existing few-shot segmentation models. We show the effectiveness of our proposed approach on three publicly available multi-organ segmentation datasets in both in-domain and cross-domain settings by attacking the support and query sets without the need for ad-hoc adversarial training.



## **45. Semi-quantum private comparison and its generalization to the key agreement, summation, and anonymous ranking**

quant-ph

19 pages 5 tables

**SubmitDate**: 2022-10-07    [abs](http://arxiv.org/abs/2210.03421v1) [paper-pdf](http://arxiv.org/pdf/2210.03421v1)

**Authors**: Chong-Qiang Ye, Jian Li, Xiu-Bo Chen, Yanyan Hou, Zhou Wang

**Abstract**: Semi-quantum protocols construct connections between quantum users and ``classical'' users who can only perform certain ``classical'' operations. In this paper, we present a new semi-quantum private comparison protocol based on entangled states and single particles, which does not require pre-shared keys between the ``classical'' users to guarantee the security of their private data. By utilizing multi-particle entangled states and single particles, our protocol can be easily extended to multi-party scenarios to meet the requirements of multiple ``classical'' users who want to compare their private data. The security analysis shows that the protocol can effectively prevent attacks from outside eavesdroppers and adversarial participants. Besides, we generalize the proposed protocol to other semi-quantum protocols such as semi-quantum key agreement, semi-quantum summation, and semi-quantum anonymous ranking protocols. We compare and discuss the proposed protocols with previous similar protocols. The results show that our protocols satisfy the demands of their respective counterparts separately. Therefore, our protocols have a wide range of application scenarios.



## **46. Pre-trained Adversarial Perturbations**

cs.CV

**SubmitDate**: 2022-10-07    [abs](http://arxiv.org/abs/2210.03372v1) [paper-pdf](http://arxiv.org/pdf/2210.03372v1)

**Authors**: Yuanhao Ban, Yinpeng Dong

**Abstract**: Self-supervised pre-training has drawn increasing attention in recent years due to its superior performance on numerous downstream tasks after fine-tuning. However, it is well-known that deep learning models lack the robustness to adversarial examples, which can also invoke security issues to pre-trained models, despite being less explored. In this paper, we delve into the robustness of pre-trained models by introducing Pre-trained Adversarial Perturbations (PAPs), which are universal perturbations crafted for the pre-trained models to maintain the effectiveness when attacking fine-tuned ones without any knowledge of the downstream tasks. To this end, we propose a Low-Level Layer Lifting Attack (L4A) method to generate effective PAPs by lifting the neuron activations of low-level layers of the pre-trained models. Equipped with an enhanced noise augmentation strategy, L4A is effective at generating more transferable PAPs against fine-tuned models. Extensive experiments on typical pre-trained vision models and ten downstream tasks demonstrate that our method improves the attack success rate by a large margin compared with state-of-the-art methods.



## **47. Preprocessors Matter! Realistic Decision-Based Attacks on Machine Learning Systems**

cs.CR

Code can be found at  https://github.com/google-research/preprocessor-aware-black-box-attack

**SubmitDate**: 2022-10-07    [abs](http://arxiv.org/abs/2210.03297v1) [paper-pdf](http://arxiv.org/pdf/2210.03297v1)

**Authors**: Chawin Sitawarin, Florian Tramèr, Nicholas Carlini

**Abstract**: Decision-based adversarial attacks construct inputs that fool a machine-learning model into making targeted mispredictions by making only hard-label queries. For the most part, these attacks have been applied directly to isolated neural network models. However, in practice, machine learning models are just a component of a much larger system. By adding just a single preprocessor in front of a classifier, we find that state-of-the-art query-based attacks are as much as seven times less effective at attacking a prediction pipeline than attacking the machine learning model alone. Hence, attacks that are unaware of this invariance inevitably waste a large number of queries to re-discover or overcome it. We, therefore, develop techniques to first reverse-engineer the preprocessor and then use this extracted information to attack the end-to-end system. Our extraction method requires only a few hundred queries to learn the preprocessors used by most publicly available model pipelines, and our preprocessor-aware attacks recover the same efficacy as just attacking the model alone. The code can be found at https://github.com/google-research/preprocessor-aware-black-box-attack.



## **48. Adversarial Training for High-Stakes Reliability**

cs.LG

30 pages, 7 figures, draft NeurIPS version

**SubmitDate**: 2022-10-07    [abs](http://arxiv.org/abs/2205.01663v4) [paper-pdf](http://arxiv.org/pdf/2205.01663v4)

**Authors**: Daniel M. Ziegler, Seraphina Nix, Lawrence Chan, Tim Bauman, Peter Schmidt-Nielsen, Tao Lin, Adam Scherlis, Noa Nabeshima, Ben Weinstein-Raun, Daniel de Haas, Buck Shlegeris, Nate Thomas

**Abstract**: In the future, powerful AI systems may be deployed in high-stakes settings, where a single failure could be catastrophic. One technique for improving AI safety in high-stakes settings is adversarial training, which uses an adversary to generate examples to train on in order to achieve better worst-case performance.   In this work, we used a safe language generation task (``avoid injuries'') as a testbed for achieving high reliability through adversarial training. We created a series of adversarial training techniques -- including a tool that assists human adversaries -- to find and eliminate failures in a classifier that filters text completions suggested by a generator. In our task, we determined that we can set very conservative classifier thresholds without significantly impacting the quality of the filtered outputs. We found that adversarial training increased robustness to the adversarial attacks that we trained on -- doubling the time for our contractors to find adversarial examples both with our tool (from 13 to 26 minutes) and without (from 20 to 44 minutes) -- without affecting in-distribution performance.   We hope to see further work in the high-stakes reliability setting, including more powerful tools for enhancing human adversaries and better ways to measure high levels of reliability, until we can confidently rule out the possibility of catastrophic deployment-time failures of powerful models.



## **49. Truth Serum: Poisoning Machine Learning Models to Reveal Their Secrets**

cs.CR

ACM CCS 2022

**SubmitDate**: 2022-10-06    [abs](http://arxiv.org/abs/2204.00032v2) [paper-pdf](http://arxiv.org/pdf/2204.00032v2)

**Authors**: Florian Tramèr, Reza Shokri, Ayrton San Joaquin, Hoang Le, Matthew Jagielski, Sanghyun Hong, Nicholas Carlini

**Abstract**: We introduce a new class of attacks on machine learning models. We show that an adversary who can poison a training dataset can cause models trained on this dataset to leak significant private details of training points belonging to other parties. Our active inference attacks connect two independent lines of work targeting the integrity and privacy of machine learning training data.   Our attacks are effective across membership inference, attribute inference, and data extraction. For example, our targeted attacks can poison <0.1% of the training dataset to boost the performance of inference attacks by 1 to 2 orders of magnitude. Further, an adversary who controls a significant fraction of the training data (e.g., 50%) can launch untargeted attacks that enable 8x more precise inference on all other users' otherwise-private data points.   Our results cast doubts on the relevance of cryptographic privacy guarantees in multiparty computation protocols for machine learning, if parties can arbitrarily select their share of training data.



## **50. EvilScreen Attack: Smart TV Hijacking via Multi-channel Remote Control Mimicry**

cs.CR

**SubmitDate**: 2022-10-06    [abs](http://arxiv.org/abs/2210.03014v1) [paper-pdf](http://arxiv.org/pdf/2210.03014v1)

**Authors**: Yiwei Zhang, Siqi Ma, Tiancheng Chen, Juanru Li, Robert H. Deng, Elisa Bertino

**Abstract**: Modern smart TVs often communicate with their remote controls (including those smart phone simulated ones) using multiple wireless channels (e.g., Infrared, Bluetooth, and Wi-Fi). However, this multi-channel remote control communication introduces a new attack surface. An inherent security flaw is that remote controls of most smart TVs are designed to work in a benign environment rather than an adversarial one, and thus wireless communications between a smart TV and its remote controls are not strongly protected. Attackers could leverage such flaw to abuse the remote control communication and compromise smart TV systems. In this paper, we propose EvilScreen, a novel attack that exploits ill-protected remote control communications to access protected resources of a smart TV or even control the screen. EvilScreen exploits a multi-channel remote control mimicry vulnerability present in today smart TVs. Unlike other attacks, which compromise the TV system by exploiting code vulnerabilities or malicious third-party apps, EvilScreen directly reuses commands of different remote controls, combines them together to circumvent deployed authentication and isolation policies, and finally accesses or controls TV resources remotely. We evaluated eight mainstream smart TVs and found that they are all vulnerable to EvilScreen attacks, including a Samsung product adopting the ISO/IEC security specification.



