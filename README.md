# Latest Adversarial Attack Papers
**update at 2023-03-27 11:02:57**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. How many dimensions are required to find an adversarial example?**

cs.LG

Comments welcome!

**SubmitDate**: 2023-03-24    [abs](http://arxiv.org/abs/2303.14173v1) [paper-pdf](http://arxiv.org/pdf/2303.14173v1)

**Authors**: Charles Godfrey, Henry Kvinge, Elise Bishoff, Myles Mckay, Davis Brown, Tim Doster, Eleanor Byler

**Abstract**: Past work exploring adversarial vulnerability have focused on situations where an adversary can perturb all dimensions of model input. On the other hand, a range of recent works consider the case where either (i) an adversary can perturb a limited number of input parameters or (ii) a subset of modalities in a multimodal problem. In both of these cases, adversarial examples are effectively constrained to a subspace $V$ in the ambient input space $\mathcal{X}$. Motivated by this, in this work we investigate how adversarial vulnerability depends on $\dim(V)$. In particular, we show that the adversarial success of standard PGD attacks with $\ell^p$ norm constraints behaves like a monotonically increasing function of $\epsilon (\frac{\dim(V)}{\dim \mathcal{X}})^{\frac{1}{q}}$ where $\epsilon$ is the perturbation budget and $\frac{1}{p} + \frac{1}{q} =1$, provided $p > 1$ (the case $p=1$ presents additional subtleties which we analyze in some detail). This functional form can be easily derived from a simple toy linear model, and as such our results land further credence to arguments that adversarial examples are endemic to locally linear models on high dimensional spaces.



## **2. Adversarial Attack and Defense for Medical Image Analysis: Methods and Applications**

eess.IV

**SubmitDate**: 2023-03-24    [abs](http://arxiv.org/abs/2303.14133v1) [paper-pdf](http://arxiv.org/pdf/2303.14133v1)

**Authors**: Junhao Dong, Junxi Chen, Xiaohua Xie, Jianhuang Lai, Hao Chen

**Abstract**: Deep learning techniques have achieved superior performance in computer-aided medical image analysis, yet they are still vulnerable to imperceptible adversarial attacks, resulting in potential misdiagnosis in clinical practice. Oppositely, recent years have also witnessed remarkable progress in defense against these tailored adversarial examples in deep medical diagnosis systems. In this exposition, we present a comprehensive survey on recent advances in adversarial attack and defense for medical image analysis with a novel taxonomy in terms of the application scenario. We also provide a unified theoretical framework for different types of adversarial attack and defense methods for medical image analysis. For a fair comparison, we establish a new benchmark for adversarially robust medical diagnosis models obtained by adversarial training under various scenarios. To the best of our knowledge, this is the first survey paper that provides a thorough evaluation of adversarially robust medical diagnosis models. By analyzing qualitative and quantitative results, we conclude this survey with a detailed discussion of current challenges for adversarial attack and defense in medical image analysis systems to shed light on future research directions.



## **3. Improved Adversarial Training Through Adaptive Instance-wise Loss Smoothing**

cs.CV

12 pages, work in submission

**SubmitDate**: 2023-03-24    [abs](http://arxiv.org/abs/2303.14077v1) [paper-pdf](http://arxiv.org/pdf/2303.14077v1)

**Authors**: Lin Li, Michael Spratling

**Abstract**: Deep neural networks can be easily fooled into making incorrect predictions through corruption of the input by adversarial perturbations: human-imperceptible artificial noise. So far adversarial training has been the most successful defense against such adversarial attacks. This work focuses on improving adversarial training to boost adversarial robustness. We first analyze, from an instance-wise perspective, how adversarial vulnerability evolves during adversarial training. We find that during training an overall reduction of adversarial loss is achieved by sacrificing a considerable proportion of training samples to be more vulnerable to adversarial attack, which results in an uneven distribution of adversarial vulnerability among data. Such "uneven vulnerability", is prevalent across several popular robust training methods and, more importantly, relates to overfitting in adversarial training. Motivated by this observation, we propose a new adversarial training method: Instance-adaptive Smoothness Enhanced Adversarial Training (ISEAT). It jointly smooths both input and weight loss landscapes in an adaptive, instance-specific, way to enhance robustness more for those samples with higher adversarial vulnerability. Extensive experiments demonstrate the superiority of our method over existing defense methods. Noticeably, our method, when combined with the latest data augmentation and semi-supervised learning techniques, achieves state-of-the-art robustness against $\ell_{\infty}$-norm constrained attacks on CIFAR10 of 59.32% for Wide ResNet34-10 without extra data, and 61.55% for Wide ResNet28-10 with extra data. Code is available at https://github.com/TreeLLi/Instance-adaptive-Smoothness-Enhanced-AT.



## **4. PIAT: Parameter Interpolation based Adversarial Training for Image Classification**

cs.CV

**SubmitDate**: 2023-03-24    [abs](http://arxiv.org/abs/2303.13955v1) [paper-pdf](http://arxiv.org/pdf/2303.13955v1)

**Authors**: Kun He, Xin Liu, Yichen Yang, Zhou Qin, Weigao Wen, Hui Xue, John E. Hopcroft

**Abstract**: Adversarial training has been demonstrated to be the most effective approach to defend against adversarial attacks. However, existing adversarial training methods show apparent oscillations and overfitting issue in the training process, degrading the defense efficacy. In this work, we propose a novel framework, termed Parameter Interpolation based Adversarial Training (PIAT), that makes full use of the historical information during training. Specifically, at the end of each epoch, PIAT tunes the model parameters as the interpolation of the parameters of the previous and current epochs. Besides, we suggest to use the Normalized Mean Square Error (NMSE) to further improve the robustness by aligning the clean and adversarial examples. Compared with other regularization methods, NMSE focuses more on the relative magnitude of the logits rather than the absolute magnitude. Extensive experiments on several benchmark datasets and various networks show that our method could prominently improve the model robustness and reduce the generalization error. Moreover, our framework is general and could further boost the robust accuracy when combined with other adversarial training methods.



## **5. EC-CFI: Control-Flow Integrity via Code Encryption Counteracting Fault Attacks**

cs.CR

Accepted at HOST'23

**SubmitDate**: 2023-03-24    [abs](http://arxiv.org/abs/2301.13760v2) [paper-pdf](http://arxiv.org/pdf/2301.13760v2)

**Authors**: Pascal Nasahl, Salmin Sultana, Hans Liljestrand, Karanvir Grewal, Michael LeMay, David M. Durham, David Schrammel, Stefan Mangard

**Abstract**: Fault attacks enable adversaries to manipulate the control-flow of security-critical applications. By inducing targeted faults into the CPU, the software's call graph can be escaped and the control-flow can be redirected to arbitrary functions inside the program. To protect the control-flow from these attacks, dedicated fault control-flow integrity (CFI) countermeasures are commonly deployed. However, these schemes either have high detection latencies or require intrusive hardware changes. In this paper, we present EC-CFI, a software-based cryptographically enforced CFI scheme with no detection latency utilizing hardware features of recent Intel platforms. Our EC-CFI prototype is designed to prevent an adversary from escaping the program's call graph using faults by encrypting each function with a different key before execution. At runtime, the instrumented program dynamically derives the decryption key, ensuring that the code only can be successfully decrypted when the program follows the intended call graph. To enable this level of protection on Intel commodity systems, we introduce extended page table (EPT) aliasing allowing us to achieve function-granular encryption by combing Intel's TME-MK and virtualization technology. We open-source our custom LLVM-based toolchain automatically protecting arbitrary programs with EC-CFI. Furthermore, we evaluate our EPT aliasing approach with the SPEC CPU2017 and Embench-IoT benchmarks and discuss and evaluate potential TME-MK hardware changes minimizing runtime overheads.



## **6. SCRAMBLE-CFI: Mitigating Fault-Induced Control-Flow Attacks on OpenTitan**

cs.CR

Accepted at GLSVLSI'23

**SubmitDate**: 2023-03-24    [abs](http://arxiv.org/abs/2303.03711v3) [paper-pdf](http://arxiv.org/pdf/2303.03711v3)

**Authors**: Pascal Nasahl, Stefan Mangard

**Abstract**: Secure elements physically exposed to adversaries are frequently targeted by fault attacks. These attacks can be utilized to hijack the control-flow of software allowing the attacker to bypass security measures, extract sensitive data, or gain full code execution. In this paper, we systematically analyze the threat vector of fault-induced control-flow manipulations on the open-source OpenTitan secure element. Our thorough analysis reveals that current countermeasures of this chip either induce large area overheads or still cannot prevent the attacker from exploiting the identified threats. In this context, we introduce SCRAMBLE-CFI, an encryption-based control-flow integrity scheme utilizing existing hardware features of OpenTitan. SCRAMBLE-CFI confines, with minimal hardware overhead, the impact of fault-induced control-flow attacks by encrypting each function with a different encryption tweak at load-time. At runtime, code only can be successfully decrypted when the correct decryption tweak is active. We open-source our hardware changes and release our LLVM toolchain automatically protecting programs. Our analysis shows that SCRAMBLE-CFI complementarily enhances security guarantees of OpenTitan with a negligible hardware overhead of less than 3.97 % and a runtime overhead of 7.02 % for the Embench-IoT benchmarks.



## **7. Foiling Explanations in Deep Neural Networks**

cs.CV

**SubmitDate**: 2023-03-24    [abs](http://arxiv.org/abs/2211.14860v2) [paper-pdf](http://arxiv.org/pdf/2211.14860v2)

**Authors**: Snir Vitrack Tamam, Raz Lapid, Moshe Sipper

**Abstract**: Deep neural networks (DNNs) have greatly impacted numerous fields over the past decade. Yet despite exhibiting superb performance over many problems, their black-box nature still poses a significant challenge with respect to explainability. Indeed, explainable artificial intelligence (XAI) is crucial in several fields, wherein the answer alone -- sans a reasoning of how said answer was derived -- is of little value. This paper uncovers a troubling property of explanation methods for image-based DNNs: by making small visual changes to the input image -- hardly influencing the network's output -- we demonstrate how explanations may be arbitrarily manipulated through the use of evolution strategies. Our novel algorithm, AttaXAI, a model-agnostic, adversarial attack on XAI algorithms, only requires access to the output logits of a classifier and to the explanation map; these weak assumptions render our approach highly useful where real-world models and data are concerned. We compare our method's performance on two benchmark datasets -- CIFAR100 and ImageNet -- using four different pretrained deep-learning models: VGG16-CIFAR100, VGG16-ImageNet, MobileNet-CIFAR100, and Inception-v3-ImageNet. We find that the XAI methods can be manipulated without the use of gradients or other model internals. Our novel algorithm is successfully able to manipulate an image in a manner imperceptible to the human eye, such that the XAI method outputs a specific explanation map. To our knowledge, this is the first such method in a black-box setting, and we believe it has significant value where explainability is desired, required, or legally mandatory.



## **8. Effective black box adversarial attack with handcrafted kernels**

cs.CV

12 pages, 5 figures, 3 tables, IWANN conference

**SubmitDate**: 2023-03-24    [abs](http://arxiv.org/abs/2303.13887v1) [paper-pdf](http://arxiv.org/pdf/2303.13887v1)

**Authors**: Petr Dvořáček, Petr Hurtik, Petra Števuliáková

**Abstract**: We propose a new, simple framework for crafting adversarial examples for black box attacks. The idea is to simulate the substitution model with a non-trainable model compounded of just one layer of handcrafted convolutional kernels and then train the generator neural network to maximize the distance of the outputs for the original and generated adversarial image. We show that fooling the prediction of the first layer causes the whole network to be fooled and decreases its accuracy on adversarial inputs. Moreover, we do not train the neural network to obtain the first convolutional layer kernels, but we create them using the technique of F-transform. Therefore, our method is very time and resource effective.



## **9. Physically Adversarial Infrared Patches with Learnable Shapes and Locations**

cs.CV

accepted by CVPR2023

**SubmitDate**: 2023-03-24    [abs](http://arxiv.org/abs/2303.13868v1) [paper-pdf](http://arxiv.org/pdf/2303.13868v1)

**Authors**: Wei Xingxing, Yu Jie, Huang Yao

**Abstract**: Owing to the extensive application of infrared object detectors in the safety-critical tasks, it is necessary to evaluate their robustness against adversarial examples in the real world. However, current few physical infrared attacks are complicated to implement in practical application because of their complex transformation from digital world to physical world. To address this issue, in this paper, we propose a physically feasible infrared attack method called "adversarial infrared patches". Considering the imaging mechanism of infrared cameras by capturing objects' thermal radiation, adversarial infrared patches conduct attacks by attaching a patch of thermal insulation materials on the target object to manipulate its thermal distribution. To enhance adversarial attacks, we present a novel aggregation regularization to guide the simultaneous learning for the patch' shape and location on the target object. Thus, a simple gradient-based optimization can be adapted to solve for them. We verify adversarial infrared patches in different object detection tasks with various object detectors. Experimental results show that our method achieves more than 90\% Attack Success Rate (ASR) versus the pedestrian detector and vehicle detector in the physical environment, where the objects are captured in different angles, distances, postures, and scenes. More importantly, adversarial infrared patch is easy to implement, and it only needs 0.5 hours to be constructed in the physical world, which verifies its effectiveness and efficiency.



## **10. Feature Separation and Recalibration for Adversarial Robustness**

cs.CV

CVPR 2023 (Highlight)

**SubmitDate**: 2023-03-24    [abs](http://arxiv.org/abs/2303.13846v1) [paper-pdf](http://arxiv.org/pdf/2303.13846v1)

**Authors**: Woo Jae Kim, Yoonki Cho, Junsik Jung, Sung-Eui Yoon

**Abstract**: Deep neural networks are susceptible to adversarial attacks due to the accumulation of perturbations in the feature level, and numerous works have boosted model robustness by deactivating the non-robust feature activations that cause model mispredictions. However, we claim that these malicious activations still contain discriminative cues and that with recalibration, they can capture additional useful information for correct model predictions. To this end, we propose a novel, easy-to-plugin approach named Feature Separation and Recalibration (FSR) that recalibrates the malicious, non-robust activations for more robust feature maps through Separation and Recalibration. The Separation part disentangles the input feature map into the robust feature with activations that help the model make correct predictions and the non-robust feature with activations that are responsible for model mispredictions upon adversarial attack. The Recalibration part then adjusts the non-robust activations to restore the potentially useful cues for model predictions. Extensive experiments verify the superiority of FSR compared to traditional deactivation techniques and demonstrate that it improves the robustness of existing adversarial training methods by up to 8.57% with small computational overhead. Codes are available at https://github.com/wkim97/FSR.



## **11. Near Optimal Adversarial Attack on UCB Bandits**

cs.LG

**SubmitDate**: 2023-03-24    [abs](http://arxiv.org/abs/2008.09312v3) [paper-pdf](http://arxiv.org/pdf/2008.09312v3)

**Authors**: Shiliang Zuo

**Abstract**: We consider a stochastic multi-arm bandit problem where rewards are subject to adversarial corruption. We propose a novel attack strategy that manipulates a UCB principle into pulling some non-optimal target arm $T - o(T)$ times with a cumulative cost that scales as $\sqrt{\log T}$, where $T$ is the number of rounds. We also prove the first lower bound on the cumulative attack cost. Our lower bound matches our upper bound up to $\log \log T$ factors, showing our attack to be near optimal.



## **12. RamBoAttack: A Robust Query Efficient Deep Neural Network Decision Exploit**

cs.LG

Published in Network and Distributed System Security (NDSS) Symposium  2022. Code is available at https://ramboattack.github.io/

**SubmitDate**: 2023-03-24    [abs](http://arxiv.org/abs/2112.05282v3) [paper-pdf](http://arxiv.org/pdf/2112.05282v3)

**Authors**: Viet Quoc Vo, Ehsan Abbasnejad, Damith C. Ranasinghe

**Abstract**: Machine learning models are critically susceptible to evasion attacks from adversarial examples. Generally, adversarial examples, modified inputs deceptively similar to the original input, are constructed under whitebox settings by adversaries with full access to the model. However, recent attacks have shown a remarkable reduction in query numbers to craft adversarial examples using blackbox attacks. Particularly, alarming is the ability to exploit the classification decision from the access interface of a trained model provided by a growing number of Machine Learning as a Service providers including Google, Microsoft, IBM and used by a plethora of applications incorporating these models. The ability of an adversary to exploit only the predicted label from a model to craft adversarial examples is distinguished as a decision-based attack. In our study, we first deep dive into recent state-of-the-art decision-based attacks in ICLR and SP to highlight the costly nature of discovering low distortion adversarial employing gradient estimation methods. We develop a robust query efficient attack capable of avoiding entrapment in a local minimum and misdirection from noisy gradients seen in gradient estimation methods. The attack method we propose, RamBoAttack, exploits the notion of Randomized Block Coordinate Descent to explore the hidden classifier manifold, targeting perturbations to manipulate only localized input features to address the issues of gradient estimation methods. Importantly, the RamBoAttack is more robust to the different sample inputs available to an adversary and the targeted class. Overall, for a given target class, RamBoAttack is demonstrated to be more robust at achieving a lower distortion within a given query budget. We curate our extensive results using the large-scale high-resolution ImageNet dataset and open-source our attack, test samples and artifacts on GitHub.



## **13. Query Efficient Decision Based Sparse Attacks Against Black-Box Deep Learning Models**

cs.LG

Published as a conference paper at the International Conference on  Learning Representations (ICLR 2022). Code is available at  https://sparseevoattack.github.io/

**SubmitDate**: 2023-03-24    [abs](http://arxiv.org/abs/2202.00091v2) [paper-pdf](http://arxiv.org/pdf/2202.00091v2)

**Authors**: Viet Quoc Vo, Ehsan Abbasnejad, Damith C. Ranasinghe

**Abstract**: Despite our best efforts, deep learning models remain highly vulnerable to even tiny adversarial perturbations applied to the inputs. The ability to extract information from solely the output of a machine learning model to craft adversarial perturbations to black-box models is a practical threat against real-world systems, such as autonomous cars or machine learning models exposed as a service (MLaaS). Of particular interest are sparse attacks. The realization of sparse attacks in black-box models demonstrates that machine learning models are more vulnerable than we believe. Because these attacks aim to minimize the number of perturbed pixels measured by l_0 norm-required to mislead a model by solely observing the decision (the predicted label) returned to a model query; the so-called decision-based attack setting. But, such an attack leads to an NP-hard optimization problem. We develop an evolution-based algorithm-SparseEvo-for the problem and evaluate against both convolutional deep neural networks and vision transformers. Notably, vision transformers are yet to be investigated under a decision-based attack setting. SparseEvo requires significantly fewer model queries than the state-of-the-art sparse attack Pointwise for both untargeted and targeted attacks. The attack algorithm, although conceptually simple, is also competitive with only a limited query budget against the state-of-the-art gradient-based whitebox attacks in standard computer vision tasks such as ImageNet. Importantly, the query efficient SparseEvo, along with decision-based attacks, in general, raise new questions regarding the safety of deployed systems and poses new directions to study and understand the robustness of machine learning models.



## **14. CBA: Contextual Background Attack against Optical Aerial Detection in the Physical World**

cs.CV

**SubmitDate**: 2023-03-24    [abs](http://arxiv.org/abs/2302.13519v3) [paper-pdf](http://arxiv.org/pdf/2302.13519v3)

**Authors**: Jiawei Lian, Xiaofei Wang, Yuru Su, Mingyang Ma, Shaohui Mei

**Abstract**: Patch-based physical attacks have increasingly aroused concerns.   However, most existing methods focus on obscuring targets captured on the ground, and some of these methods are simply extended to deceive aerial detectors.   They smear the targeted objects in the physical world with the elaborated adversarial patches, which can only slightly sway the aerial detectors' prediction and with weak attack transferability.   To address the above issues, we propose to perform Contextual Background Attack (CBA), a novel physical attack framework against aerial detection, which can achieve strong attack efficacy and transferability in the physical world even without smudging the interested objects at all.   Specifically, the targets of interest, i.e. the aircraft in aerial images, are adopted to mask adversarial patches.   The pixels outside the mask area are optimized to make the generated adversarial patches closely cover the critical contextual background area for detection, which contributes to gifting adversarial patches with more robust and transferable attack potency in the real world.   To further strengthen the attack performance, the adversarial patches are forced to be outside targets during training, by which the detected objects of interest, both on and outside patches, benefit the accumulation of attack efficacy.   Consequently, the sophisticatedly designed patches are gifted with solid fooling efficacy against objects both on and outside the adversarial patches simultaneously.   Extensive proportionally scaled experiments are performed in physical scenarios, demonstrating the superiority and potential of the proposed framework for physical attacks.   We expect that the proposed physical attack method will serve as a benchmark for assessing the adversarial robustness of diverse aerial detectors and defense methods.



## **15. TrojViT: Trojan Insertion in Vision Transformers**

cs.LG

10 pages, 4 figures, 11 tables

**SubmitDate**: 2023-03-23    [abs](http://arxiv.org/abs/2208.13049v3) [paper-pdf](http://arxiv.org/pdf/2208.13049v3)

**Authors**: Mengxin Zheng, Qian Lou, Lei Jiang

**Abstract**: Vision Transformers (ViTs) have demonstrated the state-of-the-art performance in various vision-related tasks. The success of ViTs motivates adversaries to perform backdoor attacks on ViTs. Although the vulnerability of traditional CNNs to backdoor attacks is well-known, backdoor attacks on ViTs are seldom-studied. Compared to CNNs capturing pixel-wise local features by convolutions, ViTs extract global context information through patches and attentions. Na\"ively transplanting CNN-specific backdoor attacks to ViTs yields only a low clean data accuracy and a low attack success rate. In this paper, we propose a stealth and practical ViT-specific backdoor attack $TrojViT$. Rather than an area-wise trigger used by CNN-specific backdoor attacks, TrojViT generates a patch-wise trigger designed to build a Trojan composed of some vulnerable bits on the parameters of a ViT stored in DRAM memory through patch salience ranking and attention-target loss. TrojViT further uses minimum-tuned parameter update to reduce the bit number of the Trojan. Once the attacker inserts the Trojan into the ViT model by flipping the vulnerable bits, the ViT model still produces normal inference accuracy with benign inputs. But when the attacker embeds a trigger into an input, the ViT model is forced to classify the input to a predefined target class. We show that flipping only few vulnerable bits identified by TrojViT on a ViT model using the well-known RowHammer can transform the model into a backdoored one. We perform extensive experiments of multiple datasets on various ViT models. TrojViT can classify $99.64\%$ of test images to a target class by flipping $345$ bits on a ViT for ImageNet.



## **16. Adversarial Robustness and Feature Impact Analysis for Driver Drowsiness Detection**

cs.LG

10 pages, 2 tables, 3 figures, AIME 2023 conference

**SubmitDate**: 2023-03-23    [abs](http://arxiv.org/abs/2303.13649v1) [paper-pdf](http://arxiv.org/pdf/2303.13649v1)

**Authors**: João Vitorino, Lourenço Rodrigues, Eva Maia, Isabel Praça, André Lourenço

**Abstract**: Drowsy driving is a major cause of road accidents, but drivers are dismissive of the impact that fatigue can have on their reaction times. To detect drowsiness before any impairment occurs, a promising strategy is using Machine Learning (ML) to monitor Heart Rate Variability (HRV) signals. This work presents multiple experiments with different HRV time windows and ML models, a feature impact analysis using Shapley Additive Explanations (SHAP), and an adversarial robustness analysis to assess their reliability when processing faulty input data and perturbed HRV signals. The most reliable model was Extreme Gradient Boosting (XGB) and the optimal time window had between 120 and 150 seconds. Furthermore, SHAP enabled the selection of the 18 most impactful features and the training of new smaller models that achieved a performance as good as the initial ones. Despite the susceptibility of all models to adversarial attacks, adversarial training enabled them to preserve significantly higher results, especially XGB. Therefore, ML models can significantly benefit from realistic adversarial training to provide a more robust driver drowsiness detection.



## **17. Efficient Symbolic Reasoning for Neural-Network Verification**

cs.AI

**SubmitDate**: 2023-03-23    [abs](http://arxiv.org/abs/2303.13588v1) [paper-pdf](http://arxiv.org/pdf/2303.13588v1)

**Authors**: Zi Wang, Somesh Jha, Krishnamurthy, Dvijotham

**Abstract**: The neural network has become an integral part of modern software systems. However, they still suffer from various problems, in particular, vulnerability to adversarial attacks. In this work, we present a novel program reasoning framework for neural-network verification, which we refer to as symbolic reasoning. The key components of our framework are the use of the symbolic domain and the quadratic relation. The symbolic domain has very flexible semantics, and the quadratic relation is quite expressive. They allow us to encode many verification problems for neural networks as quadratic programs. Our scheme then relaxes the quadratic programs to semidefinite programs, which can be efficiently solved. This framework allows us to verify various neural-network properties under different scenarios, especially those that appear challenging for non-symbolic domains. Moreover, it introduces new representations and perspectives for the verification tasks. We believe that our framework can bring new theoretical insights and practical tools to verification problems for neural networks.



## **18. Symmetries, flat minima, and the conserved quantities of gradient flow**

cs.LG

To appear at ICLR 2023

**SubmitDate**: 2023-03-23    [abs](http://arxiv.org/abs/2210.17216v2) [paper-pdf](http://arxiv.org/pdf/2210.17216v2)

**Authors**: Bo Zhao, Iordan Ganev, Robin Walters, Rose Yu, Nima Dehmamy

**Abstract**: Empirical studies of the loss landscape of deep networks have revealed that many local minima are connected through low-loss valleys. Yet, little is known about the theoretical origin of such valleys. We present a general framework for finding continuous symmetries in the parameter space, which carve out low-loss valleys. Our framework uses equivariances of the activation functions and can be applied to different layer architectures. To generalize this framework to nonlinear neural networks, we introduce a novel set of nonlinear, data-dependent symmetries. These symmetries can transform a trained model such that it performs similarly on new samples, which allows ensemble building that improves robustness under certain adversarial attacks. We then show that conserved quantities associated with linear symmetries can be used to define coordinates along low-loss valleys. The conserved quantities help reveal that using common initialization methods, gradient flow only explores a small part of the global minimum. By relating conserved quantities to convergence rate and sharpness of the minimum, we provide insights on how initialization impacts convergence and generalizability.



## **19. Decentralized Adversarial Training over Graphs**

cs.LG

arXiv admin note: text overlap with arXiv:2303.01936

**SubmitDate**: 2023-03-23    [abs](http://arxiv.org/abs/2303.13326v1) [paper-pdf](http://arxiv.org/pdf/2303.13326v1)

**Authors**: Ying Cao, Elsa Rizk, Stefan Vlaski, Ali H. Sayed

**Abstract**: The vulnerability of machine learning models to adversarial attacks has been attracting considerable attention in recent years. Most existing studies focus on the behavior of stand-alone single-agent learners. In comparison, this work studies adversarial training over graphs, where individual agents are subjected to perturbations of varied strength levels across space. It is expected that interactions by linked agents, and the heterogeneity of the attack models that are possible over the graph, can help enhance robustness in view of the coordination power of the group. Using a min-max formulation of diffusion learning, we develop a decentralized adversarial training framework for multi-agent systems. We analyze the convergence properties of the proposed scheme for both convex and non-convex environments, and illustrate the enhanced robustness to adversarial attacks.



## **20. Source-independent quantum random number generator against tailored detector blinding attacks**

quant-ph

14 pages, 6 figures, 6 tables, comments are welcome

**SubmitDate**: 2023-03-23    [abs](http://arxiv.org/abs/2204.12156v2) [paper-pdf](http://arxiv.org/pdf/2204.12156v2)

**Authors**: Wen-Bo Liu, Yu-Shuo Lu, Yao Fu, Si-Cheng Huang, Ze-Jie Yin, Kun Jiang, Hua-Lei Yin, Zeng-Bing Chen

**Abstract**: Randomness, mainly in the form of random numbers, is the fundamental prerequisite for the security of many cryptographic tasks. Quantum randomness can be extracted even if adversaries are fully aware of the protocol and even control the randomness source. However, an adversary can further manipulate the randomness via tailored detector blinding attacks, which are hacking attacks suffered by protocols with trusted detectors. Here, by treating no-click events as valid events, we propose a quantum random number generation protocol that can simultaneously address source vulnerability and ferocious tailored detector blinding attacks. The method can be extended to high-dimensional random number generation. We experimentally demonstrate the ability of our protocol to generate random numbers for two-dimensional measurement with a generation speed of 0.1 bit per pulse.



## **21. Watch Out for the Confusing Faces: Detecting Face Swapping with the Probability Distribution of Face Identification Models**

cs.CV

**SubmitDate**: 2023-03-23    [abs](http://arxiv.org/abs/2303.13131v1) [paper-pdf](http://arxiv.org/pdf/2303.13131v1)

**Authors**: Yuxuan Duan, Xuhong Zhang, Chuer Yu, Zonghui Wang, Shouling Ji, Wenzhi Chen

**Abstract**: Recently, face swapping has been developing rapidly and achieved a surprising reality, raising concerns about fake content. As a countermeasure, various detection approaches have been proposed and achieved promising performance. However, most existing detectors struggle to maintain performance on unseen face swapping methods and low-quality images. Apart from the generalization problem, current detection approaches have been shown vulnerable to evasion attacks crafted by detection-aware manipulators. Lack of robustness under adversary scenarios leaves threats for applying face swapping detection in real world. In this paper, we propose a novel face swapping detection approach based on face identification probability distributions, coined as IdP_FSD, to improve the generalization and robustness. IdP_FSD is specially designed for detecting swapped faces whose identities belong to a finite set, which is meaningful in real-world applications. Compared with previous general detection methods, we make use of the available real faces with concerned identities and require no fake samples for training. IdP_FSD exploits face swapping's common nature that the identity of swapped face combines that of two faces involved in swapping. We reflect this nature with the confusion of a face identification model and measure the confusion with the maximum value of the output probability distribution. What's more, to defend our detector under adversary scenarios, an attention-based finetuning scheme is proposed for the face identification models used in IdP_FSD. Extensive experiments show that the proposed IdP_FSD not only achieves high detection performance on different benchmark datasets and image qualities but also raises the bar for manipulators to evade the detection.



## **22. Patch of Invisibility: Naturalistic Black-Box Adversarial Attacks on Object Detectors**

cs.CV

**SubmitDate**: 2023-03-23    [abs](http://arxiv.org/abs/2303.04238v3) [paper-pdf](http://arxiv.org/pdf/2303.04238v3)

**Authors**: Raz Lapid, Moshe Sipper

**Abstract**: Adversarial attacks on deep-learning models have been receiving increased attention in recent years. Work in this area has mostly focused on gradient-based techniques, so-called white-box attacks, wherein the attacker has access to the targeted model's internal parameters; such an assumption is usually unrealistic in the real world. Some attacks additionally use the entire pixel space to fool a given model, which is neither practical nor physical (i.e., real-world). On the contrary, we propose herein a gradient-free method that uses the learned image manifold of a pretrained generative adversarial network (GAN) to generate naturalistic physical adversarial patches for object detectors. We show that our proposed method works both digitally and physically.



## **23. BlockFW -- Towards Blockchain-based Rule-Sharing Firewall**

cs.CR

The 16th International Conference on Emerging Security Information,  Systems and Technologies (SECURWARE 2022), pp. 70-75, IARIA 2022

**SubmitDate**: 2023-03-23    [abs](http://arxiv.org/abs/2303.13073v1) [paper-pdf](http://arxiv.org/pdf/2303.13073v1)

**Authors**: Wei-Yang Chiu, Weizhi Meng

**Abstract**: Central-managed security mechanisms are often utilized in many organizations, but such server is also a security breaking point. This is because the server has the authority for all nodes that share the security protection. Hence if the attackers successfully tamper the server, the organization will be in trouble. Also, the settings and policies saved on the server are usually not cryptographically secured and ensured with hash. Thus, changing the settings from alternative way is feasible, without causing the security solution to raise any alarms. To mitigate these issues, in this work, we develop BlockFW - a blockchain-based rule sharing firewall to create a managed security mechanism, which provides validation and monitoring from multiple nodes. For BlockFW, all occurred transactions are cryptographically protected to ensure its integrity, making tampering attempts in utmost challenging for attackers. In the evaluation, we explore the performance of BlockFW under several adversarial conditions and demonstrate its effectiveness.



## **24. Semantic Image Attack for Visual Model Diagnosis**

cs.CV

Initial version submitted to NeurIPS 2022

**SubmitDate**: 2023-03-23    [abs](http://arxiv.org/abs/2303.13010v1) [paper-pdf](http://arxiv.org/pdf/2303.13010v1)

**Authors**: Jinqi Luo, Zhaoning Wang, Chen Henry Wu, Dong Huang, Fernando De la Torre

**Abstract**: In practice, metric analysis on a specific train and test dataset does not guarantee reliable or fair ML models. This is partially due to the fact that obtaining a balanced, diverse, and perfectly labeled dataset is typically expensive, time-consuming, and error-prone. Rather than relying on a carefully designed test set to assess ML models' failures, fairness, or robustness, this paper proposes Semantic Image Attack (SIA), a method based on the adversarial attack that provides semantic adversarial images to allow model diagnosis, interpretability, and robustness. Traditional adversarial training is a popular methodology for robustifying ML models against attacks. However, existing adversarial methods do not combine the two aspects that enable the interpretation and analysis of the model's flaws: semantic traceability and perceptual quality. SIA combines the two features via iterative gradient ascent on a predefined semantic attribute space and the image space. We illustrate the validity of our approach in three scenarios for keypoint detection and classification. (1) Model diagnosis: SIA generates a histogram of attributes that highlights the semantic vulnerability of the ML model (i.e., attributes that make the model fail). (2) Stronger attacks: SIA generates adversarial examples with visually interpretable attributes that lead to higher attack success rates than baseline methods. The adversarial training on SIA improves the transferable robustness across different gradient-based attacks. (3) Robustness to imbalanced datasets: we use SIA to augment the underrepresented classes, which outperforms strong augmentation and re-balancing baselines.



## **25. Connected Superlevel Set in (Deep) Reinforcement Learning and its Application to Minimax Theorems**

cs.LG

**SubmitDate**: 2023-03-23    [abs](http://arxiv.org/abs/2303.12981v1) [paper-pdf](http://arxiv.org/pdf/2303.12981v1)

**Authors**: Sihan Zeng, Thinh T. Doan, Justin Romberg

**Abstract**: The aim of this paper is to improve the understanding of the optimization landscape for policy optimization problems in reinforcement learning. Specifically, we show that the superlevel set of the objective function with respect to the policy parameter is always a connected set both in the tabular setting and under policies represented by a class of neural networks. In addition, we show that the optimization objective as a function of the policy parameter and reward satisfies a stronger "equiconnectedness" property. To our best knowledge, these are novel and previously unknown discoveries.   We present an application of the connectedness of these superlevel sets to the derivation of minimax theorems for robust reinforcement learning. We show that any minimax optimization program which is convex on one side and is equiconnected on the other side observes the minimax equality (i.e. has a Nash equilibrium). We find that this exact structure is exhibited by an interesting robust reinforcement learning problem under an adversarial reward attack, and the validity of its minimax equality immediately follows. This is the first time such a result is established in the literature.



## **26. Test-time Defense against Adversarial Attacks: Detection and Reconstruction of Adversarial Examples via Masked Autoencoder**

cs.CV

**SubmitDate**: 2023-03-22    [abs](http://arxiv.org/abs/2303.12848v1) [paper-pdf](http://arxiv.org/pdf/2303.12848v1)

**Authors**: Yun-Yun Tsai, Ju-Chin Chao, Albert Wen, Zhaoyuan Yang, Chengzhi Mao, Tapan Shah, Junfeng Yang

**Abstract**: Existing defense methods against adversarial attacks can be categorized into training time and test time defenses. Training time defense, i.e., adversarial training, requires a significant amount of extra time for training and is often not able to be generalized to unseen attacks. On the other hand, test time defense by test time weight adaptation requires access to perform gradient descent on (part of) the model weights, which could be infeasible for models with frozen weights. To address these challenges, we propose DRAM, a novel defense method to Detect and Reconstruct multiple types of Adversarial attacks via Masked autoencoder (MAE). We demonstrate how to use MAE losses to build a KS-test to detect adversarial attacks. Moreover, the MAE losses can be used to repair adversarial samples from unseen attack types. In this sense, DRAM neither requires model weight updates in test time nor augments the training set with more adversarial samples. Evaluating DRAM on the large-scale ImageNet data, we achieve the best detection rate of 82% on average on eight types of adversarial attacks compared with other detection baselines. For reconstruction, DRAM improves the robust accuracy by 6% ~ 41% for Standard ResNet50 and 3% ~ 8% for Robust ResNet50 compared with other self-supervision tasks, such as rotation prediction and contrastive learning.



## **27. Evaluating the Role of Target Arguments in Rumour Stance Classification**

cs.CL

**SubmitDate**: 2023-03-22    [abs](http://arxiv.org/abs/2303.12665v1) [paper-pdf](http://arxiv.org/pdf/2303.12665v1)

**Authors**: Yue Li, Carolina Scarton

**Abstract**: Considering a conversation thread, stance classification aims to identify the opinion (e.g. agree or disagree) of replies towards a given target. The target of the stance is expected to be an essential component in this task, being one of the main factors that make it different from sentiment analysis. However, a recent study shows that a target-oblivious model outperforms target-aware models, suggesting that targets are not useful when predicting stance. This paper re-examines this phenomenon for rumour stance classification (RSC) on social media, where a target is a rumour story implied by the source tweet in the conversation. We propose adversarial attacks in the test data, aiming to assess the models robustness and evaluate the role of the data in the models performance. Results show that state-of-the-art models, including approaches that use the entire conversation thread, overly relying on superficial signals. Our hypothesis is that the naturally high occurrence of target-independent direct replies in RSC (e.g. "this is fake" or just "fake") results in the impressive performance of target-oblivious models, highlighting the risk of target instances being treated as noise during training.



## **28. Reliable and Efficient Evaluation of Adversarial Robustness for Deep Hashing-Based Retrieval**

cs.CV

arXiv admin note: text overlap with arXiv:2204.10779

**SubmitDate**: 2023-03-22    [abs](http://arxiv.org/abs/2303.12658v1) [paper-pdf](http://arxiv.org/pdf/2303.12658v1)

**Authors**: Xunguang Wang, Jiawang Bai, Xinyue Xu, Xiaomeng Li

**Abstract**: Deep hashing has been extensively applied to massive image retrieval due to its efficiency and effectiveness. Recently, several adversarial attacks have been presented to reveal the vulnerability of deep hashing models against adversarial examples. However, existing attack methods suffer from degraded performance or inefficiency because they underutilize the semantic relations between original samples or spend a lot of time learning these relations with a deep neural network. In this paper, we propose a novel Pharos-guided Attack, dubbed PgA, to evaluate the adversarial robustness of deep hashing networks reliably and efficiently. Specifically, we design pharos code to represent the semantics of the benign image, which preserves the similarity to semantically relevant samples and dissimilarity to irrelevant ones. It is proven that we can quickly calculate the pharos code via a simple math formula. Accordingly, PgA can directly conduct a reliable and efficient attack on deep hashing-based retrieval by maximizing the similarity between the hash code of the adversarial example and the pharos code. Extensive experiments on the benchmark datasets verify that the proposed algorithm outperforms the prior state-of-the-arts in both attack strength and speed.



## **29. RoBIC: A benchmark suite for assessing classifiers robustness**

cs.CV

4 pages, accepted to ICIP 2021

**SubmitDate**: 2023-03-22    [abs](http://arxiv.org/abs/2102.05368v2) [paper-pdf](http://arxiv.org/pdf/2102.05368v2)

**Authors**: Thibault Maho, Benoît Bonnet, Teddy Furon, Erwan Le Merrer

**Abstract**: Many defenses have emerged with the development of adversarial attacks. Models must be objectively evaluated accordingly. This paper systematically tackles this concern by proposing a new parameter-free benchmark we coin RoBIC. RoBIC fairly evaluates the robustness of image classifiers using a new half-distortion measure. It gauges the robustness of the network against white and black box attacks, independently of its accuracy. RoBIC is faster than the other available benchmarks. We present the significant differences in the robustness of 16 recent models as assessed by RoBIC.



## **30. CgAT: Center-Guided Adversarial Training for Deep Hashing-Based Retrieval**

cs.CV

**SubmitDate**: 2023-03-22    [abs](http://arxiv.org/abs/2204.10779v5) [paper-pdf](http://arxiv.org/pdf/2204.10779v5)

**Authors**: Xunguang Wang, Yiqun Lin, Xiaomeng Li

**Abstract**: Deep hashing has been extensively utilized in massive image retrieval because of its efficiency and effectiveness. However, deep hashing models are vulnerable to adversarial examples, making it essential to develop adversarial defense methods for image retrieval. Existing solutions achieved limited defense performance because of using weak adversarial samples for training and lacking discriminative optimization objectives to learn robust features. In this paper, we present a min-max based Center-guided Adversarial Training, namely CgAT, to improve the robustness of deep hashing networks through worst adversarial examples. Specifically, we first formulate the center code as a semantically-discriminative representative of the input image content, which preserves the semantic similarity with positive samples and dissimilarity with negative examples. We prove that a mathematical formula can calculate the center code immediately. After obtaining the center codes in each optimization iteration of the deep hashing network, they are adopted to guide the adversarial training process. On the one hand, CgAT generates the worst adversarial examples as augmented data by maximizing the Hamming distance between the hash codes of the adversarial examples and the center codes. On the other hand, CgAT learns to mitigate the effects of adversarial samples by minimizing the Hamming distance to the center codes. Extensive experiments on the benchmark datasets demonstrate the effectiveness of our adversarial training algorithm in defending against adversarial attacks for deep hashing-based retrieval. Compared with the current state-of-the-art defense method, we significantly improve the defense performance by an average of 18.61\%, 12.35\%, and 11.56\% on FLICKR-25K, NUS-WIDE, and MS-COCO, respectively. The code is available at https://github.com/xunguangwang/CgAT.



## **31. Membership Inference Attacks against Diffusion Models**

cs.CR

**SubmitDate**: 2023-03-22    [abs](http://arxiv.org/abs/2302.03262v2) [paper-pdf](http://arxiv.org/pdf/2302.03262v2)

**Authors**: Tomoya Matsumoto, Takayuki Miura, Naoto Yanai

**Abstract**: Diffusion models have attracted attention in recent years as innovative generative models. In this paper, we investigate whether a diffusion model is resistant to a membership inference attack, which evaluates the privacy leakage of a machine learning model. We primarily discuss the diffusion model from the standpoints of comparison with a generative adversarial network (GAN) as conventional models and hyperparameters unique to the diffusion model, i.e., time steps, sampling steps, and sampling variances. We conduct extensive experiments with DDIM as a diffusion model and DCGAN as a GAN on the CelebA and CIFAR-10 datasets in both white-box and black-box settings and then confirm if the diffusion model is comparably resistant to a membership inference attack as GAN. Next, we demonstrate that the impact of time steps is significant and intermediate steps in a noise schedule are the most vulnerable to the attack. We also found two key insights through further analysis. First, we identify that DDIM is vulnerable to the attack for small sample sizes instead of achieving a lower FID. Second, sampling steps in hyperparameters are important for resistance to the attack, whereas the impact of sampling variances is quite limited.



## **32. Do Backdoors Assist Membership Inference Attacks?**

cs.CR

**SubmitDate**: 2023-03-22    [abs](http://arxiv.org/abs/2303.12589v1) [paper-pdf](http://arxiv.org/pdf/2303.12589v1)

**Authors**: Yumeki Goto, Nami Ashizawa, Toshiki Shibahara, Naoto Yanai

**Abstract**: When an adversary provides poison samples to a machine learning model, privacy leakage, such as membership inference attacks that infer whether a sample was included in the training of the model, becomes effective by moving the sample to an outlier. However, the attacks can be detected because inference accuracy deteriorates due to poison samples. In this paper, we discuss a \textit{backdoor-assisted membership inference attack}, a novel membership inference attack based on backdoors that return the adversary's expected output for a triggered sample. We found three crucial insights through experiments with an academic benchmark dataset. We first demonstrate that the backdoor-assisted membership inference attack is unsuccessful. Second, when we analyzed loss distributions to understand the reason for the unsuccessful results, we found that backdoors cannot separate loss distributions of training and non-training samples. In other words, backdoors cannot affect the distribution of clean samples. Third, we also show that poison and triggered samples activate neurons of different distributions. Specifically, backdoors make any clean sample an inlier, contrary to poisoning samples. As a result, we confirm that backdoors cannot assist membership inference.



## **33. Autonomous Intelligent Cyber-defense Agent (AICA) Reference Architecture. Release 2.0**

cs.CR

This is a major revision and extension of the earlier release of AICA  Reference Architecture

**SubmitDate**: 2023-03-22    [abs](http://arxiv.org/abs/1803.10664v3) [paper-pdf](http://arxiv.org/pdf/1803.10664v3)

**Authors**: Alexander Kott, Paul Théron, Martin Drašar, Edlira Dushku, Benoît LeBlanc, Paul Losiewicz, Alessandro Guarino, Luigi Mancini, Agostino Panico, Mauno Pihelgas, Krzysztof Rzadca, Fabio De Gaspari

**Abstract**: This report - a major revision of its previous release - describes a reference architecture for intelligent software agents performing active, largely autonomous cyber-defense actions on military networks of computing and communicating devices. The report is produced by the North Atlantic Treaty Organization (NATO) Research Task Group (RTG) IST-152 "Intelligent Autonomous Agents for Cyber Defense and Resilience". In a conflict with a technically sophisticated adversary, NATO military tactical networks will operate in a heavily contested battlefield. Enemy software cyber agents - malware - will infiltrate friendly networks and attack friendly command, control, communications, computers, intelligence, surveillance, and reconnaissance and computerized weapon systems. To fight them, NATO needs artificial cyber hunters - intelligent, autonomous, mobile agents specialized in active cyber defense. With this in mind, in 2016, NATO initiated RTG IST-152. Its objective has been to help accelerate the development and transition to practice of such software agents by producing a reference architecture and technical roadmap. This report presents the concept and architecture of an Autonomous Intelligent Cyber-defense Agent (AICA). We describe the rationale of the AICA concept, explain the methodology and purpose that drive the definition of the AICA Reference Architecture, and review some of the main features and challenges of AICAs.



## **34. Sibling-Attack: Rethinking Transferable Adversarial Attacks against Face Recognition**

cs.CV

8 pages, 5 fivures, accepted by CVPR 2023 as a poster paper

**SubmitDate**: 2023-03-22    [abs](http://arxiv.org/abs/2303.12512v1) [paper-pdf](http://arxiv.org/pdf/2303.12512v1)

**Authors**: Zexin Li, Bangjie Yin, Taiping Yao, Juefeng Guo, Shouhong Ding, Simin Chen, Cong Liu

**Abstract**: A hard challenge in developing practical face recognition (FR) attacks is due to the black-box nature of the target FR model, i.e., inaccessible gradient and parameter information to attackers. While recent research took an important step towards attacking black-box FR models through leveraging transferability, their performance is still limited, especially against online commercial FR systems that can be pessimistic (e.g., a less than 50% ASR--attack success rate on average). Motivated by this, we present Sibling-Attack, a new FR attack technique for the first time explores a novel multi-task perspective (i.e., leveraging extra information from multi-correlated tasks to boost attacking transferability). Intuitively, Sibling-Attack selects a set of tasks correlated with FR and picks the Attribute Recognition (AR) task as the task used in Sibling-Attack based on theoretical and quantitative analysis. Sibling-Attack then develops an optimization framework that fuses adversarial gradient information through (1) constraining the cross-task features to be under the same space, (2) a joint-task meta optimization framework that enhances the gradient compatibility among tasks, and (3) a cross-task gradient stabilization method which mitigates the oscillation effect during attacking. Extensive experiments demonstrate that Sibling-Attack outperforms state-of-the-art FR attack techniques by a non-trivial margin, boosting ASR by 12.61% and 55.77% on average on state-of-the-art pre-trained FR models and two well-known, widely used commercial FR systems.



## **35. Revisiting DeepFool: generalization and improvement**

cs.LG

**SubmitDate**: 2023-03-22    [abs](http://arxiv.org/abs/2303.12481v1) [paper-pdf](http://arxiv.org/pdf/2303.12481v1)

**Authors**: Alireza Abdollahpourrostam, Mahed Abroshan, Seyed-Mohsen Moosavi-Dezfooli

**Abstract**: Deep neural networks have been known to be vulnerable to adversarial examples, which are inputs that are modified slightly to fool the network into making incorrect predictions. This has led to a significant amount of research on evaluating the robustness of these networks against such perturbations. One particularly important robustness metric is the robustness to minimal l2 adversarial perturbations. However, existing methods for evaluating this robustness metric are either computationally expensive or not very accurate. In this paper, we introduce a new family of adversarial attacks that strike a balance between effectiveness and computational efficiency. Our proposed attacks are generalizations of the well-known DeepFool (DF) attack, while they remain simple to understand and implement. We demonstrate that our attacks outperform existing methods in terms of both effectiveness and computational efficiency. Our proposed attacks are also suitable for evaluating the robustness of large models and can be used to perform adversarial training (AT) to achieve state-of-the-art robustness to minimal l2 adversarial perturbations.



## **36. Distribution-restrained Softmax Loss for the Model Robustness**

cs.LG

**SubmitDate**: 2023-03-22    [abs](http://arxiv.org/abs/2303.12363v1) [paper-pdf](http://arxiv.org/pdf/2303.12363v1)

**Authors**: Hao Wang, Chen Li, Jinzhe Jiang, Xin Zhang, Yaqian Zhao, Weifeng Gong

**Abstract**: Recently, the robustness of deep learning models has received widespread attention, and various methods for improving model robustness have been proposed, including adversarial training, model architecture modification, design of loss functions, certified defenses, and so on. However, the principle of the robustness to attacks is still not fully understood, also the related research is still not sufficient. Here, we have identified a significant factor that affects the robustness of models: the distribution characteristics of softmax values for non-real label samples. We found that the results after an attack are highly correlated with the distribution characteristics, and thus we proposed a loss function to suppress the distribution diversity of softmax. A large number of experiments have shown that our method can improve robustness without significant time consumption.



## **37. Wasserstein Adversarial Examples on Univariant Time Series Data**

cs.LG

**SubmitDate**: 2023-03-22    [abs](http://arxiv.org/abs/2303.12357v1) [paper-pdf](http://arxiv.org/pdf/2303.12357v1)

**Authors**: Wenjie Wang, Li Xiong, Jian Lou

**Abstract**: Adversarial examples are crafted by adding indistinguishable perturbations to normal examples in order to fool a well-trained deep learning model to misclassify. In the context of computer vision, this notion of indistinguishability is typically bounded by $L_{\infty}$ or other norms. However, these norms are not appropriate for measuring indistinguishiability for time series data. In this work, we propose adversarial examples in the Wasserstein space for time series data for the first time and utilize Wasserstein distance to bound the perturbation between normal examples and adversarial examples. We introduce Wasserstein projected gradient descent (WPGD), an adversarial attack method for perturbing univariant time series data. We leverage the closed-form solution of Wasserstein distance in the 1D space to calculate the projection step of WPGD efficiently with the gradient descent method. We further propose a two-step projection so that the search of adversarial examples in the Wasserstein space is guided and constrained by Euclidean norms to yield more effective and imperceptible perturbations. We empirically evaluate the proposed attack on several time series datasets in the healthcare domain. Extensive results demonstrate that the Wasserstein attack is powerful and can successfully attack most of the target classifiers with a high attack success rate. To better study the nature of Wasserstein adversarial example, we evaluate a strong defense mechanism named Wasserstein smoothing for potential certified robustness defense. Although the defense can achieve some accuracy gain, it still has limitations in many cases and leaves space for developing a stronger certified robustness method to Wasserstein adversarial examples on univariant time series data.



## **38. Bankrupting Sybil Despite Churn**

cs.CR

41 pages, 6 figures. arXiv admin note: text overlap with  arXiv:2006.02893, arXiv:1911.06462

**SubmitDate**: 2023-03-22    [abs](http://arxiv.org/abs/2010.06834v4) [paper-pdf](http://arxiv.org/pdf/2010.06834v4)

**Authors**: Diksha Gupta, Jared Saia, Maxwell Young

**Abstract**: A Sybil attack occurs when an adversary controls multiple identifiers (IDs) in a system. Limiting the number of Sybil (bad) IDs to a minority is critical to the use of well-established tools for tolerating malicious behavior, such as Byzantine agreement and secure multiparty computation.   A popular technique for enforcing a Sybil minority is resource burning: the verifiable consumption of a network resource, such as computational power, bandwidth, or memory. Unfortunately, typical defenses based on resource burning require non-Sybil (good) IDs to consume at least as many resources as the adversary. Additionally, they have a high resource burning cost, even when the system membership is relatively stable.   Here, we present a new Sybil defense, ERGO, that guarantees (1) there is always a minority of bad IDs; and (2) when the system is under significant attack, the good IDs consume asymptotically less resources than the bad. In particular, for churn rate that can vary exponentially, the resource burning rate for good IDs under ERGO is O(\sqrt{TJ} + J), where T is the resource burning rate of the adversary, and J is the join rate of good IDs. We show this resource burning rate is asymptotically optimal for a large class of algorithms.   We empirically evaluate ERGO alongside prior Sybil defenses. Additionally, we show that ERGO can be combined with machine learning techniques for classifying Sybil IDs, while preserving its theoretical guarantees. Based on our experiments comparing ERGO with two previous Sybil defenses, ERGO improves on the amount of resource burning relative to the adversary by up to 2 orders of magnitude without machine learning, and up to 3 orders of magnitude using machine learning.



## **39. X-CANIDS: Signal-Aware Explainable Intrusion Detection System for Controller Area Network-Based In-Vehicle Network**

cs.CR

**SubmitDate**: 2023-03-22    [abs](http://arxiv.org/abs/2303.12278v1) [paper-pdf](http://arxiv.org/pdf/2303.12278v1)

**Authors**: Seonghoon Jeong, Sangho Lee, Hwejae Lee, Huy Kang Kim

**Abstract**: Controller Area Network (CAN) is an essential networking protocol that connects multiple electronic control units (ECUs) in a vehicle. However, CAN-based in-vehicle networks (IVNs) face security risks owing to the CAN mechanisms. An adversary can sabotage a vehicle by leveraging the security risks if they can access the CAN bus. Thus, recent actions and cybersecurity regulations (e.g., UNR 155) require carmakers to implement intrusion detection systems (IDSs) in their vehicles. An IDS should detect cyberattacks and provide a forensic capability to analyze attacks. Although many IDSs have been proposed, considerations regarding their feasibility and explainability remain lacking. This study proposes X-CANIDS, which is a novel IDS for CAN-based IVNs. X-CANIDS dissects the payloads in CAN messages into human-understandable signals using a CAN database. The signals improve the intrusion detection performance compared with the use of bit representations of raw payloads. These signals also enable an understanding of which signal or ECU is under attack. X-CANIDS can detect zero-day attacks because it does not require any labeled dataset in the training phase. We confirmed the feasibility of the proposed method through a benchmark test on an automotive-grade embedded device with a GPU. The results of this work will be valuable to carmakers and researchers considering the installation of in-vehicle IDSs for their vehicles.



## **40. State-of-the-art optical-based physical adversarial attacks for deep learning computer vision systems**

cs.CV

This work has been submitted to the IEEE for possible publication.  Copyright may be transferred without notice, after which this version may no  longer be accessible

**SubmitDate**: 2023-03-22    [abs](http://arxiv.org/abs/2303.12249v1) [paper-pdf](http://arxiv.org/pdf/2303.12249v1)

**Authors**: Junbin Fang, You Jiang, Canjian Jiang, Zoe L. Jiang, Siu-Ming Yiu, Chuanyi Liu

**Abstract**: Adversarial attacks can mislead deep learning models to make false predictions by implanting small perturbations to the original input that are imperceptible to the human eye, which poses a huge security threat to the computer vision systems based on deep learning. Physical adversarial attacks, which is more realistic, as the perturbation is introduced to the input before it is being captured and converted to a binary image inside the vision system, when compared to digital adversarial attacks. In this paper, we focus on physical adversarial attacks and further classify them into invasive and non-invasive. Optical-based physical adversarial attack techniques (e.g. using light irradiation) belong to the non-invasive category. As the perturbations can be easily ignored by humans as the perturbations are very similar to the effects generated by a natural environment in the real world. They are highly invisibility and executable and can pose a significant or even lethal threats to real systems. This paper focuses on optical-based physical adversarial attack techniques for computer vision systems, with emphasis on the introduction and discussion of optical-based physical adversarial attack techniques.



## **41. Task-Oriented Communications for NextG: End-to-End Deep Learning and AI Security Aspects**

cs.NI

**SubmitDate**: 2023-03-21    [abs](http://arxiv.org/abs/2212.09668v2) [paper-pdf](http://arxiv.org/pdf/2212.09668v2)

**Authors**: Yalin E. Sagduyu, Sennur Ulukus, Aylin Yener

**Abstract**: Communications systems to date are primarily designed with the goal of reliable transfer of digital sequences (bits). Next generation (NextG) communication systems are beginning to explore shifting this design paradigm to reliably executing a given task such as in task-oriented communications. In this paper, wireless signal classification is considered as the task for the NextG Radio Access Network (RAN), where edge devices collect wireless signals for spectrum awareness and communicate with the NextG base station (gNodeB) that needs to identify the signal label. Edge devices may not have sufficient processing power and may not be trusted to perform the signal classification task, whereas the transfer of signals to the gNodeB may not be feasible due to stringent delay, rate, and energy restrictions. Task-oriented communications is considered by jointly training the transmitter, receiver and classifier functionalities as an encoder-decoder pair for the edge device and the gNodeB. This approach improves the accuracy compared to the separated case of signal transfer followed by classification. Adversarial machine learning poses a major security threat to the use of deep learning for task-oriented communications. A major performance loss is shown when backdoor (Trojan) and adversarial (evasion) attacks target the training and test processes of task-oriented communications.



## **42. Towards Compositional Adversarial Robustness: Generalizing Adversarial Training to Composite Semantic Perturbations**

cs.CV

CVPR 2023. The research demo is at https://hsiung.cc/CARBEN/

**SubmitDate**: 2023-03-21    [abs](http://arxiv.org/abs/2202.04235v3) [paper-pdf](http://arxiv.org/pdf/2202.04235v3)

**Authors**: Lei Hsiung, Yun-Yun Tsai, Pin-Yu Chen, Tsung-Yi Ho

**Abstract**: Model robustness against adversarial examples of single perturbation type such as the $\ell_{p}$-norm has been widely studied, yet its generalization to more realistic scenarios involving multiple semantic perturbations and their composition remains largely unexplored. In this paper, we first propose a novel method for generating composite adversarial examples. Our method can find the optimal attack composition by utilizing component-wise projected gradient descent and automatic attack-order scheduling. We then propose generalized adversarial training (GAT) to extend model robustness from $\ell_{p}$-ball to composite semantic perturbations, such as the combination of Hue, Saturation, Brightness, Contrast, and Rotation. Results obtained using ImageNet and CIFAR-10 datasets indicate that GAT can be robust not only to all the tested types of a single attack, but also to any combination of such attacks. GAT also outperforms baseline $\ell_{\infty}$-norm bounded adversarial training approaches by a significant margin.



## **43. Efficient Decision-based Black-box Patch Attacks on Video Recognition**

cs.CV

**SubmitDate**: 2023-03-21    [abs](http://arxiv.org/abs/2303.11917v1) [paper-pdf](http://arxiv.org/pdf/2303.11917v1)

**Authors**: Kaixun Jiang, Zhaoyu Chen, Tony Huang, Jiafeng Wang, Dingkang Yang, Bo Li, Yan Wang, Wenqiang Zhang

**Abstract**: Although Deep Neural Networks (DNNs) have demonstrated excellent performance, they are vulnerable to adversarial patches that introduce perceptible and localized perturbations to the input. Generating adversarial patches on images has received much attention, while adversarial patches on videos have not been well investigated. Further, decision-based attacks, where attackers only access the predicted hard labels by querying threat models, have not been well explored on video models either, even if they are practical in real-world video recognition scenes. The absence of such studies leads to a huge gap in the robustness assessment for video models. To bridge this gap, this work first explores decision-based patch attacks on video models. We analyze that the huge parameter space brought by videos and the minimal information returned by decision-based models both greatly increase the attack difficulty and query burden. To achieve a query-efficient attack, we propose a spatial-temporal differential evolution (STDE) framework. First, STDE introduces target videos as patch textures and only adds patches on keyframes that are adaptively selected by temporal difference. Second, STDE takes minimizing the patch area as the optimization objective and adopts spatialtemporal mutation and crossover to search for the global optimum without falling into the local optimum. Experiments show STDE has demonstrated state-of-the-art performance in terms of threat, efficiency and imperceptibility. Hence, STDE has the potential to be a powerful tool for evaluating the robustness of video recognition models.



## **44. The Threat of Adversarial Attacks on Machine Learning in Network Security -- A Survey**

cs.CR

**SubmitDate**: 2023-03-21    [abs](http://arxiv.org/abs/1911.02621v3) [paper-pdf](http://arxiv.org/pdf/1911.02621v3)

**Authors**: Olakunle Ibitoye, Rana Abou-Khamis, Mohamed el Shehaby, Ashraf Matrawy, M. Omair Shafiq

**Abstract**: Machine learning models have made many decision support systems to be faster, more accurate, and more efficient. However, applications of machine learning in network security face a more disproportionate threat of active adversarial attacks compared to other domains. This is because machine learning applications in network security such as malware detection, intrusion detection, and spam filtering are by themselves adversarial in nature. In what could be considered an arm's race between attackers and defenders, adversaries constantly probe machine learning systems with inputs that are explicitly designed to bypass the system and induce a wrong prediction. In this survey, we first provide a taxonomy of machine learning techniques, tasks, and depth. We then introduce a classification of machine learning in network security applications. Next, we examine various adversarial attacks against machine learning in network security and introduce two classification approaches for adversarial attacks in network security. First, we classify adversarial attacks in network security based on a taxonomy of network security applications. Secondly, we categorize adversarial attacks in network security into a problem space vs feature space dimensional classification model. We then analyze the various defenses against adversarial attacks on machine learning-based network security applications. We conclude by introducing an adversarial risk grid map and evaluating several existing adversarial attacks against machine learning in network security using the risk grid map. We also identify where each attack classification resides within the adversarial risk grid map.



## **45. OTJR: Optimal Transport Meets Optimal Jacobian Regularization for Adversarial Robustness**

cs.CV

**SubmitDate**: 2023-03-21    [abs](http://arxiv.org/abs/2303.11793v1) [paper-pdf](http://arxiv.org/pdf/2303.11793v1)

**Authors**: Binh M. Le, Shahroz Tariq, Simon S. Woo

**Abstract**: Deep neural networks are widely recognized as being vulnerable to adversarial perturbation. To overcome this challenge, developing a robust classifier is crucial. So far, two well-known defenses have been adopted to improve the learning of robust classifiers, namely adversarial training (AT) and Jacobian regularization. However, each approach behaves differently against adversarial perturbations. First, our work carefully analyzes and characterizes these two schools of approaches, both theoretically and empirically, to demonstrate how each approach impacts the robust learning of a classifier. Next, we propose our novel Optimal Transport with Jacobian regularization method, dubbed OTJR, jointly incorporating the input-output Jacobian regularization into the AT by leveraging the optimal transport theory. In particular, we employ the Sliced Wasserstein (SW) distance that can efficiently push the adversarial samples' representations closer to those of clean samples, regardless of the number of classes within the dataset. The SW distance provides the adversarial samples' movement directions, which are much more informative and powerful for the Jacobian regularization. Our extensive experiments demonstrate the effectiveness of our proposed method, which jointly incorporates Jacobian regularization into AT. Furthermore, we demonstrate that our proposed method consistently enhances the model's robustness with CIFAR-100 dataset under various adversarial attack settings, achieving up to 28.49% under AutoAttack.



## **46. Generative AI for Cyber Threat-Hunting in 6G-enabled IoT Networks**

cs.CR

The paper is accepted and will be published in the IEEE/ACM CCGrid  2023 Conference Proceedings

**SubmitDate**: 2023-03-21    [abs](http://arxiv.org/abs/2303.11751v1) [paper-pdf](http://arxiv.org/pdf/2303.11751v1)

**Authors**: Mohamed Amine Ferrag, Merouane Debbah, Muna Al-Hawawreh

**Abstract**: The next generation of cellular technology, 6G, is being developed to enable a wide range of new applications and services for the Internet of Things (IoT). One of 6G's main advantages for IoT applications is its ability to support much higher data rates and bandwidth as well as to support ultra-low latency. However, with this increased connectivity will come to an increased risk of cyber threats, as attackers will be able to exploit the large network of connected devices. Generative Artificial Intelligence (AI) can be used to detect and prevent cyber attacks by continuously learning and adapting to new threats and vulnerabilities. In this paper, we discuss the use of generative AI for cyber threat-hunting (CTH) in 6G-enabled IoT networks. Then, we propose a new generative adversarial network (GAN) and Transformer-based model for CTH in 6G-enabled IoT Networks. The experimental analysis results with a new cyber security dataset demonstrate that the Transformer-based security model for CTH can detect IoT attacks with a high overall accuracy of 95%. We examine the challenges and opportunities and conclude by highlighting the potential of generative AI in enhancing the security of 6G-enabled IoT networks and call for further research to be conducted in this area.



## **47. Poisoning Attacks in Federated Edge Learning for Digital Twin 6G-enabled IoTs: An Anticipatory Study**

cs.CR

The paper is accepted and will be published in the IEEE ICC 2023  Conference Proceedings

**SubmitDate**: 2023-03-21    [abs](http://arxiv.org/abs/2303.11745v1) [paper-pdf](http://arxiv.org/pdf/2303.11745v1)

**Authors**: Mohamed Amine Ferrag, Burak Kantarci, Lucas C. Cordeiro, Merouane Debbah, Kim-Kwang Raymond Choo

**Abstract**: Federated edge learning can be essential in supporting privacy-preserving, artificial intelligence (AI)-enabled activities in digital twin 6G-enabled Internet of Things (IoT) environments. However, we need to also consider the potential of attacks targeting the underlying AI systems (e.g., adversaries seek to corrupt data on the IoT devices during local updates or corrupt the model updates); hence, in this article, we propose an anticipatory study for poisoning attacks in federated edge learning for digital twin 6G-enabled IoT environments. Specifically, we study the influence of adversaries on the training and development of federated learning models in digital twin 6G-enabled IoT environments. We demonstrate that attackers can carry out poisoning attacks in two different learning settings, namely: centralized learning and federated learning, and successful attacks can severely reduce the model's accuracy. We comprehensively evaluate the attacks on a new cyber security dataset designed for IoT applications with three deep neural networks under the non-independent and identically distributed (Non-IID) data and the independent and identically distributed (IID) data. The poisoning attacks, on an attack classification problem, can lead to a decrease in accuracy from 94.93% to 85.98% with IID data and from 94.18% to 30.04% with Non-IID.



## **48. Manipulating Transfer Learning for Property Inference**

cs.LG

Accepted to CVPR 2023

**SubmitDate**: 2023-03-21    [abs](http://arxiv.org/abs/2303.11643v1) [paper-pdf](http://arxiv.org/pdf/2303.11643v1)

**Authors**: Yulong Tian, Fnu Suya, Anshuman Suri, Fengyuan Xu, David Evans

**Abstract**: Transfer learning is a popular method for tuning pretrained (upstream) models for different downstream tasks using limited data and computational resources. We study how an adversary with control over an upstream model used in transfer learning can conduct property inference attacks on a victim's tuned downstream model. For example, to infer the presence of images of a specific individual in the downstream training set. We demonstrate attacks in which an adversary can manipulate the upstream model to conduct highly effective and specific property inference attacks (AUC score $> 0.9$), without incurring significant performance loss on the main task. The main idea of the manipulation is to make the upstream model generate activations (intermediate features) with different distributions for samples with and without a target property, thus enabling the adversary to distinguish easily between downstream models trained with and without training examples that have the target property. Our code is available at https://github.com/yulongt23/Transfer-Inference.



## **49. An Observer-based Switching Algorithm for Safety under Sensor Denial-of-Service Attacks**

eess.SY

Accepted at the 2023 American Control Conference (ACC)

**SubmitDate**: 2023-03-21    [abs](http://arxiv.org/abs/2303.11640v1) [paper-pdf](http://arxiv.org/pdf/2303.11640v1)

**Authors**: Santiago Jimenez Leudo, Kunal Garg, Ricardo G. Sanfelice, Alvaro A. Cardenas

**Abstract**: The design of safe-critical control algorithms for systems under Denial-of-Service (DoS) attacks on the system output is studied in this work. We aim to address scenarios where attack-mitigation approaches are not feasible, and the system needs to maintain safety under adversarial attacks. We propose an attack-recovery strategy by designing a switching observer and characterizing bounds in the error of a state estimation scheme by specifying tolerable limits on the time length of attacks. Then, we propose a switching control algorithm that renders forward invariant a set for the observer. Thus, by satisfying the error bounds of the state estimation, we guarantee that the safe set is rendered conditionally invariant with respect to a set of initial conditions. A numerical example illustrates the efficacy of the approach.



## **50. Enhancing the Self-Universality for Transferable Targeted Attacks**

cs.CV

**SubmitDate**: 2023-03-21    [abs](http://arxiv.org/abs/2209.03716v2) [paper-pdf](http://arxiv.org/pdf/2209.03716v2)

**Authors**: Zhipeng Wei, Jingjing Chen, Zuxuan Wu, Yu-Gang Jiang

**Abstract**: In this paper, we propose a novel transfer-based targeted attack method that optimizes the adversarial perturbations without any extra training efforts for auxiliary networks on training data. Our new attack method is proposed based on the observation that highly universal adversarial perturbations tend to be more transferable for targeted attacks. Therefore, we propose to make the perturbation to be agnostic to different local regions within one image, which we called as self-universality. Instead of optimizing the perturbations on different images, optimizing on different regions to achieve self-universality can get rid of using extra data. Specifically, we introduce a feature similarity loss that encourages the learned perturbations to be universal by maximizing the feature similarity between adversarial perturbed global images and randomly cropped local regions. With the feature similarity loss, our method makes the features from adversarial perturbations to be more dominant than that of benign images, hence improving targeted transferability. We name the proposed attack method as Self-Universality (SU) attack. Extensive experiments demonstrate that SU can achieve high success rates for transfer-based targeted attacks. On ImageNet-compatible dataset, SU yields an improvement of 12\% compared with existing state-of-the-art methods. Code is available at https://github.com/zhipeng-wei/Self-Universality.



