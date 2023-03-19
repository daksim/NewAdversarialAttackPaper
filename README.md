# Latest Adversarial Attack Papers
**update at 2023-03-19 16:27:03**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. Among Us: Adversarially Robust Collaborative Perception by Consensus**

cs.RO

**SubmitDate**: 2023-03-16    [abs](http://arxiv.org/abs/2303.09495v1) [paper-pdf](http://arxiv.org/pdf/2303.09495v1)

**Authors**: Yiming Li, Qi Fang, Jiamu Bai, Siheng Chen, Felix Juefei-Xu, Chen Feng

**Abstract**: Multiple robots could perceive a scene (e.g., detect objects) collaboratively better than individuals, although easily suffer from adversarial attacks when using deep learning. This could be addressed by the adversarial defense, but its training requires the often-unknown attacking mechanism. Differently, we propose ROBOSAC, a novel sampling-based defense strategy generalizable to unseen attackers. Our key idea is that collaborative perception should lead to consensus rather than dissensus in results compared to individual perception. This leads to our hypothesize-and-verify framework: perception results with and without collaboration from a random subset of teammates are compared until reaching a consensus. In such a framework, more teammates in the sampled subset often entail better perception performance but require longer sampling time to reject potential attackers. Thus, we derive how many sampling trials are needed to ensure the desired size of an attacker-free subset, or equivalently, the maximum size of such a subset that we can successfully sample within a given number of trials. We validate our method on the task of collaborative 3D object detection in autonomous driving scenarios.



## **2. Rickrolling the Artist: Injecting Backdoors into Text Encoders for Text-to-Image Synthesis**

cs.LG

30 pages, 20 figures, 5 tables

**SubmitDate**: 2023-03-16    [abs](http://arxiv.org/abs/2211.02408v2) [paper-pdf](http://arxiv.org/pdf/2211.02408v2)

**Authors**: Lukas Struppek, Dominik Hintersdorf, Kristian Kersting

**Abstract**: While text-to-image synthesis currently enjoys great popularity among researchers and the general public, the security of these models has been neglected so far. Many text-guided image generation models rely on pre-trained text encoders from external sources, and their users trust that the retrieved models will behave as promised. Unfortunately, this might not be the case. We introduce backdoor attacks against text-guided generative models and demonstrate that their text encoders pose a major tampering risk. Our attacks only slightly alter an encoder so that no suspicious model behavior is apparent for image generations with clean prompts. By then inserting a single character trigger into the prompt, e.g., a non-Latin character or emoji, the adversary can trigger the model to either generate images with pre-defined attributes or images following a hidden, potentially malicious description. We empirically demonstrate the high effectiveness of our attacks on Stable Diffusion and highlight that the injection process of a single backdoor takes less than two minutes. Besides phrasing our approach solely as an attack, it can also force an encoder to forget phrases related to certain concepts, such as nudity or violence, and help to make image generation safer.



## **3. Image Classifiers Leak Sensitive Attributes About Their Classes**

cs.LG

40 pages, 32 figures, 4 tables

**SubmitDate**: 2023-03-16    [abs](http://arxiv.org/abs/2303.09289v1) [paper-pdf](http://arxiv.org/pdf/2303.09289v1)

**Authors**: Lukas Struppek, Dominik Hintersdorf, Felix Friedrich, Manuel Brack, Patrick Schramowski, Kristian Kersting

**Abstract**: Neural network-based image classifiers are powerful tools for computer vision tasks, but they inadvertently reveal sensitive attribute information about their classes, raising concerns about their privacy. To investigate this privacy leakage, we introduce the first Class Attribute Inference Attack (Caia), which leverages recent advances in text-to-image synthesis to infer sensitive attributes of individual classes in a black-box setting, while remaining competitive with related white-box attacks. Our extensive experiments in the face recognition domain show that Caia can accurately infer undisclosed sensitive attributes, such as an individual's hair color, gender and racial appearance, which are not part of the training labels. Interestingly, we demonstrate that adversarial robust models are even more vulnerable to such privacy leakage than standard models, indicating that a trade-off between robustness and privacy exists.



## **4. QuickSync: A Quickly Synchronizing PoS-Based Blockchain Protocol**

cs.CR

**SubmitDate**: 2023-03-16    [abs](http://arxiv.org/abs/2005.03564v4) [paper-pdf](http://arxiv.org/pdf/2005.03564v4)

**Authors**: Shoeb Siddiqui, Varul Srivastava, Raj Maheshwari, Sujit Gujar

**Abstract**: To implement a blockchain, we need a blockchain protocol for all the nodes to follow. To design a blockchain protocol, we need a block publisher selection mechanism and a chain selection rule. In Proof-of-Stake (PoS) based blockchain protocols, block publisher selection mechanism selects the node to publish the next block based on the relative stake held by the node. However, PoS protocols, such as Ouroboros v1, may face vulnerability to fully adaptive corruptions.   In this paper, we propose a novel PoS-based blockchain protocol, QuickSync, to achieve security against fully adaptive corruptions while improving on performance. We propose a metric called block power, a value defined for each block, derived from the output of the verifiable random function based on the digital signature of the block publisher. With this metric, we compute chain power, the sum of block powers of all the blocks comprising the chain, for all the valid chains. These metrics are a function of the block publisher's stake to enable the PoS aspect of the protocol. The chain selection rule selects the chain with the highest chain power as the one to extend. This chain selection rule hence determines the selected block publisher of the previous block. When we use metrics to define the chain selection rule, it may lead to vulnerabilities against Sybil attacks. QuickSync uses a Sybil attack resistant function implemented using histogram matching. We prove that QuickSync satisfies common prefix, chain growth, and chain quality properties and hence it is secure. We also show that it is resilient to different types of adversarial attack strategies. Our analysis demonstrates that QuickSync performs better than Bitcoin by an order of magnitude on both transactions per second and time to finality, and better than Ouroboros v1 by a factor of three on time to finality.



## **5. Security of Blockchains at Capacity**

cs.CR

**SubmitDate**: 2023-03-16    [abs](http://arxiv.org/abs/2303.09113v1) [paper-pdf](http://arxiv.org/pdf/2303.09113v1)

**Authors**: Lucianna Kiffer, Joachim Neu, Srivatsan Sridhar, Aviv Zohar, David Tse

**Abstract**: Given a network of nodes with certain communication and computation capacities, what is the maximum rate at which a blockchain can run securely? We study this question for proof-of-work (PoW) and proof-of-stake (PoS) longest chain protocols under a 'bounded bandwidth' model which captures queuing and processing delays due to high block rate relative to capacity, bursty release of adversarial blocks, and in PoS, spamming due to equivocations.   We demonstrate that security of both PoW and PoS longest chain, when operating at capacity, requires carefully designed scheduling policies that correctly prioritize which blocks are processed first, as we show attack strategies tailored to such policies. In PoS, we show an attack exploiting equivocations, which highlights that the throughput of the PoS longest chain protocol with a broad class of scheduling policies must decrease as the desired security error probability decreases. At the same time, through an improved analysis method, our work is the first to identify block production rates under which PoW longest chain is secure in the bounded bandwidth setting. We also present the first PoS longest chain protocol, SaPoS, which is secure with a block production rate independent of the security error probability, by using an 'equivocation removal' policy to prevent equivocation spamming.



## **6. Rethinking Model Ensemble in Transfer-based Adversarial Attacks**

cs.CV

**SubmitDate**: 2023-03-16    [abs](http://arxiv.org/abs/2303.09105v1) [paper-pdf](http://arxiv.org/pdf/2303.09105v1)

**Authors**: Huanran Chen, Yichi Zhang, Yinpeng Dong, Jun Zhu

**Abstract**: Deep learning models are vulnerable to adversarial examples. Transfer-based adversarial attacks attract tremendous attention as they can identify the weaknesses of deep learning models in a black-box manner. An effective strategy to improve the transferability of adversarial examples is attacking an ensemble of models. However, previous works simply average the outputs of different models, lacking an in-depth analysis on how and why model ensemble can strongly improve the transferability. In this work, we rethink the ensemble in adversarial attacks and define the common weakness of model ensemble with the properties of the flatness of loss landscape and the closeness to the local optimum of each model. We empirically and theoretically show that these two properties are strongly correlated with the transferability and propose a Common Weakness Attack (CWA) to generate more transferable adversarial examples by promoting these two properties. Experimental results on both image classification and object detection tasks validate the effectiveness of our approach to improve the adversarial transferability, especially when attacking adversarially trained models.



## **7. Well-classified Examples are Underestimated in Classification with Deep Neural Networks**

cs.LG

Accepted by AAAI 2022; 18 pages, 11 figures, 13 tables

**SubmitDate**: 2023-03-16    [abs](http://arxiv.org/abs/2110.06537v6) [paper-pdf](http://arxiv.org/pdf/2110.06537v6)

**Authors**: Guangxiang Zhao, Wenkai Yang, Xuancheng Ren, Lei Li, Yunfang Wu, Xu Sun

**Abstract**: The conventional wisdom behind learning deep classification models is to focus on bad-classified examples and ignore well-classified examples that are far from the decision boundary. For instance, when training with cross-entropy loss, examples with higher likelihoods (i.e., well-classified examples) contribute smaller gradients in back-propagation. However, we theoretically show that this common practice hinders representation learning, energy optimization, and margin growth. To counteract this deficiency, we propose to reward well-classified examples with additive bonuses to revive their contribution to the learning process. This counterexample theoretically addresses these three issues. We empirically support this claim by directly verifying the theoretical results or significant performance improvement with our counterexample on diverse tasks, including image classification, graph classification, and machine translation. Furthermore, this paper shows that we can deal with complex scenarios, such as imbalanced classification, OOD detection, and applications under adversarial attacks because our idea can solve these three issues. Code is available at: https://github.com/lancopku/well-classified-examples-are-underestimated.



## **8. Robust Evaluation of Diffusion-Based Adversarial Purification**

cs.CV

**SubmitDate**: 2023-03-16    [abs](http://arxiv.org/abs/2303.09051v1) [paper-pdf](http://arxiv.org/pdf/2303.09051v1)

**Authors**: Minjong Lee, Dongwoo Kim

**Abstract**: We question the current evaluation practice on diffusion-based purification methods. Diffusion-based purification methods aim to remove adversarial effects from an input data point at test time. The approach gains increasing attention as an alternative to adversarial training due to the disentangling between training and testing. Well-known white-box attacks are often employed to measure the robustness of the purification. However, it is unknown whether these attacks are the most effective for the diffusion-based purification since the attacks are often tailored for adversarial training. We analyze the current practices and provide a new guideline for measuring the robustness of purification methods against adversarial attacks. Based on our analysis, we further propose a new purification strategy showing competitive results against the state-of-the-art adversarial training approaches.



## **9. DeeBBAA: A benchmark Deep Black Box Adversarial Attack against Cyber-Physical Power Systems**

cs.CR

**SubmitDate**: 2023-03-16    [abs](http://arxiv.org/abs/2303.09024v1) [paper-pdf](http://arxiv.org/pdf/2303.09024v1)

**Authors**: Arnab Bhattacharjee, Tapan K. Saha, Ashu Verma, Sukumar Mishra

**Abstract**: An increased energy demand, and environmental pressure to accommodate higher levels of renewable energy and flexible loads like electric vehicles have led to numerous smart transformations in the modern power systems. These transformations make the cyber-physical power system highly susceptible to cyber-adversaries targeting its numerous operations. In this work, a novel black box adversarial attack strategy is proposed targeting the AC state estimation operation of an unknown power system using historical data. Specifically, false data is injected into the measurements obtained from a small subset of the power system components which leads to significant deviations in the state estimates. Experiments carried out on the IEEE 39 bus and 118 bus test systems make it evident that the proposed strategy, called DeeBBAA, can evade numerous conventional and state-of-the-art attack detection mechanisms with very high probability.



## **10. Identifying Threats, Cybercrime and Digital Forensic Opportunities in Smart City Infrastructure via Threat Modeling**

cs.CR

Updated to include amendments from peer review process. Accepted in  Forensic Science International: Digital Investigation

**SubmitDate**: 2023-03-16    [abs](http://arxiv.org/abs/2210.14692v2) [paper-pdf](http://arxiv.org/pdf/2210.14692v2)

**Authors**: Yee Ching Tok, Sudipta Chattopadhyay

**Abstract**: Technological advances have enabled multiple countries to consider implementing Smart City Infrastructure to provide in-depth insights into different data points and enhance the lives of citizens. Unfortunately, these new technological implementations also entice adversaries and cybercriminals to execute cyber-attacks and commit criminal acts on these modern infrastructures. Given the borderless nature of cyber attacks, varying levels of understanding of smart city infrastructure and ongoing investigation workloads, law enforcement agencies and investigators would be hard-pressed to respond to these kinds of cybercrime. Without an investigative capability by investigators, these smart infrastructures could become new targets favored by cybercriminals.   To address the challenges faced by investigators, we propose a common definition of smart city infrastructure. Based on the definition, we utilize the STRIDE threat modeling methodology and the Microsoft Threat Modeling Tool to identify threats present in the infrastructure and create a threat model which can be further customized or extended by interested parties. Next, we map offences, possible evidence sources and types of threats identified to help investigators understand what crimes could have been committed and what evidence would be required in their investigation work. Finally, noting that Smart City Infrastructure investigations would be a global multi-faceted challenge, we discuss technical and legal opportunities in digital forensics on Smart City Infrastructure.



## **11. Certifiable (Multi)Robustness Against Patch Attacks Using ERM**

cs.LG

**SubmitDate**: 2023-03-15    [abs](http://arxiv.org/abs/2303.08944v1) [paper-pdf](http://arxiv.org/pdf/2303.08944v1)

**Authors**: Saba Ahmadi, Avrim Blum, Omar Montasser, Kevin Stangl

**Abstract**: Consider patch attacks, where at test-time an adversary manipulates a test image with a patch in order to induce a targeted misclassification. We consider a recent defense to patch attacks, Patch-Cleanser (Xiang et al. [2022]). The Patch-Cleanser algorithm requires a prediction model to have a ``two-mask correctness'' property, meaning that the prediction model should correctly classify any image when any two blank masks replace portions of the image. Xiang et al. learn a prediction model to be robust to two-mask operations by augmenting the training set with pairs of masks at random locations of training images and performing empirical risk minimization (ERM) on the augmented dataset.   However, in the non-realizable setting when no predictor is perfectly correct on all two-mask operations on all images, we exhibit an example where ERM fails. To overcome this challenge, we propose a different algorithm that provably learns a predictor robust to all two-mask operations using an ERM oracle, based on prior work by Feige et al. [2015]. We also extend this result to a multiple-group setting, where we can learn a predictor that achieves low robust loss on all groups simultaneously.



## **12. Learning When to Use Adaptive Adversarial Image Perturbations against Autonomous Vehicles**

cs.RO

**SubmitDate**: 2023-03-15    [abs](http://arxiv.org/abs/2212.13667v2) [paper-pdf](http://arxiv.org/pdf/2212.13667v2)

**Authors**: Hyung-Jin Yoon, Hamidreza Jafarnejadsani, Petros Voulgaris

**Abstract**: The deep neural network (DNN) models for object detection using camera images are widely adopted in autonomous vehicles. However, DNN models are shown to be susceptible to adversarial image perturbations. In the existing methods of generating the adversarial image perturbations, optimizations take each incoming image frame as the decision variable to generate an image perturbation. Therefore, given a new image, the typically computationally-expensive optimization needs to start over as there is no learning between the independent optimizations. Very few approaches have been developed for attacking online image streams while considering the underlying physical dynamics of autonomous vehicles, their mission, and the environment. We propose a multi-level stochastic optimization framework that monitors an attacker's capability of generating the adversarial perturbations. Based on this capability level, a binary decision attack/not attack is introduced to enhance the effectiveness of the attacker. We evaluate our proposed multi-level image attack framework using simulations for vision-guided autonomous vehicles and actual tests with a small indoor drone in an office environment. The results show our method's capability to generate the image attack in real-time while monitoring when the attacker is proficient given state estimates.



## **13. Visual Prompting for Adversarial Robustness**

cs.CV

ICASSP 2023

**SubmitDate**: 2023-03-15    [abs](http://arxiv.org/abs/2210.06284v2) [paper-pdf](http://arxiv.org/pdf/2210.06284v2)

**Authors**: Aochuan Chen, Peter Lorenz, Yuguang Yao, Pin-Yu Chen, Sijia Liu

**Abstract**: In this work, we leverage visual prompting (VP) to improve adversarial robustness of a fixed, pre-trained model at testing time. Compared to conventional adversarial defenses, VP allows us to design universal (i.e., data-agnostic) input prompting templates, which have plug-and-play capabilities at testing time to achieve desired model performance without introducing much computation overhead. Although VP has been successfully applied to improving model generalization, it remains elusive whether and how it can be used to defend against adversarial attacks. We investigate this problem and show that the vanilla VP approach is not effective in adversarial defense since a universal input prompt lacks the capacity for robust learning against sample-specific adversarial perturbations. To circumvent it, we propose a new VP method, termed Class-wise Adversarial Visual Prompting (C-AVP), to generate class-wise visual prompts so as to not only leverage the strengths of ensemble prompts but also optimize their interrelations to improve model robustness. Our experiments show that C-AVP outperforms the conventional VP method, with 2.1X standard accuracy gain and 2X robust accuracy gain. Compared to classical test-time defenses, C-AVP also yields a 42X inference time speedup.



## **14. EvalAttAI: A Holistic Approach to Evaluating Attribution Maps in Robust and Non-Robust Models**

cs.LG

**SubmitDate**: 2023-03-15    [abs](http://arxiv.org/abs/2303.08866v1) [paper-pdf](http://arxiv.org/pdf/2303.08866v1)

**Authors**: Ian E. Nielsen, Ravi P. Ramachandran, Nidhal Bouaynaya, Hassan M. Fathallah-Shaykh, Ghulam Rasool

**Abstract**: The expansion of explainable artificial intelligence as a field of research has generated numerous methods of visualizing and understanding the black box of a machine learning model. Attribution maps are generally used to highlight the parts of the input image that influence the model to make a specific decision. On the other hand, the robustness of machine learning models to natural noise and adversarial attacks is also being actively explored. This paper focuses on evaluating methods of attribution mapping to find whether robust neural networks are more explainable. We explore this problem within the application of classification for medical imaging. Explainability research is at an impasse. There are many methods of attribution mapping, but no current consensus on how to evaluate them and determine the ones that are the best. Our experiments on multiple datasets (natural and medical imaging) and various attribution methods reveal that two popular evaluation metrics, Deletion and Insertion, have inherent limitations and yield contradictory results. We propose a new explainability faithfulness metric (called EvalAttAI) that addresses the limitations of prior metrics. Using our novel evaluation, we found that Bayesian deep neural networks using the Variational Density Propagation technique were consistently more explainable when used with the best performing attribution method, the Vanilla Gradient. However, in general, various types of robust neural networks may not be more explainable, despite these models producing more visually plausible attribution maps.



## **15. Enhancing Diffusion-Based Image Synthesis with Robust Classifier Guidance**

cs.CV

Accepted to TMLR

**SubmitDate**: 2023-03-15    [abs](http://arxiv.org/abs/2208.08664v2) [paper-pdf](http://arxiv.org/pdf/2208.08664v2)

**Authors**: Bahjat Kawar, Roy Ganz, Michael Elad

**Abstract**: Denoising diffusion probabilistic models (DDPMs) are a recent family of generative models that achieve state-of-the-art results. In order to obtain class-conditional generation, it was suggested to guide the diffusion process by gradients from a time-dependent classifier. While the idea is theoretically sound, deep learning-based classifiers are infamously susceptible to gradient-based adversarial attacks. Therefore, while traditional classifiers may achieve good accuracy scores, their gradients are possibly unreliable and might hinder the improvement of the generation results. Recent work discovered that adversarially robust classifiers exhibit gradients that are aligned with human perception, and these could better guide a generative process towards semantically meaningful images. We utilize this observation by defining and training a time-dependent adversarially robust classifier and use it as guidance for a generative diffusion model. In experiments on the highly challenging and diverse ImageNet dataset, our scheme introduces significantly more intelligible intermediate gradients, better alignment with theoretical findings, as well as improved generation results under several evaluation metrics. Furthermore, we conduct an opinion survey whose findings indicate that human raters prefer our method's results.



## **16. Black-box Adversarial Example Attack towards FCG Based Android Malware Detection under Incomplete Feature Information**

cs.SE

**SubmitDate**: 2023-03-15    [abs](http://arxiv.org/abs/2303.08509v1) [paper-pdf](http://arxiv.org/pdf/2303.08509v1)

**Authors**: Heng Li, Zhang Cheng, Bang Wu, Liheng Yuan, Cuiying Gao, Wei Yuan, Xiapu Luo

**Abstract**: The function call graph (FCG) based Android malware detection methods have recently attracted increasing attention due to their promising performance. However, these methods are susceptible to adversarial examples (AEs). In this paper, we design a novel black-box AE attack towards the FCG based malware detection system, called BagAmmo. To mislead its target system, BagAmmo purposefully perturbs the FCG feature of malware through inserting "never-executed" function calls into malware code. The main challenges are two-fold. First, the malware functionality should not be changed by adversarial perturbation. Second, the information of the target system (e.g., the graph feature granularity and the output probabilities) is absent.   To preserve malware functionality, BagAmmo employs the try-catch trap to insert function calls to perturb the FCG of malware. Without the knowledge about feature granularity and output probabilities, BagAmmo adopts the architecture of generative adversarial network (GAN), and leverages a multi-population co-evolution algorithm (i.e., Apoem) to generate the desired perturbation. Every population in Apoem represents a possible feature granularity, and the real feature granularity can be achieved when Apoem converges.   Through extensive experiments on over 44k Android apps and 32 target models, we evaluate the effectiveness, efficiency and resilience of BagAmmo. BagAmmo achieves an average attack success rate of over 99.9% on MaMaDroid, APIGraph and GCN, and still performs well in the scenario of concept drift and data imbalance. Moreover, BagAmmo outperforms the state-of-the-art attack SRL in attack success rate.



## **17. The Devil's Advocate: Shattering the Illusion of Unexploitable Data using Diffusion Models**

cs.LG

**SubmitDate**: 2023-03-15    [abs](http://arxiv.org/abs/2303.08500v1) [paper-pdf](http://arxiv.org/pdf/2303.08500v1)

**Authors**: Hadi M. Dolatabadi, Sarah Erfani, Christopher Leckie

**Abstract**: Protecting personal data against the exploitation of machine learning models is of paramount importance. Recently, availability attacks have shown great promise to provide an extra layer of protection against the unauthorized use of data to train neural networks. These methods aim to add imperceptible noise to clean data so that the neural networks cannot extract meaningful patterns from the protected data, claiming that they can make personal data "unexploitable." In this paper, we provide a strong countermeasure against such approaches, showing that unexploitable data might only be an illusion. In particular, we leverage the power of diffusion models and show that a carefully designed denoising process can defuse the ramifications of the data-protecting perturbations. We rigorously analyze our algorithm, and theoretically prove that the amount of required denoising is directly related to the magnitude of the data-protecting perturbations. Our approach, called AVATAR, delivers state-of-the-art performance against a suite of recent availability attacks in various scenarios, outperforming adversarial training. Our findings call for more research into making personal data unexploitable, showing that this goal is far from over.



## **18. Similarity of Neural Architectures Based on Input Gradient Transferability**

cs.LG

21pages, 10 figures, 1.5MB

**SubmitDate**: 2023-03-15    [abs](http://arxiv.org/abs/2210.11407v2) [paper-pdf](http://arxiv.org/pdf/2210.11407v2)

**Authors**: Jaehui Hwang, Dongyoon Han, Byeongho Heo, Song Park, Sanghyuk Chun, Jong-Seok Lee

**Abstract**: In recent years, a huge amount of deep neural architectures have been developed for image classification. It remains curious whether these models are similar or different and what factors contribute to their similarities or differences. To address this question, we aim to design a quantitative and scalable similarity function between neural architectures. We utilize adversarial attack transferability, which has information related to input gradients and decision boundaries that are widely used to understand model behaviors. We conduct a large-scale analysis on 69 state-of-the-art ImageNet classifiers using our proposed similarity function to answer the question. Moreover, we observe neural architecture-related phenomena using model similarity that model diversity can lead to better performance on model ensembles and knowledge distillation under specific conditions. Our results provide insights into why the development of diverse neural architectures with distinct components is necessary.



## **19. Quantum adversarial metric learning model based on triplet loss function**

quant-ph

arXiv admin note: substantial text overlap with arXiv:2303.07906

**SubmitDate**: 2023-03-15    [abs](http://arxiv.org/abs/2303.08293v1) [paper-pdf](http://arxiv.org/pdf/2303.08293v1)

**Authors**: Yan-Yan Hou, Jian Li, Xiu-Bo Chen, Chong-Qiang Ye

**Abstract**: Metric learning plays an essential role in image analysis and classification, and it has attracted more and more attention. In this paper, we propose a quantum adversarial metric learning (QAML) model based on the triplet loss function, where samples are embedded into the high-dimensional Hilbert space and the optimal metric is obtained by minimizing the triplet loss function. The QAML model employs entanglement and interference to build superposition states for triplet samples so that only one parameterized quantum circuit is needed to calculate sample distances, which reduces the demand for quantum resources. Considering the QAML model is fragile to adversarial attacks, an adversarial sample generation strategy is designed based on the quantum gradient ascent method, effectively improving the robustness against the functional adversarial attack. Simulation results show that the QAML model can effectively distinguish samples of MNIST and Iris datasets and has higher robustness accuracy over the general quantum metric learning. The QAML model is a fundamental research problem of machine learning. As a subroutine of classification and clustering tasks, the QAML model opens an avenue for exploring quantum advantages in machine learning.



## **20. Can Adversarial Examples Be Parsed to Reveal Victim Model Information?**

cs.CV

**SubmitDate**: 2023-03-15    [abs](http://arxiv.org/abs/2303.07474v2) [paper-pdf](http://arxiv.org/pdf/2303.07474v2)

**Authors**: Yuguang Yao, Jiancheng Liu, Yifan Gong, Xiaoming Liu, Yanzhi Wang, Xue Lin, Sijia Liu

**Abstract**: Numerous adversarial attack methods have been developed to generate imperceptible image perturbations that can cause erroneous predictions of state-of-the-art machine learning (ML) models, in particular, deep neural networks (DNNs). Despite intense research on adversarial attacks, little effort was made to uncover 'arcana' carried in adversarial attacks. In this work, we ask whether it is possible to infer data-agnostic victim model (VM) information (i.e., characteristics of the ML model or DNN used to generate adversarial attacks) from data-specific adversarial instances. We call this 'model parsing of adversarial attacks' - a task to uncover 'arcana' in terms of the concealed VM information in attacks. We approach model parsing via supervised learning, which correctly assigns classes of VM's model attributes (in terms of architecture type, kernel size, activation function, and weight sparsity) to an attack instance generated from this VM. We collect a dataset of adversarial attacks across 7 attack types generated from 135 victim models (configured by 5 architecture types, 3 kernel size setups, 3 activation function types, and 3 weight sparsity ratios). We show that a simple, supervised model parsing network (MPN) is able to infer VM attributes from unseen adversarial attacks if their attack settings are consistent with the training setting (i.e., in-distribution generalization assessment). We also provide extensive experiments to justify the feasibility of VM parsing from adversarial attacks, and the influence of training and evaluation factors in the parsing performance (e.g., generalization challenge raised in out-of-distribution evaluation). We further demonstrate how the proposed MPN can be used to uncover the source VM attributes from transfer attacks, and shed light on a potential connection between model parsing and attack transferability.



## **21. Improving Adversarial Robustness with Hypersphere Embedding and Angular-based Regularizations**

cs.LG

**SubmitDate**: 2023-03-15    [abs](http://arxiv.org/abs/2303.08289v1) [paper-pdf](http://arxiv.org/pdf/2303.08289v1)

**Authors**: Olukorede Fakorede, Ashutosh Nirala, Modeste Atsague, Jin Tian

**Abstract**: Adversarial training (AT) methods have been found to be effective against adversarial attacks on deep neural networks. Many variants of AT have been proposed to improve its performance. Pang et al. [1] have recently shown that incorporating hypersphere embedding (HE) into the existing AT procedures enhances robustness. We observe that the existing AT procedures are not designed for the HE framework, and thus fail to adequately learn the angular discriminative information available in the HE framework. In this paper, we propose integrating HE into AT with regularization terms that exploit the rich angular information available in the HE framework. Specifically, our method, termed angular-AT, adds regularization terms to AT that explicitly enforce weight-feature compactness and inter-class separation; all expressed in terms of angular features. Experimental results show that angular-AT further improves adversarial robustness.



## **22. Resilient Dynamic Average Consensus based on Trusted agents**

eess.SY

Initial Draft

**SubmitDate**: 2023-03-14    [abs](http://arxiv.org/abs/2303.08171v1) [paper-pdf](http://arxiv.org/pdf/2303.08171v1)

**Authors**: Shamik Bhattacharyya, Rachel Kalpana Kalaimani

**Abstract**: In this paper, we address the discrete-time dynamic average consensus (DAC) of a multi-agent system in the presence of adversarial attacks. The adversarial attack is considered to be of Byzantine type, which compromises the computation capabilities of the agent and sends arbitrary false data to its neighbours. We assume a few of the agents cannot be compromised by adversaries, which we term trusted agents. We first formally define resilient DAC in the presence of Byzantine adversaries. Then we propose our novel Resilient Dynamic Average Consensus (ResDAC) algorithm that ensures the trusted and ordinary agents achieve resilient DAC in the presence of adversarial agents. The only requirements are that of the trusted agents forming a connected dominating set and the first-order differences of the reference signals being bounded. We do not impose any restriction on the tolerable number of adversarial agents that can be present in the network. We also do not restrict the reference signals to be bounded. Finally, we provide numerical simulations to illustrate the effectiveness of the proposed ResDAC algorithm.



## **23. BODEGA: Benchmark for Adversarial Example Generation in Credibility Assessment**

cs.CL

**SubmitDate**: 2023-03-14    [abs](http://arxiv.org/abs/2303.08032v1) [paper-pdf](http://arxiv.org/pdf/2303.08032v1)

**Authors**: Piotr Przybyła, Alexander Shvets, Horacio Saggion

**Abstract**: Text classification methods have been widely investigated as a way to detect content of low credibility: fake news, social media bots, propaganda, etc. Quite accurate models (likely based on deep neural networks) help in moderating public electronic platforms and often cause content creators to face rejection of their submissions or removal of already published texts. Having the incentive to evade further detection, content creators try to come up with a slightly modified version of the text (known as an attack with an adversarial example) that exploit the weaknesses of classifiers and result in a different output. Here we introduce BODEGA: a benchmark for testing both victim models and attack methods on four misinformation detection tasks in an evaluation framework designed to simulate real use-cases of content moderation. We also systematically test the robustness of popular text classifiers against available attacking techniques and discover that, indeed, in some cases barely significant changes in input text can mislead the models. We openly share the BODEGA code and data in hope of enhancing the comparability and replicability of further research in this area.



## **24. Dynamic Efficient Adversarial Training Guided by Gradient Magnitude**

cs.LG

18 pages, 6 figures

**SubmitDate**: 2023-03-14    [abs](http://arxiv.org/abs/2103.03076v2) [paper-pdf](http://arxiv.org/pdf/2103.03076v2)

**Authors**: Fu Wang, Yanghao Zhang, Yanbin Zheng, Wenjie Ruan

**Abstract**: Adversarial training is an effective but time-consuming way to train robust deep neural networks that can withstand strong adversarial attacks. As a response to its inefficiency, we propose Dynamic Efficient Adversarial Training (DEAT), which gradually increases the adversarial iteration during training. We demonstrate that the gradient's magnitude correlates with the curvature of the trained model's loss landscape, allowing it to reflect the effect of adversarial training. Therefore, based on the magnitude of the gradient, we propose a general acceleration strategy, M+ acceleration, which enables an automatic and highly effective method of adjusting the training procedure. M+ acceleration is computationally efficient and easy to implement. It is suited for DEAT and compatible with the majority of existing adversarial training techniques. Extensive experiments have been done on CIFAR-10 and ImageNet datasets with various training environments. The results show that the proposed M+ acceleration significantly improves the training efficiency of existing adversarial training methods while achieving similar robustness performance. This demonstrates that the strategy is highly adaptive and offers a valuable solution for automatic adversarial training.



## **25. Constrained Adversarial Learning and its applicability to Automated Software Testing: a systematic review**

cs.SE

32 pages, 5 tables, 2 figures, Information and Software Technology  journal

**SubmitDate**: 2023-03-14    [abs](http://arxiv.org/abs/2303.07546v1) [paper-pdf](http://arxiv.org/pdf/2303.07546v1)

**Authors**: João Vitorino, Tiago Dias, Tiago Fonseca, Eva Maia, Isabel Praça

**Abstract**: Every novel technology adds hidden vulnerabilities ready to be exploited by a growing number of cyber-attacks. Automated software testing can be a promising solution to quickly analyze thousands of lines of code by generating and slightly modifying function-specific testing data to encounter a multitude of vulnerabilities and attack vectors. This process draws similarities to the constrained adversarial examples generated by adversarial learning methods, so there could be significant benefits to the integration of these methods in automated testing tools. Therefore, this systematic review is focused on the current state-of-the-art of constrained data generation methods applied for adversarial learning and software testing, aiming to guide researchers and developers to enhance testing tools with adversarial learning methods and improve the resilience and robustness of their digital systems. The found constrained data generation applications for adversarial machine learning were systematized, and the advantages and limitations of approaches specific for software testing were thoroughly analyzed, identifying research gaps and opportunities to improve testing tools with adversarial attack methods.



## **26. Symmetry Defense Against CNN Adversarial Perturbation Attacks**

cs.LG

13 pages

**SubmitDate**: 2023-03-13    [abs](http://arxiv.org/abs/2210.04087v2) [paper-pdf](http://arxiv.org/pdf/2210.04087v2)

**Authors**: Blerta Lindqvist

**Abstract**: Convolutional neural network classifiers (CNNs) are susceptible to adversarial attacks that perturb original samples to fool classifiers such as an autonomous vehicle's road sign image classifier. CNNs also lack invariance in the classification of symmetric samples because CNNs can classify symmetric samples differently. Considered together, the CNN lack of adversarial robustness and the CNN lack of invariance mean that the classification of symmetric adversarial samples can differ from their incorrect classification. Could symmetric adversarial samples revert to their correct classification? This paper answers this question by designing a symmetry defense that inverts or horizontally flips adversarial samples before classification against adversaries unaware of the defense. Against adversaries aware of the defense, the defense devises a Klein four symmetry subgroup that includes the horizontal flip and pixel inversion symmetries. The symmetry defense uses the subgroup symmetries in accuracy evaluation and the subgroup closure property to confine the transformations that an adaptive adversary can apply before or after generating the adversarial sample. Without changing the preprocessing, parameters, or model, the proposed symmetry defense counters the Projected Gradient Descent (PGD) and AutoAttack attacks with near-default accuracies for ImageNet. Without using attack knowledge or adversarial samples, the proposed defense exceeds the current best defense, which trains on adversarial samples. The defense maintains and even improves the classification accuracy of non-adversarial samples.



## **27. Review on the Feasibility of Adversarial Evasion Attacks and Defenses for Network Intrusion Detection Systems**

cs.CR

Under review (Submitted to Computer Networks - Elsevier)

**SubmitDate**: 2023-03-13    [abs](http://arxiv.org/abs/2303.07003v1) [paper-pdf](http://arxiv.org/pdf/2303.07003v1)

**Authors**: Islam Debicha, Benjamin Cochez, Tayeb Kenaza, Thibault Debatty, Jean-Michel Dricot, Wim Mees

**Abstract**: Nowadays, numerous applications incorporate machine learning (ML) algorithms due to their prominent achievements. However, many studies in the field of computer vision have shown that ML can be fooled by intentionally crafted instances, called adversarial examples. These adversarial examples take advantage of the intrinsic vulnerability of ML models. Recent research raises many concerns in the cybersecurity field. An increasing number of researchers are studying the feasibility of such attacks on security systems based on ML algorithms, such as Intrusion Detection Systems (IDS). The feasibility of such adversarial attacks would be influenced by various domain-specific constraints. This can potentially increase the difficulty of crafting adversarial examples. Despite the considerable amount of research that has been done in this area, much of it focuses on showing that it is possible to fool a model using features extracted from the raw data but does not address the practical side, i.e., the reverse transformation from theory to practice. For this reason, we propose a review browsing through various important papers to provide a comprehensive analysis. Our analysis highlights some challenges that have not been addressed in the reviewed papers.



## **28. Towards Making a Trojan-horse Attack on Text-to-Image Retrieval**

cs.MM

Accepted by ICASSP 2023

**SubmitDate**: 2023-03-13    [abs](http://arxiv.org/abs/2202.03861v4) [paper-pdf](http://arxiv.org/pdf/2202.03861v4)

**Authors**: Fan Hu, Aozhu Chen, Xirong Li

**Abstract**: While deep learning based image retrieval is reported to be vulnerable to adversarial attacks, existing works are mainly on image-to-image retrieval with their attacks performed at the front end via query modification. By contrast, we present in this paper the first study about a threat that occurs at the back end of a text-to-image retrieval (T2IR) system. Our study is motivated by the fact that the image collection indexed by the system will be regularly updated due to the arrival of new images from various sources such as web crawlers and advertisers. With malicious images indexed, it is possible for an attacker to indirectly interfere with the retrieval process, letting users see certain images that are completely irrelevant w.r.t. their queries. We put this thought into practice by proposing a novel Trojan-horse attack (THA). In particular, we construct a set of Trojan-horse images by first embedding word-specific adversarial information into a QR code and then putting the code on benign advertising images. A proof-of-concept evaluation, conducted on two popular T2IR datasets (Flickr30k and MS-COCO), shows the effectiveness of the proposed THA in a white-box mode.



## **29. Robust Contrastive Language-Image Pretraining against Adversarial Attacks**

cs.CV

**SubmitDate**: 2023-03-13    [abs](http://arxiv.org/abs/2303.06854v1) [paper-pdf](http://arxiv.org/pdf/2303.06854v1)

**Authors**: Wenhan Yang, Baharan Mirzasoleiman

**Abstract**: Contrastive vision-language representation learning has achieved state-of-the-art performance for zero-shot classification, by learning from millions of image-caption pairs crawled from the internet. However, the massive data that powers large multimodal models such as CLIP, makes them extremely vulnerable to various types of adversarial attacks, including targeted and backdoor data poisoning attacks. Despite this vulnerability, robust contrastive vision-language pretraining against adversarial attacks has remained unaddressed. In this work, we propose RoCLIP, the first effective method for robust pretraining {and fine-tuning} multimodal vision-language models. RoCLIP effectively breaks the association between poisoned image-caption pairs by considering a pool of random examples, and (1) matching every image with the text that is most similar to its caption in the pool, and (2) matching every caption with the image that is most similar to its image in the pool. Our extensive experiments show that our method renders state-of-the-art targeted data poisoning and backdoor attacks ineffective during pre-training or fine-tuning of CLIP. In particular, RoCLIP decreases the poison and backdoor attack success rates down to 0\% during pre-training and 1\%-4\% during fine-tuning, and effectively improves the model's performance.



## **30. Adversarial Attacks to Direct Data-driven Control for Destabilization**

eess.SY

6 pages

**SubmitDate**: 2023-03-13    [abs](http://arxiv.org/abs/2303.06837v1) [paper-pdf](http://arxiv.org/pdf/2303.06837v1)

**Authors**: Hampei Sasahara

**Abstract**: This study investigates the vulnerability of direct data-driven control to adversarial attacks in the form of a small but sophisticated perturbation added to the original data. The directed gradient sign method (DGSM) is developed as a specific attack method, based on the fast gradient sign method (FGSM), which has originally been considered in image classification. DGSM uses the gradient of the eigenvalues of the resulting closed-loop system and crafts a perturbation in the direction where the system becomes less stable. It is demonstrated that the system can be destabilized by the attack, even if the original closed-loop system with the clean data has a large margin of stability. To increase the robustness against the attack, regularization methods that have been developed to deal with random disturbances are considered. Their effectiveness is evaluated by numerical experiments using an inverted pendulum model.



## **31. Protecting Quantum Procrastinators with Signature Lifting: A Case Study in Cryptocurrencies**

cs.CR

**SubmitDate**: 2023-03-12    [abs](http://arxiv.org/abs/2303.06754v1) [paper-pdf](http://arxiv.org/pdf/2303.06754v1)

**Authors**: Or Sattath, Shai Wyborski

**Abstract**: Current solutions to quantum vulnerabilities of widely used cryptographic schemes involve migrating users to post-quantum schemes before quantum attacks become feasible. This work deals with protecting quantum procrastinators: users that failed to migrate to post-quantum cryptography in time.   To address this problem in the context of digital signatures, we introduce a technique called signature lifting, that allows us to lift a deployed pre-quantum signature scheme satisfying a certain property to a post-quantum signature scheme that uses the same keys. Informally, the said property is that a post-quantum one-way function is used "somewhere along the way" to derive the public-key from the secret-key. Our constructions of signature lifting relies heavily on the post-quantum digital signature scheme Picnic (Chase et al., CCS'17).   Our main case-study is cryptocurrencies, where this property holds in two scenarios: when the public-key is generated via a key-derivation function or when the public-key hash is posted instead of the public-key itself. We propose a modification, based on signature lifting, that can be applied in many cryptocurrencies for securely spending pre-quantum coins in presence of quantum adversaries. Our construction improves upon existing constructions in two major ways: it is not limited to pre-quantum coins whose ECDSA public-key has been kept secret (and in particular, it handles all coins that are stored in addresses generated by HD wallets), and it does not require access to post-quantum coins or using side payments to pay for posting the transaction.



## **32. DNN-Alias: Deep Neural Network Protection Against Side-Channel Attacks via Layer Balancing**

cs.CR

10 pages

**SubmitDate**: 2023-03-12    [abs](http://arxiv.org/abs/2303.06746v1) [paper-pdf](http://arxiv.org/pdf/2303.06746v1)

**Authors**: Mahya Morid Ahmadi, Lilas Alrahis, Ozgur Sinanoglu, Muhammad Shafique

**Abstract**: Extracting the architecture of layers of a given deep neural network (DNN) through hardware-based side channels allows adversaries to steal its intellectual property and even launch powerful adversarial attacks on the target system. In this work, we propose DNN-Alias, an obfuscation method for DNNs that forces all the layers in a given network to have similar execution traces, preventing attack models from differentiating between the layers. Towards this, DNN-Alias performs various layer-obfuscation operations, e.g., layer branching, layer deepening, etc, to alter the run-time traces while maintaining the functionality. DNN-Alias deploys an evolutionary algorithm to find the best combination of obfuscation operations in terms of maximizing the security level while maintaining a user-provided latency overhead budget. We demonstrate the effectiveness of our DNN-Alias technique by obfuscating the architecture of 700 randomly generated and obfuscated DNNs running on multiple Nvidia RTX 2080 TI GPU-based machines. Our experiments show that state-of-the-art side-channel architecture stealing attacks cannot extract the original DNN accurately. Moreover, we obfuscate the architecture of various DNNs, such as the VGG-11, VGG-13, ResNet-20, and ResNet-32 networks. Training the DNNs using the standard CIFAR10 dataset, we show that our DNN-Alias maintains the functionality of the original DNNs by preserving the original inference accuracy. Further, the experiments highlight that adversarial attack on obfuscated DNNs is unsuccessful.



## **33. Adv-Bot: Realistic Adversarial Botnet Attacks against Network Intrusion Detection Systems**

cs.CR

This work is published in Computers & Security (an Elsevier journal)  https://www.sciencedirect.com/science/article/pii/S016740482300086X

**SubmitDate**: 2023-03-12    [abs](http://arxiv.org/abs/2303.06664v1) [paper-pdf](http://arxiv.org/pdf/2303.06664v1)

**Authors**: Islam Debicha, Benjamin Cochez, Tayeb Kenaza, Thibault Debatty, Jean-Michel Dricot, Wim Mees

**Abstract**: Due to the numerous advantages of machine learning (ML) algorithms, many applications now incorporate them. However, many studies in the field of image classification have shown that MLs can be fooled by a variety of adversarial attacks. These attacks take advantage of ML algorithms' inherent vulnerability. This raises many questions in the cybersecurity field, where a growing number of researchers are recently investigating the feasibility of such attacks against machine learning-based security systems, such as intrusion detection systems. The majority of this research demonstrates that it is possible to fool a model using features extracted from a raw data source, but it does not take into account the real implementation of such attacks, i.e., the reverse transformation from theory to practice. The real implementation of these adversarial attacks would be influenced by various constraints that would make their execution more difficult. As a result, the purpose of this study was to investigate the actual feasibility of adversarial attacks, specifically evasion attacks, against network-based intrusion detection systems (NIDS), demonstrating that it is entirely possible to fool these ML-based IDSs using our proposed adversarial algorithm while assuming as many constraints as possible in a black-box setting. In addition, since it is critical to design defense mechanisms to protect ML-based IDSs against such attacks, a defensive scheme is presented. Realistic botnet traffic traces are used to assess this work. Our goal is to create adversarial botnet traffic that can avoid detection while still performing all of its intended malicious functionality.



## **34. Interpreting Hidden Semantics in the Intermediate Layers of 3D Point Cloud Classification Neural Network**

cs.CV

**SubmitDate**: 2023-03-12    [abs](http://arxiv.org/abs/2303.06652v1) [paper-pdf](http://arxiv.org/pdf/2303.06652v1)

**Authors**: Weiquan Liu, Minghao Liu, Shijun Zheng, Cheng Wang

**Abstract**: Although 3D point cloud classification neural network models have been widely used, the in-depth interpretation of the activation of the neurons and layers is still a challenge. We propose a novel approach, named Relevance Flow, to interpret the hidden semantics of 3D point cloud classification neural networks. It delivers the class Relevance to the activated neurons in the intermediate layers in a back-propagation manner, and associates the activation of neurons with the input points to visualize the hidden semantics of each layer. Specially, we reveal that the 3D point cloud classification neural network has learned the plane-level and part-level hidden semantics in the intermediate layers, and utilize the normal and IoU to evaluate the consistency of both levels' hidden semantics. Besides, by using the hidden semantics, we generate the adversarial attack samples to attack 3D point cloud classifiers. Experiments show that our proposed method reveals the hidden semantics of the 3D point cloud classification neural network on ModelNet40 and ShapeNet, which can be used for the unsupervised point cloud part segmentation without labels and attacking the 3D point cloud classifiers.



## **35. Query Attack by Multi-Identity Surrogates**

cs.LG

IEEE TRANSACTIONS ON ARTIFICIAL INTELLIGENCE

**SubmitDate**: 2023-03-12    [abs](http://arxiv.org/abs/2105.15010v5) [paper-pdf](http://arxiv.org/pdf/2105.15010v5)

**Authors**: Sizhe Chen, Zhehao Huang, Qinghua Tao, Xiaolin Huang

**Abstract**: Deep Neural Networks (DNNs) are acknowledged as vulnerable to adversarial attacks, while the existing black-box attacks require extensive queries on the victim DNN to achieve high success rates. For query-efficiency, surrogate models of the victim are used to generate transferable Adversarial Examples (AEs) because of their Gradient Similarity (GS), i.e., surrogates' attack gradients are similar to the victim's ones. However, it is generally neglected to exploit their similarity on outputs, namely the Prediction Similarity (PS), to filter out inefficient queries by surrogates without querying the victim. To jointly utilize and also optimize surrogates' GS and PS, we develop QueryNet, a unified attack framework that can significantly reduce queries. QueryNet creatively attacks by multi-identity surrogates, i.e., crafts several AEs for one sample by different surrogates, and also uses surrogates to decide on the most promising AE for the query. After that, the victim's query feedback is accumulated to optimize not only surrogates' parameters but also their architectures, enhancing both the GS and the PS. Although QueryNet has no access to pre-trained surrogates' prior, it reduces queries by averagely about an order of magnitude compared to alternatives within an acceptable time, according to our comprehensive experiments: 11 victims (including two commercial models) on MNIST/CIFAR10/ImageNet, allowing only 8-bit image queries, and no access to the victim's training data. The code is available at https://github.com/Sizhe-Chen/QueryNet.



## **36. Adaptive Local Adversarial Attacks on 3D Point Clouds for Augmented Reality**

cs.CV

**SubmitDate**: 2023-03-12    [abs](http://arxiv.org/abs/2303.06641v1) [paper-pdf](http://arxiv.org/pdf/2303.06641v1)

**Authors**: Weiquan Liu, Shijun Zheng, Cheng Wang

**Abstract**: As the key technology of augmented reality (AR), 3D recognition and tracking are always vulnerable to adversarial examples, which will cause serious security risks to AR systems. Adversarial examples are beneficial to improve the robustness of the 3D neural network model and enhance the stability of the AR system. At present, most 3D adversarial attack methods perturb the entire point cloud to generate adversarial examples, which results in high perturbation costs and difficulty in reconstructing the corresponding real objects in the physical world. In this paper, we propose an adaptive local adversarial attack method (AL-Adv) on 3D point clouds to generate adversarial point clouds. First, we analyze the vulnerability of the 3D network model and extract the salient regions of the input point cloud, namely the vulnerable regions. Second, we propose an adaptive gradient attack algorithm that targets vulnerable regions. The proposed attack algorithm adaptively assigns different disturbances in different directions of the three-dimensional coordinates of the point cloud. Experimental results show that our proposed method AL-Adv achieves a higher attack success rate than the global attack method. Specifically, the adversarial examples generated by the AL-Adv demonstrate good imperceptibility and small generation costs.



## **37. Multi-metrics adaptively identifies backdoors in Federated learning**

cs.CR

13 pages, 8 figures and 6 tables

**SubmitDate**: 2023-03-12    [abs](http://arxiv.org/abs/2303.06601v1) [paper-pdf](http://arxiv.org/pdf/2303.06601v1)

**Authors**: Siquan Huang, Yijiang Li, Chong Chen, Leyu Shi, Ying Gao

**Abstract**: The decentralized and privacy-preserving nature of federated learning (FL) makes it vulnerable to backdoor attacks aiming to manipulate the behavior of the resulting model on specific adversary-chosen inputs. However, most existing defenses based on statistical differences take effect only against specific attacks, especially when the malicious gradients are similar to benign ones or the data are highly non-independent and identically distributed (non-IID). In this paper, we revisit the distance-based defense methods and discover that i) Euclidean distance becomes meaningless in high dimensions and ii) malicious gradients with diverse characteristics cannot be identified by a single metric. To this end, we present a simple yet effective defense strategy with multi-metrics and dynamic weighting to identify backdoors adaptively. Furthermore, our novel defense has no reliance on predefined assumptions over attack settings or data distributions and little impact on benign performance. To evaluate the effectiveness of our approach, we conduct comprehensive experiments on different datasets under various attack settings, where our method achieves the best defensive performance. For instance, we achieve the lowest backdoor accuracy of 3.06% under the difficult Edge-case PGD, showing significant superiority over previous defenses. The results also demonstrate that our method can be well-adapted to a wide range of non-IID degrees without sacrificing the benign performance.



## **38. STPrivacy: Spatio-Temporal Privacy-Preserving Action Recognition**

cs.CV

**SubmitDate**: 2023-03-12    [abs](http://arxiv.org/abs/2301.03046v2) [paper-pdf](http://arxiv.org/pdf/2301.03046v2)

**Authors**: Ming Li, Xiangyu Xu, Hehe Fan, Pan Zhou, Jun Liu, Jia-Wei Liu, Jiahe Li, Jussi Keppo, Mike Zheng Shou, Shuicheng Yan

**Abstract**: Existing methods of privacy-preserving action recognition (PPAR) mainly focus on frame-level (spatial) privacy removal through 2D CNNs. Unfortunately, they have two major drawbacks. First, they may compromise temporal dynamics in input videos, which are critical for accurate action recognition. Second, they are vulnerable to practical attacking scenarios where attackers probe for privacy from an entire video rather than individual frames. To address these issues, we propose a novel framework STPrivacy to perform video-level PPAR. For the first time, we introduce vision Transformers into PPAR by treating a video as a tubelet sequence, and accordingly design two complementary mechanisms, i.e., sparsification and anonymization, to remove privacy from a spatio-temporal perspective. In specific, our privacy sparsification mechanism applies adaptive token selection to abandon action-irrelevant tubelets. Then, our anonymization mechanism implicitly manipulates the remaining action-tubelets to erase privacy in the embedding space through adversarial learning. These mechanisms provide significant advantages in terms of privacy preservation for human eyes and action-privacy trade-off adjustment during deployment. We additionally contribute the first two large-scale PPAR benchmarks, VP-HMDB51 and VP-UCF101, to the community. Extensive evaluations on them, as well as two other tasks, validate the effectiveness and generalization capability of our framework.



## **39. Disclosure Risk from Homogeneity Attack in Differentially Private Frequency Distribution**

cs.CR

**SubmitDate**: 2023-03-11    [abs](http://arxiv.org/abs/2101.00311v5) [paper-pdf](http://arxiv.org/pdf/2101.00311v5)

**Authors**: Fang Liu, Xingyuan Zhao

**Abstract**: Differential privacy (DP) provides a robust model to achieve privacy guarantees for released information. We examine the protection potency of sanitized multi-dimensional frequency distributions via DP randomization mechanisms against homogeneity attack (HA). HA allows adversaries to obtain the exact values on sensitive attributes for their targets without having to identify them from the released data. We propose measures for disclosure risk from HA and derive closed-form relationships between the privacy loss parameters in DP and the disclosure risk from HA. The availability of the closed-form relationships assists understanding the abstract concepts of DP and privacy loss parameters by putting them in the context of a concrete privacy attack and offers a perspective for choosing privacy loss parameters when employing DP mechanisms in information sanitization and release in practice. We apply the closed-form mathematical relationships in real-life datasets to demonstrate the assessment of disclosure risk due to HA on differentially private sanitized frequency distributions at various privacy loss parameters.



## **40. Anomaly Detection with Ensemble of Encoder and Decoder**

cs.LG

**SubmitDate**: 2023-03-11    [abs](http://arxiv.org/abs/2303.06431v1) [paper-pdf](http://arxiv.org/pdf/2303.06431v1)

**Authors**: Xijuan Sun, Di Wu, Arnaud Zinflou, Benoit Boulet

**Abstract**: Hacking and false data injection from adversaries can threaten power grids' everyday operations and cause significant economic loss. Anomaly detection in power grids aims to detect and discriminate anomalies caused by cyber attacks against the power system, which is essential for keeping power grids working correctly and efficiently. Different methods have been applied for anomaly detection, such as statistical methods and machine learning-based methods. Usually, machine learning-based methods need to model the normal data distribution. In this work, we propose a novel anomaly detection method by modeling the data distribution of normal samples via multiple encoders and decoders. Specifically, the proposed method maps input samples into a latent space and then reconstructs output samples from latent vectors. The extra encoder finally maps reconstructed samples to latent representations. During the training phase, we optimize parameters by minimizing the reconstruction loss and encoding loss. Training samples are re-weighted to focus more on missed correlations between features of normal data. Furthermore, we employ the long short-term memory model as encoders and decoders to test its effectiveness. We also investigate a meta-learning-based framework for hyper-parameter tuning of our approach. Experiment results on network intrusion and power system datasets demonstrate the effectiveness of our proposed method, where our models consistently outperform all baselines.



## **41. Improving the Robustness of Deep Convolutional Neural Networks Through Feature Learning**

cs.CV

8 pages, 12 figures, 6 tables. Work in process

**SubmitDate**: 2023-03-11    [abs](http://arxiv.org/abs/2303.06425v1) [paper-pdf](http://arxiv.org/pdf/2303.06425v1)

**Authors**: Jin Ding, Jie-Chao Zhao, Yong-Zhi Sun, Ping Tan, Ji-En Ma, You-Tong Fang

**Abstract**: Deep convolutional neural network (DCNN for short) models are vulnerable to examples with small perturbations. Adversarial training (AT for short) is a widely used approach to enhance the robustness of DCNN models by data augmentation. In AT, the DCNN models are trained with clean examples and adversarial examples (AE for short) which are generated using a specific attack method, aiming to gain ability to defend themselves when facing the unseen AEs. However, in practice, the trained DCNN models are often fooled by the AEs generated by the novel attack methods. This naturally raises a question: can a DCNN model learn certain features which are insensitive to small perturbations, and further defend itself no matter what attack methods are presented. To answer this question, this paper makes a beginning effort by proposing a shallow binary feature module (SBFM for short), which can be integrated into any popular backbone. The SBFM includes two types of layers, i.e., Sobel layer and threshold layer. In Sobel layer, there are four parallel feature maps which represent horizontal, vertical, and diagonal edge features, respectively. And in threshold layer, it turns the edge features learnt by Sobel layer to the binary features, which then are feeded into the fully connected layers for classification with the features learnt by the backbone. We integrate SBFM into VGG16 and ResNet34, respectively, and conduct experiments on multiple datasets. Experimental results demonstrate, under FGSM attack with $\epsilon=8/255$, the SBFM integrated models can achieve averagely 35\% higher accuracy than the original ones, and in CIFAR-10 and TinyImageNet datasets, the SBFM integrated models can achieve averagely 75\% classification accuracy. The work in this paper shows it is promising to enhance the robustness of DCNN models through feature learning.



## **42. MorDIFF: Recognition Vulnerability and Attack Detectability of Face Morphing Attacks Created by Diffusion Autoencoders**

cs.CV

Accepted at the 11th International Workshop on Biometrics and  Forensics 2023 (IWBF 2023)

**SubmitDate**: 2023-03-11    [abs](http://arxiv.org/abs/2302.01843v2) [paper-pdf](http://arxiv.org/pdf/2302.01843v2)

**Authors**: Naser Damer, Meiling Fang, Patrick Siebke, Jan Niklas Kolf, Marco Huber, Fadi Boutros

**Abstract**: Investigating new methods of creating face morphing attacks is essential to foresee novel attacks and help mitigate them. Creating morphing attacks is commonly either performed on the image-level or on the representation-level. The representation-level morphing has been performed so far based on generative adversarial networks (GAN) where the encoded images are interpolated in the latent space to produce a morphed image based on the interpolated vector. Such a process was constrained by the limited reconstruction fidelity of GAN architectures. Recent advances in the diffusion autoencoder models have overcome the GAN limitations, leading to high reconstruction fidelity. This theoretically makes them a perfect candidate to perform representation-level face morphing. This work investigates using diffusion autoencoders to create face morphing attacks by comparing them to a wide range of image-level and representation-level morphs. Our vulnerability analyses on four state-of-the-art face recognition models have shown that such models are highly vulnerable to the created attacks, the MorDIFF, especially when compared to existing representation-level morphs. Detailed detectability analyses are also performed on the MorDIFF, showing that they are as challenging to detect as other morphing attacks created on the image- or representation-level. Data and morphing script are made public: https://github.com/naserdamer/MorDIFF.



## **43. Adversarial Attacks and Defenses in Machine Learning-Powered Networks: A Contemporary Survey**

cs.LG

46 pages, 21 figures

**SubmitDate**: 2023-03-11    [abs](http://arxiv.org/abs/2303.06302v1) [paper-pdf](http://arxiv.org/pdf/2303.06302v1)

**Authors**: Yulong Wang, Tong Sun, Shenghong Li, Xin Yuan, Wei Ni, Ekram Hossain, H. Vincent Poor

**Abstract**: Adversarial attacks and defenses in machine learning and deep neural network have been gaining significant attention due to the rapidly growing applications of deep learning in the Internet and relevant scenarios. This survey provides a comprehensive overview of the recent advancements in the field of adversarial attack and defense techniques, with a focus on deep neural network-based classification models. Specifically, we conduct a comprehensive classification of recent adversarial attack methods and state-of-the-art adversarial defense techniques based on attack principles, and present them in visually appealing tables and tree diagrams. This is based on a rigorous evaluation of the existing works, including an analysis of their strengths and limitations. We also categorize the methods into counter-attack detection and robustness enhancement, with a specific focus on regularization-based methods for enhancing robustness. New avenues of attack are also explored, including search-based, decision-based, drop-based, and physical-world attacks, and a hierarchical classification of the latest defense methods is provided, highlighting the challenges of balancing training costs with performance, maintaining clean accuracy, overcoming the effect of gradient masking, and ensuring method transferability. At last, the lessons learned and open challenges are summarized with future research opportunities recommended.



## **44. Investigating Stateful Defenses Against Black-Box Adversarial Examples**

cs.CR

**SubmitDate**: 2023-03-11    [abs](http://arxiv.org/abs/2303.06280v1) [paper-pdf](http://arxiv.org/pdf/2303.06280v1)

**Authors**: Ryan Feng, Ashish Hooda, Neal Mangaokar, Kassem Fawaz, Somesh Jha, Atul Prakash

**Abstract**: Defending machine-learning (ML) models against white-box adversarial attacks has proven to be extremely difficult. Instead, recent work has proposed stateful defenses in an attempt to defend against a more restricted black-box attacker. These defenses operate by tracking a history of incoming model queries, and rejecting those that are suspiciously similar. The current state-of-the-art stateful defense Blacklight was proposed at USENIX Security '22 and claims to prevent nearly 100% of attacks on both the CIFAR10 and ImageNet datasets. In this paper, we observe that an attacker can significantly reduce the accuracy of a Blacklight-protected classifier (e.g., from 82.2% to 6.4% on CIFAR10) by simply adjusting the parameters of an existing black-box attack. Motivated by this surprising observation, since existing attacks were evaluated by the Blacklight authors, we provide a systematization of stateful defenses to understand why existing stateful defense models fail. Finally, we propose a stronger evaluation strategy for stateful defenses comprised of adaptive score and hard-label based black-box attacks. We use these attacks to successfully reduce even reconfigured versions of Blacklight to as low as 0% robust accuracy.



## **45. Do we need entire training data for adversarial training?**

cs.CV

6 pages, 4 figures

**SubmitDate**: 2023-03-10    [abs](http://arxiv.org/abs/2303.06241v1) [paper-pdf](http://arxiv.org/pdf/2303.06241v1)

**Authors**: Vipul Gupta, Apurva Narayan

**Abstract**: Deep Neural Networks (DNNs) are being used to solve a wide range of problems in many domains including safety-critical domains like self-driving cars and medical imagery. DNNs suffer from vulnerability against adversarial attacks. In the past few years, numerous approaches have been proposed to tackle this problem by training networks using adversarial training. Almost all the approaches generate adversarial examples for the entire training dataset, thus increasing the training time drastically. We show that we can decrease the training time for any adversarial training algorithm by using only a subset of training data for adversarial training. To select the subset, we filter the adversarially-prone samples from the training data. We perform a simple adversarial attack on all training examples to filter this subset. In this attack, we add a small perturbation to each pixel and a few grid lines to the input image.   We perform adversarial training on the adversarially-prone subset and mix it with vanilla training performed on the entire dataset. Our results show that when our method-agnostic approach is plugged into FGSM, we achieve a speedup of 3.52x on MNIST and 1.98x on the CIFAR-10 dataset with comparable robust accuracy. We also test our approach on state-of-the-art Free adversarial training and achieve a speedup of 1.2x in training time with a marginal drop in robust accuracy on the ImageNet dataset.



## **46. Turning Strengths into Weaknesses: A Certified Robustness Inspired Attack Framework against Graph Neural Networks**

cs.CR

Accepted by CVPR 2023

**SubmitDate**: 2023-03-10    [abs](http://arxiv.org/abs/2303.06199v1) [paper-pdf](http://arxiv.org/pdf/2303.06199v1)

**Authors**: Binghui Wang, Meng Pang, Yun Dong

**Abstract**: Graph neural networks (GNNs) have achieved state-of-the-art performance in many graph learning tasks. However, recent studies show that GNNs are vulnerable to both test-time evasion and training-time poisoning attacks that perturb the graph structure. While existing attack methods have shown promising attack performance, we would like to design an attack framework to further enhance the performance. In particular, our attack framework is inspired by certified robustness, which was originally used by defenders to defend against adversarial attacks. We are the first, from the attacker perspective, to leverage its properties to better attack GNNs. Specifically, we first derive nodes' certified perturbation sizes against graph evasion and poisoning attacks based on randomized smoothing, respectively. A larger certified perturbation size of a node indicates this node is theoretically more robust to graph perturbations. Such a property motivates us to focus more on nodes with smaller certified perturbation sizes, as they are easier to be attacked after graph perturbations. Accordingly, we design a certified robustness inspired attack loss, when incorporated into (any) existing attacks, produces our certified robustness inspired attack counterpart. We apply our framework to the existing attacks and results show it can significantly enhance the existing base attacks' performance.



## **47. Harnessing the Speed and Accuracy of Machine Learning to Advance Cybersecurity**

cs.CR

**SubmitDate**: 2023-03-10    [abs](http://arxiv.org/abs/2302.12415v2) [paper-pdf](http://arxiv.org/pdf/2302.12415v2)

**Authors**: Khatoon Mohammed

**Abstract**: As cyber attacks continue to increase in frequency and sophistication, detecting malware has become a critical task for maintaining the security of computer systems. Traditional signature-based methods of malware detection have limitations in detecting complex and evolving threats. In recent years, machine learning (ML) has emerged as a promising solution to detect malware effectively. ML algorithms are capable of analyzing large datasets and identifying patterns that are difficult for humans to identify. This paper presents a comprehensive review of the state-of-the-art ML techniques used in malware detection, including supervised and unsupervised learning, deep learning, and reinforcement learning. We also examine the challenges and limitations of ML-based malware detection, such as the potential for adversarial attacks and the need for large amounts of labeled data. Furthermore, we discuss future directions in ML-based malware detection, including the integration of multiple ML algorithms and the use of explainable AI techniques to enhance the interpret ability of ML-based detection systems. Our research highlights the potential of ML-based techniques to improve the speed and accuracy of malware detection, and contribute to enhancing cybersecurity



## **48. Learning the Legibility of Visual Text Perturbations**

cs.CL

14 pages, 7 figures. Accepted at EACL 2023 (main, long)

**SubmitDate**: 2023-03-10    [abs](http://arxiv.org/abs/2303.05077v2) [paper-pdf](http://arxiv.org/pdf/2303.05077v2)

**Authors**: Dev Seth, Rickard Stureborg, Danish Pruthi, Bhuwan Dhingra

**Abstract**: Many adversarial attacks in NLP perturb inputs to produce visually similar strings ('ergo' $\rightarrow$ '$\epsilon$rgo') which are legible to humans but degrade model performance. Although preserving legibility is a necessary condition for text perturbation, little work has been done to systematically characterize it; instead, legibility is typically loosely enforced via intuitions around the nature and extent of perturbations. Particularly, it is unclear to what extent can inputs be perturbed while preserving legibility, or how to quantify the legibility of a perturbed string. In this work, we address this gap by learning models that predict the legibility of a perturbed string, and rank candidate perturbations based on their legibility. To do so, we collect and release LEGIT, a human-annotated dataset comprising the legibility of visually perturbed text. Using this dataset, we build both text- and vision-based models which achieve up to $0.91$ F1 score in predicting whether an input is legible, and an accuracy of $0.86$ in predicting which of two given perturbations is more legible. Additionally, we discover that legible perturbations from the LEGIT dataset are more effective at lowering the performance of NLP models than best-known attack strategies, suggesting that current models may be vulnerable to a broad range of perturbations beyond what is captured by existing visual attacks. Data, code, and models are available at https://github.com/dvsth/learning-legibility-2023.



## **49. Exploring the Relationship between Architecture and Adversarially Robust Generalization**

cs.LG

**SubmitDate**: 2023-03-10    [abs](http://arxiv.org/abs/2209.14105v2) [paper-pdf](http://arxiv.org/pdf/2209.14105v2)

**Authors**: Aishan Liu, Shiyu Tang, Siyuan Liang, Ruihao Gong, Boxi Wu, Xianglong Liu, Dacheng Tao

**Abstract**: Adversarial training has been demonstrated to be one of the most effective remedies for defending adversarial examples, yet it often suffers from the huge robustness generalization gap on unseen testing adversaries, deemed as the adversarially robust generalization problem. Despite the preliminary understandings devoted to adversarially robust generalization, little is known from the architectural perspective. To bridge the gap, this paper for the first time systematically investigated the relationship between adversarially robust generalization and architectural design. Inparticular, we comprehensively evaluated 20 most representative adversarially trained architectures on ImageNette and CIFAR-10 datasets towards multiple `p-norm adversarial attacks. Based on the extensive experiments, we found that, under aligned settings, Vision Transformers (e.g., PVT, CoAtNet) often yield better adversarially robust generalization while CNNs tend to overfit on specific attacks and fail to generalize on multiple adversaries. To better understand the nature behind it, we conduct theoretical analysis via the lens of Rademacher complexity. We revealed the fact that the higher weight sparsity contributes significantly towards the better adversarially robust generalization of Transformers, which can be often achieved by the specially-designed attention blocks. We hope our paper could help to better understand the mechanism for designing robust DNNs. Our model weights can be found at http://robust.art.



## **50. Machine Learning Security in Industry: A Quantitative Survey**

cs.LG

Accepted at TIFS, version with more detailed appendix containing more  detailed statistical results. 17 pages, 6 tables and 4 figures

**SubmitDate**: 2023-03-10    [abs](http://arxiv.org/abs/2207.05164v2) [paper-pdf](http://arxiv.org/pdf/2207.05164v2)

**Authors**: Kathrin Grosse, Lukas Bieringer, Tarek Richard Besold, Battista Biggio, Katharina Krombholz

**Abstract**: Despite the large body of academic work on machine learning security, little is known about the occurrence of attacks on machine learning systems in the wild. In this paper, we report on a quantitative study with 139 industrial practitioners. We analyze attack occurrence and concern and evaluate statistical hypotheses on factors influencing threat perception and exposure. Our results shed light on real-world attacks on deployed machine learning. On the organizational level, while we find no predictors for threat exposure in our sample, the amount of implement defenses depends on exposure to threats or expected likelihood to become a target. We also provide a detailed analysis of practitioners' replies on the relevance of individual machine learning attacks, unveiling complex concerns like unreliable decision making, business information leakage, and bias introduction into models. Finally, we find that on the individual level, prior knowledge about machine learning security influences threat perception. Our work paves the way for more research about adversarial machine learning in practice, but yields also insights for regulation and auditing.



