# Latest Adversarial Attack Papers
**update at 2023-04-10 10:31:53**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. AMS-DRL: Learning Multi-Pursuit Evasion for Safe Targeted Navigation of Drones**

cs.RO

**SubmitDate**: 2023-04-07    [abs](http://arxiv.org/abs/2304.03443v1) [paper-pdf](http://arxiv.org/pdf/2304.03443v1)

**Authors**: Jiaping Xiao, Mir Feroskhan

**Abstract**: Safe navigation of drones in the presence of adversarial physical attacks from multiple pursuers is a challenging task. This paper proposes a novel approach, asynchronous multi-stage deep reinforcement learning (AMS-DRL), to train an adversarial neural network that can learn from the actions of multiple pursuers and adapt quickly to their behavior, enabling the drone to avoid attacks and reach its target. Our approach guarantees convergence by ensuring Nash Equilibrium among agents from the game-theory analysis. We evaluate our method in extensive simulations and show that it outperforms baselines with higher navigation success rates. We also analyze how parameters such as the relative maximum speed affect navigation performance. Furthermore, we have conducted physical experiments and validated the effectiveness of the trained policies in real-time flights. A success rate heatmap is introduced to elucidate how spatial geometry influences navigation outcomes. Project website: https://github.com/NTU-UAVG/AMS-DRL-for-Pursuit-Evasion.



## **2. LP-BFGS attack: An adversarial attack based on the Hessian with limited pixels**

cs.CR

15 pages, 7 figures

**SubmitDate**: 2023-04-07    [abs](http://arxiv.org/abs/2210.15446v2) [paper-pdf](http://arxiv.org/pdf/2210.15446v2)

**Authors**: Jiebao Zhang, Wenhua Qian, Rencan Nie, Jinde Cao, Dan Xu

**Abstract**: Deep neural networks are vulnerable to adversarial attacks. Most $L_{0}$-norm based white-box attacks craft perturbations by the gradient of models to the input. Since the computation cost and memory limitation of calculating the Hessian matrix, the application of Hessian or approximate Hessian in white-box attacks is gradually shelved. In this work, we note that the sparsity requirement on perturbations naturally lends itself to the usage of Hessian information. We study the attack performance and computation cost of the attack method based on the Hessian with a limited number of perturbation pixels. Specifically, we propose the Limited Pixel BFGS (LP-BFGS) attack method by incorporating the perturbation pixel selection strategy and the BFGS algorithm. Pixels with top-k attribution scores calculated by the Integrated Gradient method are regarded as optimization variables of the LP-BFGS attack. Experimental results across different networks and datasets demonstrate that our approach has comparable attack ability with reasonable computation in different numbers of perturbation pixels compared with existing solutions.



## **3. EZClone: Improving DNN Model Extraction Attack via Shape Distillation from GPU Execution Profiles**

cs.LG

11 pages, 6 tables, 4 figures

**SubmitDate**: 2023-04-06    [abs](http://arxiv.org/abs/2304.03388v1) [paper-pdf](http://arxiv.org/pdf/2304.03388v1)

**Authors**: Jonah O'Brien Weiss, Tiago Alves, Sandip Kundu

**Abstract**: Deep Neural Networks (DNNs) have become ubiquitous due to their performance on prediction and classification problems. However, they face a variety of threats as their usage spreads. Model extraction attacks, which steal DNNs, endanger intellectual property, data privacy, and security. Previous research has shown that system-level side-channels can be used to leak the architecture of a victim DNN, exacerbating these risks. We propose two DNN architecture extraction techniques catering to various threat models. The first technique uses a malicious, dynamically linked version of PyTorch to expose a victim DNN architecture through the PyTorch profiler. The second, called EZClone, exploits aggregate (rather than time-series) GPU profiles as a side-channel to predict DNN architecture, employing a simple approach and assuming little adversary capability as compared to previous work. We investigate the effectiveness of EZClone when minimizing the complexity of the attack, when applied to pruned models, and when applied across GPUs. We find that EZClone correctly predicts DNN architectures for the entire set of PyTorch vision architectures with 100% accuracy. No other work has shown this degree of architecture prediction accuracy with the same adversarial constraints or using aggregate side-channel information. Prior work has shown that, once a DNN has been successfully cloned, further attacks such as model evasion or model inversion can be accelerated significantly.



## **4. Reliable Learning for Test-time Attacks and Distribution Shift**

cs.LG

**SubmitDate**: 2023-04-06    [abs](http://arxiv.org/abs/2304.03370v1) [paper-pdf](http://arxiv.org/pdf/2304.03370v1)

**Authors**: Maria-Florina Balcan, Steve Hanneke, Rattana Pukdee, Dravyansh Sharma

**Abstract**: Machine learning algorithms are often used in environments which are not captured accurately even by the most carefully obtained training data, either due to the possibility of `adversarial' test-time attacks, or on account of `natural' distribution shift. For test-time attacks, we introduce and analyze a novel robust reliability guarantee, which requires a learner to output predictions along with a reliability radius $\eta$, with the meaning that its prediction is guaranteed to be correct as long as the adversary has not perturbed the test point farther than a distance $\eta$. We provide learners that are optimal in the sense that they always output the best possible reliability radius on any test point, and we characterize the reliable region, i.e. the set of points where a given reliability radius is attainable. We additionally analyze reliable learners under distribution shift, where the test points may come from an arbitrary distribution Q different from the training distribution P. For both cases, we bound the probability mass of the reliable region for several interesting examples, for linear separators under nearly log-concave and s-concave distributions, as well as for smooth boundary classifiers under smooth probability distributions.



## **5. Improving Visual Question Answering Models through Robustness Analysis and In-Context Learning with a Chain of Basic Questions**

cs.CV

28 pages

**SubmitDate**: 2023-04-06    [abs](http://arxiv.org/abs/2304.03147v1) [paper-pdf](http://arxiv.org/pdf/2304.03147v1)

**Authors**: Jia-Hong Huang, Modar Alfadly, Bernard Ghanem, Marcel Worring

**Abstract**: Deep neural networks have been critical in the task of Visual Question Answering (VQA), with research traditionally focused on improving model accuracy. Recently, however, there has been a trend towards evaluating the robustness of these models against adversarial attacks. This involves assessing the accuracy of VQA models under increasing levels of noise in the input, which can target either the image or the proposed query question, dubbed the main question. However, there is currently a lack of proper analysis of this aspect of VQA. This work proposes a new method that utilizes semantically related questions, referred to as basic questions, acting as noise to evaluate the robustness of VQA models. It is hypothesized that as the similarity of a basic question to the main question decreases, the level of noise increases. To generate a reasonable noise level for a given main question, a pool of basic questions is ranked based on their similarity to the main question, and this ranking problem is cast as a LASSO optimization problem. Additionally, this work proposes a novel robustness measure, R_score, and two basic question datasets to standardize the analysis of VQA model robustness. The experimental results demonstrate that the proposed evaluation method effectively analyzes the robustness of VQA models. Moreover, the experiments show that in-context learning with a chain of basic questions can enhance model accuracy.



## **6. Public Key Encryption with Secure Key Leasing**

quant-ph

68 pages, 4 figures. added related works and a comparison with a  concurrent work (2023-04-07)

**SubmitDate**: 2023-04-06    [abs](http://arxiv.org/abs/2302.11663v2) [paper-pdf](http://arxiv.org/pdf/2302.11663v2)

**Authors**: Shweta Agrawal, Fuyuki Kitagawa, Ryo Nishimaki, Shota Yamada, Takashi Yamakawa

**Abstract**: We introduce the notion of public key encryption with secure key leasing (PKE-SKL). Our notion supports the leasing of decryption keys so that a leased key achieves the decryption functionality but comes with the guarantee that if the quantum decryption key returned by a user passes a validity test, then the user has lost the ability to decrypt. Our notion is similar in spirit to the notion of secure software leasing (SSL) introduced by Ananth and La Placa (Eurocrypt 2021) but captures significantly more general adversarial strategies. In more detail, our adversary is not restricted to use an honest evaluation algorithm to run pirated software. Our results can be summarized as follows:   1. Definitions: We introduce the definition of PKE with secure key leasing and formalize security notions.   2. Constructing PKE with Secure Key Leasing: We provide a construction of PKE-SKL by leveraging a PKE scheme that satisfies a new security notion that we call consistent or inconsistent security against key leasing attacks (CoIC-KLA security). We then construct a CoIC-KLA secure PKE scheme using 1-key Ciphertext-Policy Functional Encryption (CPFE) that in turn can be based on any IND-CPA secure PKE scheme.   3. Identity Based Encryption, Attribute Based Encryption and Functional Encryption with Secure Key Leasing: We provide definitions of secure key leasing in the context of advanced encryption schemes such as identity based encryption (IBE), attribute-based encryption (ABE) and functional encryption (FE). Then we provide constructions by combining the above PKE-SKL with standard IBE, ABE and FE schemes.



## **7. StratDef: Strategic Defense Against Adversarial Attacks in ML-based Malware Detection**

cs.LG

**SubmitDate**: 2023-04-06    [abs](http://arxiv.org/abs/2202.07568v5) [paper-pdf](http://arxiv.org/pdf/2202.07568v5)

**Authors**: Aqib Rashid, Jose Such

**Abstract**: Over the years, most research towards defenses against adversarial attacks on machine learning models has been in the image recognition domain. The malware detection domain has received less attention despite its importance. Moreover, most work exploring these defenses has focused on several methods but with no strategy when applying them. In this paper, we introduce StratDef, which is a strategic defense system based on a moving target defense approach. We overcome challenges related to the systematic construction, selection, and strategic use of models to maximize adversarial robustness. StratDef dynamically and strategically chooses the best models to increase the uncertainty for the attacker while minimizing critical aspects in the adversarial ML domain, like attack transferability. We provide the first comprehensive evaluation of defenses against adversarial attacks on machine learning for malware detection, where our threat model explores different levels of threat, attacker knowledge, capabilities, and attack intensities. We show that StratDef performs better than other defenses even when facing the peak adversarial threat. We also show that, of the existing defenses, only a few adversarially-trained models provide substantially better protection than just using vanilla models but are still outperformed by StratDef.



## **8. PAD: Towards Principled Adversarial Malware Detection Against Evasion Attacks**

cs.CR

Accepted by IEEE Transactions on Dependable and Secure Computing; To  appear

**SubmitDate**: 2023-04-06    [abs](http://arxiv.org/abs/2302.11328v2) [paper-pdf](http://arxiv.org/pdf/2302.11328v2)

**Authors**: Deqiang Li, Shicheng Cui, Yun Li, Jia Xu, Fu Xiao, Shouhuai Xu

**Abstract**: Machine Learning (ML) techniques can facilitate the automation of malicious software (malware for short) detection, but suffer from evasion attacks. Many studies counter such attacks in heuristic manners, lacking theoretical guarantees and defense effectiveness. In this paper, we propose a new adversarial training framework, termed Principled Adversarial Malware Detection (PAD), which offers convergence guarantees for robust optimization methods. PAD lays on a learnable convex measurement that quantifies distribution-wise discrete perturbations to protect malware detectors from adversaries, whereby for smooth detectors, adversarial training can be performed with theoretical treatments. To promote defense effectiveness, we propose a new mixture of attacks to instantiate PAD to enhance deep neural network-based measurements and malware detectors. Experimental results on two Android malware datasets demonstrate: (i) the proposed method significantly outperforms the state-of-the-art defenses; (ii) it can harden ML-based malware detection against 27 evasion attacks with detection accuracies greater than 83.45%, at the price of suffering an accuracy decrease smaller than 2.16% in the absence of attacks; (iii) it matches or outperforms many anti-malware scanners in VirusTotal against realistic adversarial malware.



## **9. Robust Neural Architecture Search**

cs.LG

**SubmitDate**: 2023-04-06    [abs](http://arxiv.org/abs/2304.02845v1) [paper-pdf](http://arxiv.org/pdf/2304.02845v1)

**Authors**: Xunyu Zhu, Jian Li, Yong Liu, Weiping Wang

**Abstract**: Neural Architectures Search (NAS) becomes more and more popular over these years. However, NAS-generated models tends to suffer greater vulnerability to various malicious attacks. Lots of robust NAS methods leverage adversarial training to enhance the robustness of NAS-generated models, however, they neglected the nature accuracy of NAS-generated models. In our paper, we propose a novel NAS method, Robust Neural Architecture Search (RNAS). To design a regularization term to balance accuracy and robustness, RNAS generates architectures with both high accuracy and good robustness. To reduce search cost, we further propose to use noise examples instead adversarial examples as input to search architectures. Extensive experiments show that RNAS achieves state-of-the-art (SOTA) performance on both image classification and adversarial attacks, which illustrates the proposed RNAS achieves a good tradeoff between robustness and accuracy.



## **10. Robust Upper Bounds for Adversarial Training**

cs.LG

**SubmitDate**: 2023-04-06    [abs](http://arxiv.org/abs/2112.09279v2) [paper-pdf](http://arxiv.org/pdf/2112.09279v2)

**Authors**: Dimitris Bertsimas, Xavier Boix, Kimberly Villalobos Carballo, Dick den Hertog

**Abstract**: Many state-of-the-art adversarial training methods for deep learning leverage upper bounds of the adversarial loss to provide security guarantees against adversarial attacks. Yet, these methods rely on convex relaxations to propagate lower and upper bounds for intermediate layers, which affect the tightness of the bound at the output layer. We introduce a new approach to adversarial training by minimizing an upper bound of the adversarial loss that is based on a holistic expansion of the network instead of separate bounds for each layer. This bound is facilitated by state-of-the-art tools from Robust Optimization; it has closed-form and can be effectively trained using backpropagation. We derive two new methods with the proposed approach. The first method (Approximated Robust Upper Bound or aRUB) uses the first order approximation of the network as well as basic tools from Linear Robust Optimization to obtain an empirical upper bound of the adversarial loss that can be easily implemented. The second method (Robust Upper Bound or RUB), computes a provable upper bound of the adversarial loss. Across a variety of tabular and vision data sets we demonstrate the effectiveness of our approach -- RUB is substantially more robust than state-of-the-art methods for larger perturbations, while aRUB matches the performance of state-of-the-art methods for small perturbations.



## **11. Improving Fast Adversarial Training with Prior-Guided Knowledge**

cs.LG

**SubmitDate**: 2023-04-06    [abs](http://arxiv.org/abs/2304.00202v2) [paper-pdf](http://arxiv.org/pdf/2304.00202v2)

**Authors**: Xiaojun Jia, Yong Zhang, Xingxing Wei, Baoyuan Wu, Ke Ma, Jue Wang, Xiaochun Cao

**Abstract**: Fast adversarial training (FAT) is an efficient method to improve robustness. However, the original FAT suffers from catastrophic overfitting, which dramatically and suddenly reduces robustness after a few training epochs. Although various FAT variants have been proposed to prevent overfitting, they require high training costs. In this paper, we investigate the relationship between adversarial example quality and catastrophic overfitting by comparing the training processes of standard adversarial training and FAT. We find that catastrophic overfitting occurs when the attack success rate of adversarial examples becomes worse. Based on this observation, we propose a positive prior-guided adversarial initialization to prevent overfitting by improving adversarial example quality without extra training costs. This initialization is generated by using high-quality adversarial perturbations from the historical training process. We provide theoretical analysis for the proposed initialization and propose a prior-guided regularization method that boosts the smoothness of the loss function. Additionally, we design a prior-guided ensemble FAT method that averages the different model weights of historical models using different decay rates. Our proposed method, called FGSM-PGK, assembles the prior-guided knowledge, i.e., the prior-guided initialization and model weights, acquired during the historical training process. Evaluations of four datasets demonstrate the superiority of the proposed method.



## **12. UNICORN: A Unified Backdoor Trigger Inversion Framework**

cs.LG

**SubmitDate**: 2023-04-05    [abs](http://arxiv.org/abs/2304.02786v1) [paper-pdf](http://arxiv.org/pdf/2304.02786v1)

**Authors**: Zhenting Wang, Kai Mei, Juan Zhai, Shiqing Ma

**Abstract**: The backdoor attack, where the adversary uses inputs stamped with triggers (e.g., a patch) to activate pre-planted malicious behaviors, is a severe threat to Deep Neural Network (DNN) models. Trigger inversion is an effective way of identifying backdoor models and understanding embedded adversarial behaviors. A challenge of trigger inversion is that there are many ways of constructing the trigger. Existing methods cannot generalize to various types of triggers by making certain assumptions or attack-specific constraints. The fundamental reason is that existing work does not consider the trigger's design space in their formulation of the inversion problem. This work formally defines and analyzes the triggers injected in different spaces and the inversion problem. Then, it proposes a unified framework to invert backdoor triggers based on the formalization of triggers and the identified inner behaviors of backdoor models from our analysis. Our prototype UNICORN is general and effective in inverting backdoor triggers in DNNs. The code can be found at https://github.com/RU-System-Software-and-Security/UNICORN.



## **13. Planning for Attacker Entrapment in Adversarial Settings**

cs.AI

**SubmitDate**: 2023-04-05    [abs](http://arxiv.org/abs/2303.00822v2) [paper-pdf](http://arxiv.org/pdf/2303.00822v2)

**Authors**: Brittany Cates, Anagha Kulkarni, Sarath Sreedharan

**Abstract**: In this paper, we propose a planning framework to generate a defense strategy against an attacker who is working in an environment where a defender can operate without the attacker's knowledge. The objective of the defender is to covertly guide the attacker to a trap state from which the attacker cannot achieve their goal. Further, the defender is constrained to achieve its goal within K number of steps, where K is calculated as a pessimistic lower bound within which the attacker is unlikely to suspect a threat in the environment. Such a defense strategy is highly useful in real world systems like honeypots or honeynets, where an unsuspecting attacker interacts with a simulated production system while assuming it is the actual production system. Typically, the interaction between an attacker and a defender is captured using game theoretic frameworks. Our problem formulation allows us to capture it as a much simpler infinite horizon discounted MDP, in which the optimal policy for the MDP gives the defender's strategy against the actions of the attacker. Through empirical evaluation, we show the merits of our problem formulation.



## **14. Domain Generalization with Adversarial Intensity Attack for Medical Image Segmentation**

eess.IV

Code is available upon publication

**SubmitDate**: 2023-04-05    [abs](http://arxiv.org/abs/2304.02720v1) [paper-pdf](http://arxiv.org/pdf/2304.02720v1)

**Authors**: Zheyuan Zhang, Bin Wang, Lanhong Yao, Ugur Demir, Debesh Jha, Ismail Baris Turkbey, Boqing Gong, Ulas Bagci

**Abstract**: Most statistical learning algorithms rely on an over-simplified assumption, that is, the train and test data are independent and identically distributed. In real-world scenarios, however, it is common for models to encounter data from new and different domains to which they were not exposed to during training. This is often the case in medical imaging applications due to differences in acquisition devices, imaging protocols, and patient characteristics. To address this problem, domain generalization (DG) is a promising direction as it enables models to handle data from previously unseen domains by learning domain-invariant features robust to variations across different domains. To this end, we introduce a novel DG method called Adversarial Intensity Attack (AdverIN), which leverages adversarial training to generate training data with an infinite number of styles and increase data diversity while preserving essential content information. We conduct extensive evaluation experiments on various multi-domain segmentation datasets, including 2D retinal fundus optic disc/cup and 3D prostate MRI. Our results demonstrate that AdverIN significantly improves the generalization ability of the segmentation models, achieving significant improvement on these challenging datasets. Code is available upon publication.



## **15. A Certified Radius-Guided Attack Framework to Image Segmentation Models**

cs.CV

Accepted by EuroSP 2023

**SubmitDate**: 2023-04-05    [abs](http://arxiv.org/abs/2304.02693v1) [paper-pdf](http://arxiv.org/pdf/2304.02693v1)

**Authors**: Wenjie Qu, Youqi Li, Binghui Wang

**Abstract**: Image segmentation is an important problem in many safety-critical applications. Recent studies show that modern image segmentation models are vulnerable to adversarial perturbations, while existing attack methods mainly follow the idea of attacking image classification models. We argue that image segmentation and classification have inherent differences, and design an attack framework specially for image segmentation models. Our attack framework is inspired by certified radius, which was originally used by defenders to defend against adversarial perturbations to classification models. We are the first, from the attacker perspective, to leverage the properties of certified radius and propose a certified radius guided attack framework against image segmentation models. Specifically, we first adapt randomized smoothing, the state-of-the-art certification method for classification models, to derive the pixel's certified radius. We then focus more on disrupting pixels with relatively smaller certified radii and design a pixel-wise certified radius guided loss, when plugged into any existing white-box attack, yields our certified radius-guided white-box attack. Next, we propose the first black-box attack to image segmentation models via bandit. We design a novel gradient estimator, based on bandit feedback, which is query-efficient and provably unbiased and stable. We use this gradient estimator to design a projected bandit gradient descent (PBGD) attack, as well as a certified radius-guided PBGD (CR-PBGD) attack. We prove our PBGD and CR-PBGD attacks can achieve asymptotically optimal attack performance with an optimal rate. We evaluate our certified-radius guided white-box and black-box attacks on multiple modern image segmentation models and datasets. Our results validate the effectiveness of our certified radius-guided attack framework.



## **16. Going Further: Flatness at the Rescue of Early Stopping for Adversarial Example Transferability**

cs.LG

**SubmitDate**: 2023-04-05    [abs](http://arxiv.org/abs/2304.02688v1) [paper-pdf](http://arxiv.org/pdf/2304.02688v1)

**Authors**: Martin Gubri, Maxime Cordy, Yves Le Traon

**Abstract**: Transferability is the property of adversarial examples to be misclassified by other models than the surrogate model for which they were crafted. Previous research has shown that transferability is substantially increased when the training of the surrogate model has been early stopped. A common hypothesis to explain this is that the later training epochs are when models learn the non-robust features that adversarial attacks exploit. Hence, an early stopped model is more robust (hence, a better surrogate) than fully trained models. We demonstrate that the reasons why early stopping improves transferability lie in the side effects it has on the learning dynamics of the model. We first show that early stopping benefits transferability even on models learning from data with non-robust features. We then establish links between transferability and the exploration of the loss landscape in the parameter space, on which early stopping has an inherent effect. More precisely, we observe that transferability peaks when the learning rate decays, which is also the time at which the sharpness of the loss significantly drops. This leads us to propose RFN, a new approach for transferability that minimizes loss sharpness during training in order to maximize transferability. We show that by searching for large flat neighborhoods, RFN always improves over early stopping (by up to 47 points of transferability rate) and is competitive to (if not better than) strong state-of-the-art baselines.



## **17. Adversarial robustness of VAEs through the lens of local geometry**

cs.LG

International Conference on Artificial Intelligence and Statistics  (AISTATS) 2023

**SubmitDate**: 2023-04-05    [abs](http://arxiv.org/abs/2208.03923v2) [paper-pdf](http://arxiv.org/pdf/2208.03923v2)

**Authors**: Asif Khan, Amos Storkey

**Abstract**: In an unsupervised attack on variational autoencoders (VAEs), an adversary finds a small perturbation in an input sample that significantly changes its latent space encoding, thereby compromising the reconstruction for a fixed decoder. A known reason for such vulnerability is the distortions in the latent space resulting from a mismatch between approximated latent posterior and a prior distribution. Consequently, a slight change in an input sample can move its encoding to a low/zero density region in the latent space resulting in an unconstrained generation. This paper demonstrates that an optimal way for an adversary to attack VAEs is to exploit a directional bias of a stochastic pullback metric tensor induced by the encoder and decoder networks. The pullback metric tensor of an encoder measures the change in infinitesimal latent volume from an input to a latent space. Thus, it can be viewed as a lens to analyse the effect of input perturbations leading to latent space distortions. We propose robustness evaluation scores using the eigenspectrum of a pullback metric tensor. Moreover, we empirically show that the scores correlate with the robustness parameter $\beta$ of the $\beta-$VAE. Since increasing $\beta$ also degrades reconstruction quality, we demonstrate a simple alternative using \textit{mixup} training to fill the empty regions in the latent space, thus improving robustness with improved reconstruction.



## **18. Existence and Minimax Theorems for Adversarial Surrogate Risks in Binary Classification**

cs.LG

37 pages

**SubmitDate**: 2023-04-05    [abs](http://arxiv.org/abs/2206.09098v2) [paper-pdf](http://arxiv.org/pdf/2206.09098v2)

**Authors**: Natalie S. Frank Jonathan Niles-Weed

**Abstract**: Adversarial training is one of the most popular methods for training methods robust to adversarial attacks, however, it is not well-understood from a theoretical perspective. We prove and existence, regularity, and minimax theorems for adversarial surrogate risks. Our results explain some empirical observations on adversarial robustness from prior work and suggest new directions in algorithm development. Furthermore, our results extend previously known existence and minimax theorems for the adversarial classification risk to surrogate risks.



## **19. How to choose your best allies for a transferable attack?**

cs.CR

**SubmitDate**: 2023-04-05    [abs](http://arxiv.org/abs/2304.02312v1) [paper-pdf](http://arxiv.org/pdf/2304.02312v1)

**Authors**: Thibault Maho, Seyed-Mohsen Moosavi-Dezfooli, Teddy Furon

**Abstract**: The transferability of adversarial examples is a key issue in the security of deep neural networks. The possibility of an adversarial example crafted for a source model fooling another targeted model makes the threat of adversarial attacks more realistic. Measuring transferability is a crucial problem, but the Attack Success Rate alone does not provide a sound evaluation. This paper proposes a new methodology for evaluating transferability by putting distortion in a central position. This new tool shows that transferable attacks may perform far worse than a black box attack if the attacker randomly picks the source model. To address this issue, we propose a new selection mechanism, called FiT, which aims at choosing the best source model with only a few preliminary queries to the target. Our experimental results show that FiT is highly effective at selecting the best source model for multiple scenarios such as single-model attacks, ensemble-model attacks and multiple attacks (Code available at: https://github.com/t-maho/transferability_measure_fit).



## **20. PatchCensor: Patch Robustness Certification for Transformers via Exhaustive Testing**

cs.CV

This paper has been accepted by ACM Transactions on Software  Engineering and Methodology (TOSEM'23) in "Continuous Special Section: AI and  SE." Please include TOSEM for any citations

**SubmitDate**: 2023-04-05    [abs](http://arxiv.org/abs/2111.10481v2) [paper-pdf](http://arxiv.org/pdf/2111.10481v2)

**Authors**: Yuheng Huang, Lei Ma, Yuanchun Li

**Abstract**: Vision Transformer (ViT) is known to be highly nonlinear like other classical neural networks and could be easily fooled by both natural and adversarial patch perturbations. This limitation could pose a threat to the deployment of ViT in the real industrial environment, especially in safety-critical scenarios. In this work, we propose PatchCensor, aiming to certify the patch robustness of ViT by applying exhaustive testing. We try to provide a provable guarantee by considering the worst patch attack scenarios. Unlike empirical defenses against adversarial patches that may be adaptively breached, certified robust approaches can provide a certified accuracy against arbitrary attacks under certain conditions. However, existing robustness certifications are mostly based on robust training, which often requires substantial training efforts and the sacrifice of model performance on normal samples. To bridge the gap, PatchCensor seeks to improve the robustness of the whole system by detecting abnormal inputs instead of training a robust model and asking it to give reliable results for every input, which may inevitably compromise accuracy. Specifically, each input is tested by voting over multiple inferences with different mutated attention masks, where at least one inference is guaranteed to exclude the abnormal patch. This can be seen as complete-coverage testing, which could provide a statistical guarantee on inference at the test time. Our comprehensive evaluation demonstrates that PatchCensor is able to achieve high certified accuracy (e.g. 67.1% on ImageNet for 2%-pixel adversarial patches), significantly outperforming state-of-the-art techniques while achieving similar clean accuracy (81.8% on ImageNet). Meanwhile, our technique also supports flexible configurations to handle different adversarial patch sizes (up to 25%) by simply changing the masking strategy.



## **21. Dynamic Adversarial Resource Allocation: the dDAB Game**

cs.MA

**SubmitDate**: 2023-04-05    [abs](http://arxiv.org/abs/2304.02172v1) [paper-pdf](http://arxiv.org/pdf/2304.02172v1)

**Authors**: Daigo Shishika, Yue Guan, Jason R. Marden, Michael Dorothy, Panagiotis Tsiotras, Vijay Kumar

**Abstract**: This work proposes a dynamic and adversarial resource allocation problem in a graph environment, which is referred to as the dynamic Defender-Attacker Blotto (dDAB) game. A team of defender robots is tasked to ensure numerical advantage at every node in the graph against a team of attacker robots. The engagement is formulated as a discrete-time dynamic game, where the two teams reallocate their robots in sequence and each robot can move at most one hop at each time step. The game terminates with the attacker's victory if any node has more attacker robots than defender robots. Our goal is to identify the necessary and sufficient number of defender robots to guarantee defense. Through a reachability analysis, we first solve the problem for the case where the attacker team stays as a single group. The results are then generalized to the case where the attacker team can freely split and merge into subteams. Crucially, our analysis indicates that there is no incentive for the attacker team to split, which significantly reduces the search space for the attacker's winning strategies and also enables us to design defender counter-strategies using superposition. We also present an efficient numerical algorithm to identify the necessary and sufficient number of defender robots to defend a given graph. Finally, we present illustrative examples to verify the efficacy of the proposed framework.



## **22. Do we need entire training data for adversarial training?**

cs.CV

6 pages, 4 figures

**SubmitDate**: 2023-04-05    [abs](http://arxiv.org/abs/2303.06241v2) [paper-pdf](http://arxiv.org/pdf/2303.06241v2)

**Authors**: Vipul Gupta, Apurva Narayan

**Abstract**: Deep Neural Networks (DNNs) are being used to solve a wide range of problems in many domains including safety-critical domains like self-driving cars and medical imagery. DNNs suffer from vulnerability against adversarial attacks. In the past few years, numerous approaches have been proposed to tackle this problem by training networks using adversarial training. Almost all the approaches generate adversarial examples for the entire training dataset, thus increasing the training time drastically. We show that we can decrease the training time for any adversarial training algorithm by using only a subset of training data for adversarial training. To select the subset, we filter the adversarially-prone samples from the training data. We perform a simple adversarial attack on all training examples to filter this subset. In this attack, we add a small perturbation to each pixel and a few grid lines to the input image.   We perform adversarial training on the adversarially-prone subset and mix it with vanilla training performed on the entire dataset. Our results show that when our method-agnostic approach is plugged into FGSM, we achieve a speedup of 3.52x on MNIST and 1.98x on the CIFAR-10 dataset with comparable robust accuracy. We also test our approach on state-of-the-art Free adversarial training and achieve a speedup of 1.2x in training time with a marginal drop in robust accuracy on the ImageNet dataset.



## **23. Boosting Adversarial Transferability using Dynamic Cues**

cs.CV

International Conference on Learning Representations (ICLR'23),  Code:https://bit.ly/3Xd9gRQ

**SubmitDate**: 2023-04-04    [abs](http://arxiv.org/abs/2302.12252v2) [paper-pdf](http://arxiv.org/pdf/2302.12252v2)

**Authors**: Muzammal Naseer, Ahmad Mahmood, Salman Khan, Fahad Khan

**Abstract**: The transferability of adversarial perturbations between image models has been extensively studied. In this case, an attack is generated from a known surrogate \eg, the ImageNet trained model, and transferred to change the decision of an unknown (black-box) model trained on an image dataset. However, attacks generated from image models do not capture the dynamic nature of a moving object or a changing scene due to a lack of temporal cues within image models. This leads to reduced transferability of adversarial attacks from representation-enriched \emph{image} models such as Supervised Vision Transformers (ViTs), Self-supervised ViTs (\eg, DINO), and Vision-language models (\eg, CLIP) to black-box \emph{video} models. In this work, we induce dynamic cues within the image models without sacrificing their original performance on images. To this end, we optimize \emph{temporal prompts} through frozen image models to capture motion dynamics. Our temporal prompts are the result of a learnable transformation that allows optimizing for temporal gradients during an adversarial attack to fool the motion dynamics. Specifically, we introduce spatial (image) and temporal (video) cues within the same source model through task-specific prompts. Attacking such prompts maximizes the adversarial transferability from image-to-video and image-to-image models using the attacks designed for image models. Our attack results indicate that the attacker does not need specialized architectures, \eg, divided space-time attention, 3D convolutions, or multi-view convolution networks for different data modalities. Image models are effective surrogates to optimize an adversarial attack to fool black-box models in a changing environment over time. Code is available at https://bit.ly/3Xd9gRQ



## **24. Risk-based Security Measure Allocation Against Injection Attacks on Actuators**

eess.SY

Submitted to IEEE Open Journal of Control Systems (OJ-CSYS)

**SubmitDate**: 2023-04-04    [abs](http://arxiv.org/abs/2304.02055v1) [paper-pdf](http://arxiv.org/pdf/2304.02055v1)

**Authors**: Sribalaji C. Anand, André M. H. Teixeira

**Abstract**: This article considers the problem of risk-optimal allocation of security measures when the actuators of an uncertain control system are under attack. We consider an adversary injecting false data into the actuator channels. The attack impact is characterized by the maximum performance loss caused by a stealthy adversary with bounded energy. Since the impact is a random variable, due to system uncertainty, we use Conditional Value-at-Risk (CVaR) to characterize the risk associated with the attack. We then consider the problem of allocating the security measures which minimize the risk. We assume that there are only a limited number of security measures available. Under this constraint, we observe that the allocation problem is a mixed-integer optimization problem. Thus we use relaxation techniques to approximate the security allocation problem into a Semi-Definite Program (SDP). We also compare our allocation method $(i)$ across different risk measures: the worst-case measure, the average (nominal) measure, and $(ii)$ across different search algorithms: the exhaustive and the greedy search algorithms. We depict the efficacy of our approach through numerical examples.



## **25. EGC: Image Generation and Classification via a Single Energy-Based Model**

cs.CV

Technical report

**SubmitDate**: 2023-04-04    [abs](http://arxiv.org/abs/2304.02012v1) [paper-pdf](http://arxiv.org/pdf/2304.02012v1)

**Authors**: Qiushan Guo, Chuofan Ma, Yi Jiang, Zehuan Yuan, Yizhou Yu, Ping Luo

**Abstract**: Learning image classification and image generation using the same set of network parameters is a challenging problem. Recent advanced approaches perform well in one task often exhibit poor performance in the other. This work introduces an energy-based classifier and generator, namely EGC, which can achieve superior performance in both tasks using a single neural network. Unlike a conventional classifier that outputs a label given an image (i.e., a conditional distribution $p(y|\mathbf{x})$), the forward pass in EGC is a classifier that outputs a joint distribution $p(\mathbf{x},y)$, enabling an image generator in its backward pass by marginalizing out the label $y$. This is done by estimating the energy and classification probability given a noisy image in the forward pass, while denoising it using the score function estimated in the backward pass. EGC achieves competitive generation results compared with state-of-the-art approaches on ImageNet-1k, CelebA-HQ and LSUN Church, while achieving superior classification accuracy and robustness against adversarial attacks on CIFAR-10. This work represents the first successful attempt to simultaneously excel in both tasks using a single set of network parameters. We believe that EGC bridges the gap between discriminative and generative learning.



## **26. Cross-Class Feature Augmentation for Class Incremental Learning**

cs.CV

**SubmitDate**: 2023-04-04    [abs](http://arxiv.org/abs/2304.01899v1) [paper-pdf](http://arxiv.org/pdf/2304.01899v1)

**Authors**: Taehoon Kim, Jaeyoo Park, Bohyung Han

**Abstract**: We propose a novel class incremental learning approach by incorporating a feature augmentation technique motivated by adversarial attacks. We employ a classifier learned in the past to complement training examples rather than simply play a role as a teacher for knowledge distillation towards subsequent models. The proposed approach has a unique perspective to utilize the previous knowledge in class incremental learning since it augments features of arbitrary target classes using examples in other classes via adversarial attacks on a previously learned classifier. By allowing the cross-class feature augmentations, each class in the old tasks conveniently populates samples in the feature space, which alleviates the collapse of the decision boundaries caused by sample deficiency for the previous tasks, especially when the number of stored exemplars is small. This idea can be easily incorporated into existing class incremental learning algorithms without any architecture modification. Extensive experiments on the standard benchmarks show that our method consistently outperforms existing class incremental learning methods by significant margins in various scenarios, especially under an environment with an extremely limited memory budget.



## **27. Adversarial Detection: Attacking Object Detection in Real Time**

cs.AI

Accepted by IEEE Intelligent Vehicle Symposium, 2023

**SubmitDate**: 2023-04-04    [abs](http://arxiv.org/abs/2209.01962v4) [paper-pdf](http://arxiv.org/pdf/2209.01962v4)

**Authors**: Han Wu, Syed Yunas, Sareh Rowlands, Wenjie Ruan, Johan Wahlstrom

**Abstract**: Intelligent robots rely on object detection models to perceive the environment. Following advances in deep learning security it has been revealed that object detection models are vulnerable to adversarial attacks. However, prior research primarily focuses on attacking static images or offline videos. Therefore, it is still unclear if such attacks could jeopardize real-world robotic applications in dynamic environments. This paper bridges this gap by presenting the first real-time online attack against object detection models. We devise three attacks that fabricate bounding boxes for nonexistent objects at desired locations. The attacks achieve a success rate of about 90\% within about 20 iterations. The demo video is available at https://youtu.be/zJZ1aNlXsMU.



## **28. Adversarial Driving: Attacking End-to-End Autonomous Driving**

cs.CV

Accepted by IEEE Intelligent Vehicle Symposium, 2023

**SubmitDate**: 2023-04-04    [abs](http://arxiv.org/abs/2103.09151v6) [paper-pdf](http://arxiv.org/pdf/2103.09151v6)

**Authors**: Han Wu, Syed Yunas, Sareh Rowlands, Wenjie Ruan, Johan Wahlstrom

**Abstract**: As research in deep neural networks advances, deep convolutional networks become promising for autonomous driving tasks. In particular, there is an emerging trend of employing end-to-end neural network models for autonomous driving. However, previous research has shown that deep neural network classifiers are vulnerable to adversarial attacks. While for regression tasks, the effect of adversarial attacks is not as well understood. In this research, we devise two white-box targeted attacks against end-to-end autonomous driving models. Our attacks manipulate the behavior of the autonomous driving system by perturbing the input image. In an average of 800 attacks with the same attack strength (epsilon=1), the image-specific and image-agnostic attack deviates the steering angle from the original output by 0.478 and 0.111, respectively, which is much stronger than random noises that only perturbs the steering angle by 0.002 (The steering angle ranges from [-1, 1]). Both attacks can be initiated in real-time on CPUs without employing GPUs. Demo video: https://youtu.be/I0i8uN2oOP0.



## **29. MENLI: Robust Evaluation Metrics from Natural Language Inference**

cs.CL

TACL 2023 Camera-ready github link fixed

**SubmitDate**: 2023-04-04    [abs](http://arxiv.org/abs/2208.07316v3) [paper-pdf](http://arxiv.org/pdf/2208.07316v3)

**Authors**: Yanran Chen, Steffen Eger

**Abstract**: Recently proposed BERT-based evaluation metrics for text generation perform well on standard benchmarks but are vulnerable to adversarial attacks, e.g., relating to information correctness. We argue that this stems (in part) from the fact that they are models of semantic similarity. In contrast, we develop evaluation metrics based on Natural Language Inference (NLI), which we deem a more appropriate modeling. We design a preference-based adversarial attack framework and show that our NLI based metrics are much more robust to the attacks than the recent BERT-based metrics. On standard benchmarks, our NLI based metrics outperform existing summarization metrics, but perform below SOTA MT metrics. However, when combining existing metrics with our NLI metrics, we obtain both higher adversarial robustness (15%-30%) and higher quality metrics as measured on standard benchmarks (+5% to 30%).



## **30. Tracklet-Switch Adversarial Attack against Pedestrian Multi-Object Tracking Trackers**

cs.CV

**SubmitDate**: 2023-04-04    [abs](http://arxiv.org/abs/2111.08954v3) [paper-pdf](http://arxiv.org/pdf/2111.08954v3)

**Authors**: Delv Lin, Qi Chen, Chengyu Zhou, Kun He

**Abstract**: Multi-Object Tracking (MOT) has achieved aggressive progress and derived many excellent deep learning trackers. Meanwhile, most deep learning models are known to be vulnerable to adversarial examples that are crafted with small perturbations but could mislead the model prediction. In this work, we observe that the robustness on the MOT trackers is rarely studied, and it is challenging to attack the MOT system since its mature association algorithms are designed to be robust against errors during the tracking. To this end, we analyze the vulnerability of popular MOT trackers and propose a novel adversarial attack method called Tracklet-Switch (TraSw) against the complete tracking pipeline of MOT. The proposed TraSw can fool the advanced deep pedestrian trackers (i.e., FairMOT and ByteTrack), causing them fail to track the targets in the subsequent frames by perturbing very few frames. Experiments on the MOT-Challenge datasets (i.e., 2DMOT15, MOT17, and MOT20) show that TraSw can achieve an extraordinarily high success attack rate of over 95% by attacking only four frames on average. To our knowledge, this is the first work on the adversarial attack against the pedestrian MOT trackers. Code is available at https://github.com/JHL-HUST/TraSw .



## **31. On the Feasibility of Specialized Ability Extracting for Large Language Code Models**

cs.SE

11 pages

**SubmitDate**: 2023-04-04    [abs](http://arxiv.org/abs/2303.03012v3) [paper-pdf](http://arxiv.org/pdf/2303.03012v3)

**Authors**: Zongjie Li, Chaozheng Wang, Pingchuan Ma, Chaowei Liu, Shuai Wang, Daoyuan Wu, Cuiyun Gao

**Abstract**: Recent progress in large language code models (LLCMs) has led to a dramatic surge in the use of software development. Nevertheless, it is widely known that training a well-performed LLCM requires a plethora of workforce for collecting the data and high quality annotation. Additionally, the training dataset may be proprietary (or partially open source to the public), and the training process is often conducted on a large-scale cluster of GPUs with high costs. Inspired by the recent success of imitation attacks in extracting computer vision and natural language models, this work launches the first imitation attack on LLCMs: by querying a target LLCM with carefully-designed queries and collecting the outputs, the adversary can train an imitation model that manifests close behavior with the target LLCM. We systematically investigate the effectiveness of launching imitation attacks under different query schemes and different LLCM tasks. We also design novel methods to polish the LLCM outputs, resulting in an effective imitation training process. We summarize our findings and provide lessons harvested in this study that can help better depict the attack surface of LLCMs. Our research contributes to the growing body of knowledge on imitation attacks and defenses in deep neural models, particularly in the domain of code related tasks.



## **32. Defending Against Patch-based Backdoor Attacks on Self-Supervised Learning**

cs.CV

Accepted to CVPR 2023

**SubmitDate**: 2023-04-04    [abs](http://arxiv.org/abs/2304.01482v1) [paper-pdf](http://arxiv.org/pdf/2304.01482v1)

**Authors**: Ajinkya Tejankar, Maziar Sanjabi, Qifan Wang, Sinong Wang, Hamed Firooz, Hamed Pirsiavash, Liang Tan

**Abstract**: Recently, self-supervised learning (SSL) was shown to be vulnerable to patch-based data poisoning backdoor attacks. It was shown that an adversary can poison a small part of the unlabeled data so that when a victim trains an SSL model on it, the final model will have a backdoor that the adversary can exploit. This work aims to defend self-supervised learning against such attacks. We use a three-step defense pipeline, where we first train a model on the poisoned data. In the second step, our proposed defense algorithm (PatchSearch) uses the trained model to search the training data for poisoned samples and removes them from the training set. In the third step, a final model is trained on the cleaned-up training set. Our results show that PatchSearch is an effective defense. As an example, it improves a model's accuracy on images containing the trigger from 38.2% to 63.7% which is very close to the clean model's accuracy, 64.6%. Moreover, we show that PatchSearch outperforms baselines and state-of-the-art defense approaches including those using additional clean, trusted data. Our code is available at https://github.com/UCDvision/PatchSearch



## **33. NetFlick: Adversarial Flickering Attacks on Deep Learning Based Video Compression**

eess.IV

8 pages; Accepted to ICLR 2023 ML4IoT workshop

**SubmitDate**: 2023-04-04    [abs](http://arxiv.org/abs/2304.01441v1) [paper-pdf](http://arxiv.org/pdf/2304.01441v1)

**Authors**: Jung-Woo Chang, Nojan Sheybani, Shehzeen Samarah Hussain, Mojan Javaheripi, Seira Hidano, Farinaz Koushanfar

**Abstract**: Video compression plays a significant role in IoT devices for the efficient transport of visual data while satisfying all underlying bandwidth constraints. Deep learning-based video compression methods are rapidly replacing traditional algorithms and providing state-of-the-art results on edge devices. However, recently developed adversarial attacks demonstrate that digitally crafted perturbations can break the Rate-Distortion relationship of video compression. In this work, we present a real-world LED attack to target video compression frameworks. Our physically realizable attack, dubbed NetFlick, can degrade the spatio-temporal correlation between successive frames by injecting flickering temporal perturbations. In addition, we propose universal perturbations that can downgrade performance of incoming video without prior knowledge of the contents. Experimental results demonstrate that NetFlick can successfully deteriorate the performance of video compression frameworks in both digital- and physical-settings and can be further extended to attack downstream video classification networks.



## **34. Is Stochastic Mirror Descent Vulnerable to Adversarial Delay Attacks? A Traffic Assignment Resilience Study**

cs.LG

Preprint under review

**SubmitDate**: 2023-04-03    [abs](http://arxiv.org/abs/2304.01161v1) [paper-pdf](http://arxiv.org/pdf/2304.01161v1)

**Authors**: Yunian Pan, Tao Li, Quanyan Zhu

**Abstract**: \textit{Intelligent Navigation Systems} (INS) are exposed to an increasing number of informational attack vectors, which often intercept through the communication channels between the INS and the transportation network during the data collecting process. To measure the resilience of INS, we use the concept of a Wardrop Non-Equilibrium Solution (WANES), which is characterized by the probabilistic outcome of learning within a bounded number of interactions. By using concentration arguments, we have discovered that any bounded feedback delaying attack only degrades the systematic performance up to order $\tilde{\mathcal{O}}(\sqrt{{d^3}{T^{-1}}})$ along the traffic flow trajectory within the Delayed Mirror Descent (DMD) online-learning framework. This degradation in performance can occur with only mild assumptions imposed. Our result implies that learning-based INS infrastructures can achieve Wardrop Non-equilibrium even when experiencing a certain period of disruption in the information structure. These findings provide valuable insights for designing defense mechanisms against possible jamming attacks across different layers of the transportation ecosystem.



## **35. Learning About Simulated Adversaries from Human Defenders using Interactive Cyber-Defense Games**

cs.CR

Submitted to Journal of Cybersecurity

**SubmitDate**: 2023-04-03    [abs](http://arxiv.org/abs/2304.01142v1) [paper-pdf](http://arxiv.org/pdf/2304.01142v1)

**Authors**: Baptiste Prebot, Yinuo Du, Cleotilde Gonzalez

**Abstract**: Given the increase in cybercrime, cybersecurity analysts (i.e. Defenders) are in high demand. Defenders must monitor an organization's network to evaluate threats and potential breaches into the network. Adversary simulation is commonly used to test defenders' performance against known threats to organizations. However, it is unclear how effective this training process is in preparing defenders for this highly demanding job. In this paper, we demonstrate how to use adversarial algorithms to investigate defenders' learning of defense strategies, using interactive cyber defense games. Our Interactive Defense Game (IDG) represents a cyber defense scenario that requires constant monitoring of incoming network alerts and allows a defender to analyze, remove, and restore services based on the events observed in a network. The participants in our study faced one of two types of simulated adversaries. A Beeline adversary is a fast, targeted, and informed attacker; and a Meander adversary is a slow attacker that wanders the network until it finds the right target to exploit. Our results suggest that although human defenders have more difficulty to stop the Beeline adversary initially, they were able to learn to stop this adversary by taking advantage of their attack strategy. Participants who played against the Beeline adversary learned to anticipate the adversary and take more proactive actions, while decreasing their reactive actions. These findings have implications for understanding how to help cybersecurity analysts speed up their training.



## **36. A Pilot Study of Query-Free Adversarial Attack against Stable Diffusion**

cs.CV

The 3rd Workshop of Adversarial Machine Learning on Computer Vision:  Art of Robustness

**SubmitDate**: 2023-04-03    [abs](http://arxiv.org/abs/2303.16378v2) [paper-pdf](http://arxiv.org/pdf/2303.16378v2)

**Authors**: Haomin Zhuang, Yihua Zhang, Sijia Liu

**Abstract**: Despite the record-breaking performance in Text-to-Image (T2I) generation by Stable Diffusion, less research attention is paid to its adversarial robustness. In this work, we study the problem of adversarial attack generation for Stable Diffusion and ask if an adversarial text prompt can be obtained even in the absence of end-to-end model queries. We call the resulting problem 'query-free attack generation'. To resolve this problem, we show that the vulnerability of T2I models is rooted in the lack of robustness of text encoders, e.g., the CLIP text encoder used for attacking Stable Diffusion. Based on such insight, we propose both untargeted and targeted query-free attacks, where the former is built on the most influential dimensions in the text embedding space, which we call steerable key dimensions. By leveraging the proposed attacks, we empirically show that only a five-character perturbation to the text prompt is able to cause the significant content shift of synthesized images using Stable Diffusion. Moreover, we show that the proposed target attack can precisely steer the diffusion model to scrub the targeted image content without causing much change in untargeted image content. Our code is available at https://github.com/OPTML-Group/QF-Attack.



## **37. Improving RF-DNA Fingerprinting Performance in an Indoor Multipath Environment Using Semi-Supervised Learning**

eess.SP

16 pages, 14 figures. Submitted to IEEE Transactions on Information  Forensics & Security

**SubmitDate**: 2023-04-02    [abs](http://arxiv.org/abs/2304.00648v1) [paper-pdf](http://arxiv.org/pdf/2304.00648v1)

**Authors**: Mohamed k. Fadul, Donald R. Reising, Lakmali P. Weerasena, T. Daniel Loveless, Mina Sartipi

**Abstract**: The number of Internet of Things (IoT) deployments is expected to reach 75.4 billion by 2025. Roughly 70% of all IoT devices employ weak or no encryption; thus, putting them and their connected infrastructure at risk of attack by devices that are wrongly authenticated or not authenticated at all. A physical layer security approach -- known as Specific Emitter Identification (SEI) -- has been proposed and is being pursued as a viable IoT security mechanism. SEI is advantageous because it is a passive technique that exploits inherent and distinct features that are unintentionally added to the signal by the IoT Radio Frequency (RF) front-end. SEI's passive exploitation of unintentional signal features removes any need to modify the IoT device, which makes it ideal for existing and future IoT deployments. Despite the amount of SEI research conducted, some challenges must be addressed to make SEI a viable IoT security approach. One challenge is the extraction of SEI features from signals collected under multipath fading conditions. Multipath corrupts the inherent SEI features that are used to discriminate one IoT device from another; thus, degrading authentication performance and increasing the chance of attack. This work presents two semi-supervised Deep Learning (DL) equalization approaches and compares their performance with the current state of the art. The two approaches are the Conditional Generative Adversarial Network (CGAN) and Joint Convolutional Auto-Encoder and Convolutional Neural Network (JCAECNN). Both approaches learn the channel distribution to enable multipath correction while simultaneously preserving the SEI exploited features. CGAN and JCAECNN performance is assessed using a Rayleigh fading channel under degrading SNR, up to thirty-two IoT devices, and two publicly available signal sets. The JCAECNN improves SEI performance by 10% beyond that of the current state of the art.



## **38. Test-time Detection and Repair of Adversarial Samples via Masked Autoencoder**

cs.CV

**SubmitDate**: 2023-04-02    [abs](http://arxiv.org/abs/2303.12848v3) [paper-pdf](http://arxiv.org/pdf/2303.12848v3)

**Authors**: Yun-Yun Tsai, Ju-Chin Chao, Albert Wen, Zhaoyuan Yang, Chengzhi Mao, Tapan Shah, Junfeng Yang

**Abstract**: Training-time defenses, known as adversarial training, incur high training costs and do not generalize to unseen attacks. Test-time defenses solve these issues but most existing test-time defenses require adapting the model weights, therefore they do not work on frozen models and complicate model memory management. The only test-time defense that does not adapt model weights aims to adapt the input with self-supervision tasks. However, we empirically found these self-supervision tasks are not sensitive enough to detect adversarial attacks accurately. In this paper, we propose DRAM, a novel defense method to detect and repair adversarial samples at test time via Masked autoencoder (MAE). We demonstrate how to use MAE losses to build a Kolmogorov-Smirnov test to detect adversarial samples. Moreover, we use the MAE losses to calculate input reversal vectors that repair adversarial samples resulting from previously unseen attacks. Results on large-scale ImageNet dataset show that, compared to all detection baselines evaluated, DRAM achieves the best detection rate (82% on average) on all eight adversarial attacks evaluated. For attack repair, DRAM improves the robust accuracy by 6% ~ 41% for standard ResNet50 and 3% ~ 8% for robust ResNet50 compared with the baselines that use contrastive learning and rotation prediction.



## **39. FACM: Intermediate Layer Still Retain Effective Features against Adversarial Examples**

cs.CV

**SubmitDate**: 2023-04-02    [abs](http://arxiv.org/abs/2206.00924v2) [paper-pdf](http://arxiv.org/pdf/2206.00924v2)

**Authors**: Xiangyuan Yang, Jie Lin, Hanlin Zhang, Xinyu Yang, Peng Zhao

**Abstract**: In strong adversarial attacks against deep neural networks (DNN), the generated adversarial example will mislead the DNN-implemented classifier by destroying the output features of the last layer. To enhance the robustness of the classifier, in our paper, a \textbf{F}eature \textbf{A}nalysis and \textbf{C}onditional \textbf{M}atching prediction distribution (FACM) model is proposed to utilize the features of intermediate layers to correct the classification. Specifically, we first prove that the intermediate layers of the classifier can still retain effective features for the original category, which is defined as the correction property in our paper. According to this, we propose the FACM model consisting of \textbf{F}eature \textbf{A}nalysis (FA) correction module, \textbf{C}onditional \textbf{M}atching \textbf{P}rediction \textbf{D}istribution (CMPD) correction module and decision module. The FA correction module is the fully connected layers constructed with the output of the intermediate layers as the input to correct the classification of the classifier. The CMPD correction module is a conditional auto-encoder, which can not only use the output of intermediate layers as the condition to accelerate convergence but also mitigate the negative effect of adversarial example training with the Kullback-Leibler loss to match prediction distribution. Through the empirically verified diversity property, the correction modules can be implemented synergistically to reduce the adversarial subspace. Hence, the decision module is proposed to integrate the correction modules to enhance the DNN classifier's robustness. Specially, our model can be achieved by fine-tuning and can be combined with other model-specific defenses.



## **40. Adversarial Training of Self-supervised Monocular Depth Estimation against Physical-World Attacks**

cs.CV

Initially accepted at ICLR2023 (Spotlight)

**SubmitDate**: 2023-04-02    [abs](http://arxiv.org/abs/2301.13487v3) [paper-pdf](http://arxiv.org/pdf/2301.13487v3)

**Authors**: Zhiyuan Cheng, James Liang, Guanhong Tao, Dongfang Liu, Xiangyu Zhang

**Abstract**: Monocular Depth Estimation (MDE) is a critical component in applications such as autonomous driving. There are various attacks against MDE networks. These attacks, especially the physical ones, pose a great threat to the security of such systems. Traditional adversarial training method requires ground-truth labels hence cannot be directly applied to self-supervised MDE that does not have ground-truth depth. Some self-supervised model hardening techniques (e.g., contrastive learning) ignore the domain knowledge of MDE and can hardly achieve optimal performance. In this work, we propose a novel adversarial training method for self-supervised MDE models based on view synthesis without using ground-truth depth. We improve adversarial robustness against physical-world attacks using L0-norm-bounded perturbation in training. We compare our method with supervised learning based and contrastive learning based methods that are tailored for MDE. Results on two representative MDE networks show that we achieve better robustness against various adversarial attacks with nearly no benign performance degradation.



## **41. Instance-level Trojan Attacks on Visual Question Answering via Adversarial Learning in Neuron Activation Space**

cs.CV

**SubmitDate**: 2023-04-02    [abs](http://arxiv.org/abs/2304.00436v1) [paper-pdf](http://arxiv.org/pdf/2304.00436v1)

**Authors**: Yuwei Sun, Hideya Ochiai, Jun Sakuma

**Abstract**: Malicious perturbations embedded in input data, known as Trojan attacks, can cause neural networks to misbehave. However, the impact of a Trojan attack is reduced during fine-tuning of the model, which involves transferring knowledge from a pretrained large-scale model like visual question answering (VQA) to the target model. To mitigate the effects of a Trojan attack, replacing and fine-tuning multiple layers of the pretrained model is possible. This research focuses on sample efficiency, stealthiness and variation, and robustness to model fine-tuning. To address these challenges, we propose an instance-level Trojan attack that generates diverse Trojans across input samples and modalities. Adversarial learning establishes a correlation between a specified perturbation layer and the misbehavior of the fine-tuned model. We conducted extensive experiments on the VQA-v2 dataset using a range of metrics. The results show that our proposed method can effectively adapt to a fine-tuned model with minimal samples. Specifically, we found that a model with a single fine-tuning layer can be compromised using a single shot of adversarial samples, while a model with more fine-tuning layers can be compromised using only a few shots.



## **42. Coordinated Defense Allocation in Reach-Avoid Scenarios with Efficient Online Optimization**

cs.RO

**SubmitDate**: 2023-04-01    [abs](http://arxiv.org/abs/2304.00234v1) [paper-pdf](http://arxiv.org/pdf/2304.00234v1)

**Authors**: Junwei Liu, Zikai Ouyang, Jiahui Yang, Hua Chen, Haibo Lu, Wei Zhang

**Abstract**: Deriving strategies for multiple agents under adversarial scenarios poses a significant challenge in attaining both optimality and efficiency. In this paper, we propose an efficient defense strategy for cooperative defense against a group of attackers in a convex environment. The defenders aim to minimize the total number of attackers that successfully enter the target set without prior knowledge of the attacker's strategy. Our approach involves a two-scale method that decomposes the problem into coordination against a single attacker and assigning defenders to attackers. We first develop a coordination strategy for multiple defenders against a single attacker, implementing online convex programming. This results in the maximum defense-winning region of initial joint states from which the defender can successfully defend against a single attacker. We then propose an allocation algorithm that significantly reduces computational effort required to solve the induced integer linear programming problem. The allocation guarantees defense performance enhancement as the game progresses. We perform various simulations to verify the efficiency of our algorithm compared to the state-of-the-art approaches, including the one using the Gazabo platform with Robot Operating System.



## **43. Proximal Splitting Adversarial Attacks for Semantic Segmentation**

cs.LG

CVPR 2023. Code available at:  https://github.com/jeromerony/alma_prox_segmentation

**SubmitDate**: 2023-03-31    [abs](http://arxiv.org/abs/2206.07179v2) [paper-pdf](http://arxiv.org/pdf/2206.07179v2)

**Authors**: Jérôme Rony, Jean-Christophe Pesquet, Ismail Ben Ayed

**Abstract**: Classification has been the focal point of research on adversarial attacks, but only a few works investigate methods suited to denser prediction tasks, such as semantic segmentation. The methods proposed in these works do not accurately solve the adversarial segmentation problem and, therefore, overestimate the size of the perturbations required to fool models. Here, we propose a white-box attack for these models based on a proximal splitting to produce adversarial perturbations with much smaller $\ell_\infty$ norms. Our attack can handle large numbers of constraints within a nonconvex minimization framework via an Augmented Lagrangian approach, coupled with adaptive constraint scaling and masking strategies. We demonstrate that our attack significantly outperforms previously proposed ones, as well as classification attacks that we adapted for segmentation, providing a first comprehensive benchmark for this dense task.



## **44. Fides: A Generative Framework for Result Validation of Outsourced Machine Learning Workloads via TEE**

cs.CR

**SubmitDate**: 2023-03-31    [abs](http://arxiv.org/abs/2304.00083v1) [paper-pdf](http://arxiv.org/pdf/2304.00083v1)

**Authors**: Abhinav Kumar, Miguel A. Guirao Aguilera, Reza Tourani, Satyajayant Misra

**Abstract**: The growing popularity of Machine Learning (ML) has led to its deployment in various sensitive domains, which has resulted in significant research focused on ML security and privacy. However, in some applications, such as autonomous driving, integrity verification of the outsourced ML workload is more critical-a facet that has not received much attention. Existing solutions, such as multi-party computation and proof-based systems, impose significant computation overhead, which makes them unfit for real-time applications. We propose Fides, a novel framework for real-time validation of outsourced ML workloads. Fides features a novel and efficient distillation technique-Greedy Distillation Transfer Learning-that dynamically distills and fine-tunes a space and compute-efficient verification model for verifying the corresponding service model while running inside a trusted execution environment. Fides features a client-side attack detection model that uses statistical analysis and divergence measurements to identify, with a high likelihood, if the service model is under attack. Fides also offers a re-classification functionality that predicts the original class whenever an attack is identified. We devised a generative adversarial network framework for training the attack detection and re-classification models. The extensive evaluation shows that Fides achieves an accuracy of up to 98% for attack detection and 94% for re-classification.



## **45. Decentralized Attack Search and the Design of Bug Bounty Schemes**

econ.TH

**SubmitDate**: 2023-03-31    [abs](http://arxiv.org/abs/2304.00077v1) [paper-pdf](http://arxiv.org/pdf/2304.00077v1)

**Authors**: Hans Gersbach, Akaki Mamageishvili, Fikri Pitsuwan

**Abstract**: Systems and blockchains often have security vulnerabilities and can be attacked by adversaries, with potentially significant negative consequences. Therefore, organizations and blockchain infrastructure providers increasingly rely on bug bounty programs, where external individuals probe the system and report any vulnerabilities (bugs) in exchange for monetary rewards (bounty). We develop a contest model for bug bounty programs with an arbitrary number of agents who decide whether to undertake a costly search for bugs or not. Search costs are private information. Besides characterizing the ensuing equilibria, we show that even inviting an unlimited crowd does not guarantee that bugs are found. Adding paid agents can increase the efficiency of the bug bounty scheme although the crowd that is attracted becomes smaller. Finally, adding (known) bugs increases the likelihood that unknown bugs are found, but to limit reward payments it may be optimal to add them only with some probability.



## **46. To be Robust and to be Fair: Aligning Fairness with Robustness**

cs.LG

**SubmitDate**: 2023-03-31    [abs](http://arxiv.org/abs/2304.00061v1) [paper-pdf](http://arxiv.org/pdf/2304.00061v1)

**Authors**: Junyi Chai, Xiaoqian Wang

**Abstract**: Adversarial training has been shown to be reliable in improving robustness against adversarial samples. However, the problem of adversarial training in terms of fairness has not yet been properly studied, and the relationship between fairness and accuracy attack still remains unclear. Can we simultaneously improve robustness w.r.t. both fairness and accuracy? To tackle this topic, in this paper, we study the problem of adversarial training and adversarial attack w.r.t. both metrics. We propose a unified structure for fairness attack which brings together common notions in group fairness, and we theoretically prove the equivalence of fairness attack against different notions. Moreover, we show the alignment of fairness and accuracy attack, and theoretically demonstrate that robustness w.r.t. one metric benefits from robustness w.r.t. the other metric. Our study suggests a novel way to unify adversarial training and attack w.r.t. fairness and accuracy, and experimental results show that our proposed method achieves better performance in terms of robustness w.r.t. both metrics.



## **47. PEOPL: Characterizing Privately Encoded Open Datasets with Public Labels**

cs.LG

Submitted to IEEE Transactions on Information Forensics and Security

**SubmitDate**: 2023-03-31    [abs](http://arxiv.org/abs/2304.00047v1) [paper-pdf](http://arxiv.org/pdf/2304.00047v1)

**Authors**: Homa Esfahanizadeh, Adam Yala, Rafael G. L. D'Oliveira, Andrea J. D. Jaba, Victor Quach, Ken R. Duffy, Tommi S. Jaakkola, Vinod Vaikuntanathan, Manya Ghobadi, Regina Barzilay, Muriel Médard

**Abstract**: Allowing organizations to share their data for training of machine learning (ML) models without unintended information leakage is an open problem in practice. A promising technique for this still-open problem is to train models on the encoded data. Our approach, called Privately Encoded Open Datasets with Public Labels (PEOPL), uses a certain class of randomly constructed transforms to encode sensitive data. Organizations publish their randomly encoded data and associated raw labels for ML training, where training is done without knowledge of the encoding realization. We investigate several important aspects of this problem: We introduce information-theoretic scores for privacy and utility, which quantify the average performance of an unfaithful user (e.g., adversary) and a faithful user (e.g., model developer) that have access to the published encoded data. We then theoretically characterize primitives in building families of encoding schemes that motivate the use of random deep neural networks. Empirically, we compare the performance of our randomized encoding scheme and a linear scheme to a suite of computational attacks, and we also show that our scheme achieves competitive prediction accuracy to raw-sample baselines. Moreover, we demonstrate that multiple institutions, using independent random encoders, can collaborate to train improved ML models.



## **48. Packet-Level Adversarial Network Traffic Crafting using Sequence Generative Adversarial Networks**

cs.CR

The authors agreed to withdraw the manuscript due to privacy reason

**SubmitDate**: 2023-03-31    [abs](http://arxiv.org/abs/2103.04794v2) [paper-pdf](http://arxiv.org/pdf/2103.04794v2)

**Authors**: Qiumei Cheng, Shiying Zhou, Yi Shen, Dezhang Kong, Chunming Wu

**Abstract**: The surge in the internet of things (IoT) devices seriously threatens the current IoT security landscape, which requires a robust network intrusion detection system (NIDS). Despite superior detection accuracy, existing machine learning or deep learning based NIDS are vulnerable to adversarial examples. Recently, generative adversarial networks (GANs) have become a prevailing method in adversarial examples crafting. However, the nature of discrete network traffic at the packet level makes it hard for GAN to craft adversarial traffic as GAN is efficient in generating continuous data like image synthesis. Unlike previous methods that convert discrete network traffic into a grayscale image, this paper gains inspiration from SeqGAN in sequence generation with policy gradient. Based on the structure of SeqGAN, we propose Attack-GAN to generate adversarial network traffic at packet level that complies with domain constraints. Specifically, the adversarial packet generation is formulated into a sequential decision making process. In this case, each byte in a packet is regarded as a token in a sequence. The objective of the generator is to select a token to maximize its expected end reward. To bypass the detection of NIDS, the generated network traffic and benign traffic are classified by a black-box NIDS. The prediction results returned by the NIDS are fed into the discriminator to guide the update of the generator. We generate malicious adversarial traffic based on a real public available dataset with attack functionality unchanged. The experimental results validate that the generated adversarial samples are able to deceive many existing black-box NIDS.



## **49. Fooling Polarization-based Vision using Locally Controllable Polarizing Projection**

cs.CV

**SubmitDate**: 2023-03-31    [abs](http://arxiv.org/abs/2303.17890v1) [paper-pdf](http://arxiv.org/pdf/2303.17890v1)

**Authors**: Zhuoxiao Li, Zhihang Zhong, Shohei Nobuhara, Ko Nishino, Yinqiang Zheng

**Abstract**: Polarization is a fundamental property of light that encodes abundant information regarding surface shape, material, illumination and viewing geometry. The computer vision community has witnessed a blossom of polarization-based vision applications, such as reflection removal, shape-from-polarization, transparent object segmentation and color constancy, partially due to the emergence of single-chip mono/color polarization sensors that make polarization data acquisition easier than ever. However, is polarization-based vision vulnerable to adversarial attacks? If so, is that possible to realize these adversarial attacks in the physical world, without being perceived by human eyes? In this paper, we warn the community of the vulnerability of polarization-based vision, which can be more serious than RGB-based vision. By adapting a commercial LCD projector, we achieve locally controllable polarizing projection, which is successfully utilized to fool state-of-the-art polarization-based vision algorithms for glass segmentation and color constancy. Compared with existing physical attacks on RGB-based vision, which always suffer from the trade-off between attack efficacy and eye conceivability, the adversarial attackers based on polarizing projection are contact-free and visually imperceptible, since naked human eyes can rarely perceive the difference of viciously manipulated polarizing light and ordinary illumination. This poses unprecedented risks on polarization-based vision, both in the monochromatic and trichromatic domain, for which due attentions should be paid and counter measures be considered.



## **50. Pentimento: Data Remanence in Cloud FPGAs**

cs.CR

17 Pages, 8 Figures

**SubmitDate**: 2023-03-31    [abs](http://arxiv.org/abs/2303.17881v1) [paper-pdf](http://arxiv.org/pdf/2303.17881v1)

**Authors**: Colin Drewes, Olivia Weng, Andres Meza, Alric Althoff, David Kohlbrenner, Ryan Kastner, Dustin Richmond

**Abstract**: Cloud FPGAs strike an alluring balance between computational efficiency, energy efficiency, and cost. It is the flexibility of the FPGA architecture that enables these benefits, but that very same flexibility that exposes new security vulnerabilities. We show that a remote attacker can recover "FPGA pentimenti" - long-removed secret data belonging to a prior user of a cloud FPGA. The sensitive data constituting an FPGA pentimento is an analog imprint from bias temperature instability (BTI) effects on the underlying transistors. We demonstrate how this slight degradation can be measured using a time-to-digital (TDC) converter when an adversary programs one into the target cloud FPGA.   This technique allows an attacker to ascertain previously safe information on cloud FPGAs, even after it is no longer explicitly present. Notably, it can allow an attacker who knows a non-secret "skeleton" (the physical structure, but not the contents) of the victim's design to (1) extract proprietary details from an encrypted FPGA design image available on the AWS marketplace and (2) recover data loaded at runtime by a previous user of a cloud FPGA using a known design. Our experiments show that BTI degradation (burn-in) and recovery are measurable and constitute a security threat to commercial cloud FPGAs.



