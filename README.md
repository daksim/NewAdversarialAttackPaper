# Latest Adversarial Attack Papers
**update at 2022-10-24 06:31:20**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. Similarity of Neural Architectures Based on Input Gradient Transferability**

cs.LG

21pages, 10 figures, 1.5MB

**SubmitDate**: 2022-10-20    [abs](http://arxiv.org/abs/2210.11407v1) [paper-pdf](http://arxiv.org/pdf/2210.11407v1)

**Authors**: Jaehui Hwang, Dongyoon Han, Byeongho Heo, Song Park, Sanghyuk Chun, Jong-Seok Lee

**Abstract**: In this paper, we aim to design a quantitative similarity function between two neural architectures. Specifically, we define a model similarity using input gradient transferability. We generate adversarial samples of two networks and measure the average accuracy of the networks on adversarial samples of each other. If two networks are highly correlated, then the attack transferability will be high, resulting in high similarity. Using the similarity score, we investigate two topics: (1) Which network component contributes to the model diversity? (2) How does model diversity affect practical scenarios? We answer the first question by providing feature importance analysis and clustering analysis. The second question is validated by two different scenarios: model ensemble and knowledge distillation. Our findings show that model diversity takes a key role when interacting with different neural architectures. For example, we found that more diversity leads to better ensemble performance. We also observe that the relationship between teacher and student networks and distillation performance depends on the choice of the base architecture of the teacher and student networks. We expect our analysis tool helps a high-level understanding of differences between various neural architectures as well as practical guidance when using multiple architectures.



## **2. Surprises in adversarially-trained linear regression**

stat.ML

**SubmitDate**: 2022-10-20    [abs](http://arxiv.org/abs/2205.12695v2) [paper-pdf](http://arxiv.org/pdf/2205.12695v2)

**Authors**: Antônio H. Ribeiro, Dave Zachariah, Thomas B. Schön

**Abstract**: State-of-the-art machine learning models can be vulnerable to very small input perturbations that are adversarially constructed. Adversarial training is an effective approach to defend against such examples. It is formulated as a min-max problem, searching for the best solution when the training data was corrupted by the worst-case attacks. For linear regression problems, adversarial training can be formulated as a convex problem. We use this reformulation to make two technical contributions: First, we formulate the training problem as an instance of robust regression to reveal its connection to parameter-shrinking methods, specifically that $\ell_\infty$-adversarial training produces sparse solutions. Secondly, we study adversarial training in the overparameterized regime, i.e. when there are more parameters than data. We prove that adversarial training with small disturbances gives the solution with the minimum-norm that interpolates the training data. Ridge regression and lasso approximate such interpolating solutions as their regularization parameter vanishes. By contrast, for adversarial training, the transition into the interpolation regime is abrupt and for non-zero values of disturbance. This result is proved and illustrated with numerical examples.



## **3. Attacking Motion Estimation with Adversarial Snow**

cs.CV

**SubmitDate**: 2022-10-20    [abs](http://arxiv.org/abs/2210.11242v1) [paper-pdf](http://arxiv.org/pdf/2210.11242v1)

**Authors**: Jenny Schmalfuss, Lukas Mehl, Andrés Bruhn

**Abstract**: Current adversarial attacks for motion estimation (optical flow) optimize small per-pixel perturbations, which are unlikely to appear in the real world. In contrast, we exploit a real-world weather phenomenon for a novel attack with adversarially optimized snow. At the core of our attack is a differentiable renderer that consistently integrates photorealistic snowflakes with realistic motion into the 3D scene. Through optimization we obtain adversarial snow that significantly impacts the optical flow while being indistinguishable from ordinary snow. Surprisingly, the impact of our novel attack is largest on methods that previously showed a high robustness to small L_p perturbations.



## **4. UKP-SQuARE v2: Explainability and Adversarial Attacks for Trustworthy QA**

cs.CL

Accepted at AACL 2022 as Demo Paper

**SubmitDate**: 2022-10-20    [abs](http://arxiv.org/abs/2208.09316v3) [paper-pdf](http://arxiv.org/pdf/2208.09316v3)

**Authors**: Rachneet Sachdeva, Haritz Puerto, Tim Baumgärtner, Sewin Tariverdian, Hao Zhang, Kexin Wang, Hossain Shaikh Saadi, Leonardo F. R. Ribeiro, Iryna Gurevych

**Abstract**: Question Answering (QA) systems are increasingly deployed in applications where they support real-world decisions. However, state-of-the-art models rely on deep neural networks, which are difficult to interpret by humans. Inherently interpretable models or post hoc explainability methods can help users to comprehend how a model arrives at its prediction and, if successful, increase their trust in the system. Furthermore, researchers can leverage these insights to develop new methods that are more accurate and less biased. In this paper, we introduce SQuARE v2, the new version of SQuARE, to provide an explainability infrastructure for comparing models based on methods such as saliency maps and graph-based explanations. While saliency maps are useful to inspect the importance of each input token for the model's prediction, graph-based explanations from external Knowledge Graphs enable the users to verify the reasoning behind the model prediction. In addition, we provide multiple adversarial attacks to compare the robustness of QA models. With these explainability methods and adversarial attacks, we aim to ease the research on trustworthy QA models. SQuARE is available on https://square.ukp-lab.de.



## **5. Analyzing the Robustness of Decentralized Horizontal and Vertical Federated Learning Architectures in a Non-IID Scenario**

cs.LG

**SubmitDate**: 2022-10-20    [abs](http://arxiv.org/abs/2210.11061v1) [paper-pdf](http://arxiv.org/pdf/2210.11061v1)

**Authors**: Pedro Miguel Sánchez Sánchez, Alberto Huertas Celdrán, Enrique Tomás Martínez Beltrán, Daniel Demeter, Gérôme Bovet, Gregorio Martínez Pérez, Burkhard Stiller

**Abstract**: Federated learning (FL) allows participants to collaboratively train machine and deep learning models while protecting data privacy. However, the FL paradigm still presents drawbacks affecting its trustworthiness since malicious participants could launch adversarial attacks against the training process. Related work has studied the robustness of horizontal FL scenarios under different attacks. However, there is a lack of work evaluating the robustness of decentralized vertical FL and comparing it with horizontal FL architectures affected by adversarial attacks. Thus, this work proposes three decentralized FL architectures, one for horizontal and two for vertical scenarios, namely HoriChain, VertiChain, and VertiComb. These architectures present different neural networks and training protocols suitable for horizontal and vertical scenarios. Then, a decentralized, privacy-preserving, and federated use case with non-IID data to classify handwritten digits is deployed to evaluate the performance of the three architectures. Finally, a set of experiments computes and compares the robustness of the proposed architectures when they are affected by different data poisoning based on image watermarks and gradient poisoning adversarial attacks. The experiments show that even though particular configurations of both attacks can destroy the classification performance of the architectures, HoriChain is the most robust one.



## **6. Defending Person Detection Against Adversarial Patch Attack by using Universal Defensive Frame**

cs.CV

Accepted at IEEE Transactions on Image Processing (TIP), 2022

**SubmitDate**: 2022-10-20    [abs](http://arxiv.org/abs/2204.13004v2) [paper-pdf](http://arxiv.org/pdf/2204.13004v2)

**Authors**: Youngjoon Yu, Hong Joo Lee, Hakmin Lee, Yong Man Ro

**Abstract**: Person detection has attracted great attention in the computer vision area and is an imperative element in human-centric computer vision. Although the predictive performances of person detection networks have been improved dramatically, they are vulnerable to adversarial patch attacks. Changing the pixels in a restricted region can easily fool the person detection network in safety-critical applications such as autonomous driving and security systems. Despite the necessity of countering adversarial patch attacks, very few efforts have been dedicated to defending person detection against adversarial patch attack. In this paper, we propose a novel defense strategy that defends against an adversarial patch attack by optimizing a defensive frame for person detection. The defensive frame alleviates the effect of the adversarial patch while maintaining person detection performance with clean person. The proposed defensive frame in the person detection is generated with a competitive learning algorithm which makes an iterative competition between detection threatening module and detection shielding module in person detection. Comprehensive experimental results demonstrate that the proposed method effectively defends person detection against adversarial patch attacks.



## **7. Towards Adversarial Attack on Vision-Language Pre-training Models**

cs.LG

Accepted by ACM MM2022. Code is available in GitHub

**SubmitDate**: 2022-10-20    [abs](http://arxiv.org/abs/2206.09391v2) [paper-pdf](http://arxiv.org/pdf/2206.09391v2)

**Authors**: Jiaming Zhang, Qi Yi, Jitao Sang

**Abstract**: While vision-language pre-training model (VLP) has shown revolutionary improvements on various vision-language (V+L) tasks, the studies regarding its adversarial robustness remain largely unexplored. This paper studied the adversarial attack on popular VLP models and V+L tasks. First, we analyzed the performance of adversarial attacks under different settings. By examining the influence of different perturbed objects and attack targets, we concluded some key observations as guidance on both designing strong multimodal adversarial attack and constructing robust VLP models. Second, we proposed a novel multimodal attack method on the VLP models called Collaborative Multimodal Adversarial Attack (Co-Attack), which collectively carries out the attacks on the image modality and the text modality. Experimental results demonstrated that the proposed method achieves improved attack performances on different V+L downstream tasks and VLP models. The analysis observations and novel attack method hopefully provide new understanding into the adversarial robustness of VLP models, so as to contribute their safe and reliable deployment in more real-world scenarios. Code is available at https://github.com/adversarial-for-goodness/Co-Attack.



## **8. Rewriting Meaningful Sentences via Conditional BERT Sampling and an application on fooling text classifiers**

cs.CL

Please see an updated version of this paper at arXiv:2104.08453

**SubmitDate**: 2022-10-19    [abs](http://arxiv.org/abs/2010.11869v2) [paper-pdf](http://arxiv.org/pdf/2010.11869v2)

**Authors**: Lei Xu, Ivan Ramirez, Kalyan Veeramachaneni

**Abstract**: Most adversarial attack methods that are designed to deceive a text classifier change the text classifier's prediction by modifying a few words or characters. Few try to attack classifiers by rewriting a whole sentence, due to the difficulties inherent in sentence-level rephrasing as well as the problem of setting the criteria for legitimate rewriting.   In this paper, we explore the problem of creating adversarial examples with sentence-level rewriting. We design a new sampling method, named ParaphraseSampler, to efficiently rewrite the original sentence in multiple ways. Then we propose a new criteria for modification, called a sentence-level threaten model. This criteria allows for both word- and sentence-level changes, and can be adjusted independently in two dimensions: semantic similarity and grammatical quality. Experimental results show that many of these rewritten sentences are misclassified by the classifier. On all 6 datasets, our ParaphraseSampler achieves a better attack success rate than our baseline.



## **9. R&R: Metric-guided Adversarial Sentence Generation**

cs.CL

Accepted to Finding of AACL-IJCNLP2022

**SubmitDate**: 2022-10-19    [abs](http://arxiv.org/abs/2104.08453v3) [paper-pdf](http://arxiv.org/pdf/2104.08453v3)

**Authors**: Lei Xu, Alfredo Cuesta-Infante, Laure Berti-Equille, Kalyan Veeramachaneni

**Abstract**: Adversarial examples are helpful for analyzing and improving the robustness of text classifiers. Generating high-quality adversarial examples is a challenging task as it requires generating fluent adversarial sentences that are semantically similar to the original sentences and preserve the original labels, while causing the classifier to misclassify them. Existing methods prioritize misclassification by maximizing each perturbation's effectiveness at misleading a text classifier; thus, the generated adversarial examples fall short in terms of fluency and similarity. In this paper, we propose a rewrite and rollback (R&R) framework for adversarial attack. It improves the quality of adversarial examples by optimizing a critique score which combines the fluency, similarity, and misclassification metrics. R&R generates high-quality adversarial examples by allowing exploration of perturbations that do not have immediate impact on the misclassification metric but can improve fluency and similarity metrics. We evaluate our method on 5 representative datasets and 3 classifier architectures. Our method outperforms current state-of-the-art in attack success rate by +16.2%, +12.8%, and +14.0% on the classifiers respectively. Code is available at https://github.com/DAI-Lab/fibber



## **10. Backdoor Attack and Defense in Federated Generative Adversarial Network-based Medical Image Synthesis**

cs.CV

25 pages, 7 figures. arXiv admin note: text overlap with  arXiv:2207.00762

**SubmitDate**: 2022-10-19    [abs](http://arxiv.org/abs/2210.10886v1) [paper-pdf](http://arxiv.org/pdf/2210.10886v1)

**Authors**: Ruinan Jin, Xiaoxiao Li

**Abstract**: Deep Learning-based image synthesis techniques have been applied in healthcare research for generating medical images to support open research and augment medical datasets. Training generative adversarial neural networks (GANs) usually require large amounts of training data. Federated learning (FL) provides a way of training a central model using distributed data while keeping raw data locally. However, given that the FL server cannot access the raw data, it is vulnerable to backdoor attacks, an adversarial by poisoning training data. Most backdoor attack strategies focus on classification models and centralized domains. It is still an open question if the existing backdoor attacks can affect GAN training and, if so, how to defend against the attack in the FL setting. In this work, we investigate the overlooked issue of backdoor attacks in federated GANs (FedGANs). The success of this attack is subsequently determined to be the result of some local discriminators overfitting the poisoned data and corrupting the local GAN equilibrium, which then further contaminates other clients when averaging the generator's parameters and yields high generator loss. Therefore, we proposed FedDetect, an efficient and effective way of defending against the backdoor attack in the FL setting, which allows the server to detect the client's adversarial behavior based on their losses and block the malicious clients. Our extensive experiments on two medical datasets with different modalities demonstrate the backdoor attack on FedGANs can result in synthetic images with low fidelity. After detecting and suppressing the detected malicious clients using the proposed defense strategy, we show that FedGANs can synthesize high-quality medical datasets (with labels) for data augmentation to improve classification models' performance.



## **11. On the Perils of Cascading Robust Classifiers**

cs.LG

**SubmitDate**: 2022-10-19    [abs](http://arxiv.org/abs/2206.00278v2) [paper-pdf](http://arxiv.org/pdf/2206.00278v2)

**Authors**: Ravi Mangal, Zifan Wang, Chi Zhang, Klas Leino, Corina Pasareanu, Matt Fredrikson

**Abstract**: Ensembling certifiably robust neural networks is a promising approach for improving the \emph{certified robust accuracy} of neural models. Black-box ensembles that assume only query-access to the constituent models (and their robustness certifiers) during prediction are particularly attractive due to their modular structure. Cascading ensembles are a popular instance of black-box ensembles that appear to improve certified robust accuracies in practice. However, we show that the robustness certifier used by a cascading ensemble is unsound. That is, when a cascading ensemble is certified as locally robust at an input $x$ (with respect to $\epsilon$), there can be inputs $x'$ in the $\epsilon$-ball centered at $x$, such that the cascade's prediction at $x'$ is different from $x$ and thus the ensemble is not locally robust. Our theoretical findings are accompanied by empirical results that further demonstrate this unsoundness. We present \emph{cascade attack} (CasA), an adversarial attack against cascading ensembles, and show that: (1) there exists an adversarial input for up to 88\% of the samples where the ensemble claims to be certifiably robust and accurate; and (2) the accuracy of a cascading ensemble under our attack is as low as 11\% when it claims to be certifiably robust and accurate on 97\% of the test set. Our work reveals a critical pitfall of cascading certifiably robust models by showing that the seemingly beneficial strategy of cascading can actually hurt the robustness of the resulting ensemble. Our code is available at \url{https://github.com/TristaChi/ensembleKW}.



## **12. Why Should Adversarial Perturbations be Imperceptible? Rethink the Research Paradigm in Adversarial NLP**

cs.CL

Accepted to EMNLP 2022, main conference

**SubmitDate**: 2022-10-19    [abs](http://arxiv.org/abs/2210.10683v1) [paper-pdf](http://arxiv.org/pdf/2210.10683v1)

**Authors**: Yangyi Chen, Hongcheng Gao, Ganqu Cui, Fanchao Qi, Longtao Huang, Zhiyuan Liu, Maosong Sun

**Abstract**: Textual adversarial samples play important roles in multiple subfields of NLP research, including security, evaluation, explainability, and data augmentation. However, most work mixes all these roles, obscuring the problem definitions and research goals of the security role that aims to reveal the practical concerns of NLP models. In this paper, we rethink the research paradigm of textual adversarial samples in security scenarios. We discuss the deficiencies in previous work and propose our suggestions that the research on the Security-oriented adversarial NLP (SoadNLP) should: (1) evaluate their methods on security tasks to demonstrate the real-world concerns; (2) consider real-world attackers' goals, instead of developing impractical methods. To this end, we first collect, process, and release a security datasets collection Advbench. Then, we reformalize the task and adjust the emphasis on different goals in SoadNLP. Next, we propose a simple method based on heuristic rules that can easily fulfill the actual adversarial goals to simulate real-world attack methods. We conduct experiments on both the attack and the defense sides on Advbench. Experimental results show that our method has higher practical value, indicating that the research paradigm in SoadNLP may start from our new benchmark. All the code and data of Advbench can be obtained at \url{https://github.com/thunlp/Advbench}.



## **13. Textual Backdoor Attacks Can Be More Harmful via Two Simple Tricks**

cs.CR

Accepted to EMNLP 2022, main conference

**SubmitDate**: 2022-10-19    [abs](http://arxiv.org/abs/2110.08247v2) [paper-pdf](http://arxiv.org/pdf/2110.08247v2)

**Authors**: Yangyi Chen, Fanchao Qi, Hongcheng Gao, Zhiyuan Liu, Maosong Sun

**Abstract**: Backdoor attacks are a kind of emergent security threat in deep learning. After being injected with a backdoor, a deep neural model will behave normally on standard inputs but give adversary-specified predictions once the input contains specific backdoor triggers. In this paper, we find two simple tricks that can make existing textual backdoor attacks much more harmful. The first trick is to add an extra training task to distinguish poisoned and clean data during the training of the victim model, and the second one is to use all the clean training data rather than remove the original clean data corresponding to the poisoned data. These two tricks are universally applicable to different attack models. We conduct experiments in three tough situations including clean data fine-tuning, low-poisoning-rate, and label-consistent attacks. Experimental results show that the two tricks can significantly improve attack performance. This paper exhibits the great potential harmfulness of backdoor attacks. All the code and data can be obtained at \url{https://github.com/thunlp/StyleAttack}.



## **14. Few-shot Transferable Robust Representation Learning via Bilevel Attacks**

cs.LG

*Equal contribution. Author ordering determined by coin flip

**SubmitDate**: 2022-10-19    [abs](http://arxiv.org/abs/2210.10485v1) [paper-pdf](http://arxiv.org/pdf/2210.10485v1)

**Authors**: Minseon Kim, Hyeonjeong Ha, Sung Ju Hwang

**Abstract**: Existing adversarial learning methods for enhancing the robustness of deep neural networks assume the availability of a large amount of data from which we can generate adversarial examples. However, in an adversarial meta-learning setting, the model needs to train with only a few adversarial examples to learn a robust model for unseen tasks, which is a very difficult goal to achieve. Further, learning transferable robust representations for unseen domains is a difficult problem even with a large amount of data. To tackle such a challenge, we propose a novel adversarial self-supervised meta-learning framework with bilevel attacks which aims to learn robust representations that can generalize across tasks and domains. Specifically, in the inner loop, we update the parameters of the given encoder by taking inner gradient steps using two different sets of augmented samples, and generate adversarial examples for each view by maximizing the instance classification loss. Then, in the outer loop, we meta-learn the encoder parameter to maximize the agreement between the two adversarial examples, which enables it to learn robust representations. We experimentally validate the effectiveness of our approach on unseen domain adaptation tasks, on which it achieves impressive performance. Specifically, our method significantly outperforms the state-of-the-art meta-adversarial learning methods on few-shot learning tasks, as well as self-supervised learning baselines in standard learning settings with large-scale datasets.



## **15. Emerging Threats in Deep Learning-Based Autonomous Driving: A Comprehensive Survey**

cs.CR

28 pages,10 figures

**SubmitDate**: 2022-10-19    [abs](http://arxiv.org/abs/2210.11237v1) [paper-pdf](http://arxiv.org/pdf/2210.11237v1)

**Authors**: Hui Cao, Wenlong Zou, Yinkun Wang, Ting Song, Mengjun Liu

**Abstract**: Since the 2004 DARPA Grand Challenge, the autonomous driving technology has witnessed nearly two decades of rapid development. Particularly, in recent years, with the application of new sensors and deep learning technologies extending to the autonomous field, the development of autonomous driving technology has continued to make breakthroughs. Thus, many carmakers and high-tech giants dedicated to research and system development of autonomous driving. However, as the foundation of autonomous driving, the deep learning technology faces many new security risks. The academic community has proposed deep learning countermeasures against the adversarial examples and AI backdoor, and has introduced them into the autonomous driving field for verification. Deep learning security matters to autonomous driving system security, and then matters to personal safety, which is an issue that deserves attention and research.This paper provides an summary of the concepts, developments and recent research in deep learning security technologies in autonomous driving. Firstly, we briefly introduce the deep learning framework and pipeline in the autonomous driving system, which mainly include the deep learning technologies and algorithms commonly used in this field. Moreover, we focus on the potential security threats of the deep learning based autonomous driving system in each functional layer in turn. We reviews the development of deep learning attack technologies to autonomous driving, investigates the State-of-the-Art algorithms, and reveals the potential risks. At last, we provides an outlook on deep learning security in the autonomous driving field and proposes recommendations for building a safe and trustworthy autonomous driving system.



## **16. On the Adversarial Robustness of Mixture of Experts**

cs.LG

Accepted to NeurIPS 2022

**SubmitDate**: 2022-10-19    [abs](http://arxiv.org/abs/2210.10253v1) [paper-pdf](http://arxiv.org/pdf/2210.10253v1)

**Authors**: Joan Puigcerver, Rodolphe Jenatton, Carlos Riquelme, Pranjal Awasthi, Srinadh Bhojanapalli

**Abstract**: Adversarial robustness is a key desirable property of neural networks. It has been empirically shown to be affected by their sizes, with larger networks being typically more robust. Recently, Bubeck and Sellke proved a lower bound on the Lipschitz constant of functions that fit the training data in terms of their number of parameters. This raises an interesting open question, do -- and can -- functions with more parameters, but not necessarily more computational cost, have better robustness? We study this question for sparse Mixture of Expert models (MoEs), that make it possible to scale up the model size for a roughly constant computational cost. We theoretically show that under certain conditions on the routing and the structure of the data, MoEs can have significantly smaller Lipschitz constants than their dense counterparts. The robustness of MoEs can suffer when the highest weighted experts for an input implement sufficiently different functions. We next empirically evaluate the robustness of MoEs on ImageNet using adversarial attacks and show they are indeed more robust than dense models with the same computational cost. We make key observations showing the robustness of MoEs to the choice of experts, highlighting the redundancy of experts in models trained in practice.



## **17. Parsimonious Black-Box Adversarial Attacks via Efficient Combinatorial Optimization**

cs.LG

Accepted and to appear at ICML 2019

**SubmitDate**: 2022-10-18    [abs](http://arxiv.org/abs/1905.06635v2) [paper-pdf](http://arxiv.org/pdf/1905.06635v2)

**Authors**: Seungyong Moon, Gaon An, Hyun Oh Song

**Abstract**: Solving for adversarial examples with projected gradient descent has been demonstrated to be highly effective in fooling the neural network based classifiers. However, in the black-box setting, the attacker is limited only to the query access to the network and solving for a successful adversarial example becomes much more difficult. To this end, recent methods aim at estimating the true gradient signal based on the input queries but at the cost of excessive queries. We propose an efficient discrete surrogate to the optimization problem which does not require estimating the gradient and consequently becomes free of the first order update hyperparameters to tune. Our experiments on Cifar-10 and ImageNet show the state of the art black-box attack performance with significant reduction in the required queries compared to a number of recently proposed methods. The source code is available at https://github.com/snu-mllab/parsimonious-blackbox-attack.



## **18. Scaling Adversarial Training to Large Perturbation Bounds**

cs.LG

ECCV 2022

**SubmitDate**: 2022-10-18    [abs](http://arxiv.org/abs/2210.09852v1) [paper-pdf](http://arxiv.org/pdf/2210.09852v1)

**Authors**: Sravanti Addepalli, Samyak Jain, Gaurang Sriramanan, R. Venkatesh Babu

**Abstract**: The vulnerability of Deep Neural Networks to Adversarial Attacks has fuelled research towards building robust models. While most Adversarial Training algorithms aim at defending attacks constrained within low magnitude Lp norm bounds, real-world adversaries are not limited by such constraints. In this work, we aim to achieve adversarial robustness within larger bounds, against perturbations that may be perceptible, but do not change human (or Oracle) prediction. The presence of images that flip Oracle predictions and those that do not makes this a challenging setting for adversarial robustness. We discuss the ideal goals of an adversarial defense algorithm beyond perceptual limits, and further highlight the shortcomings of naively extending existing training algorithms to higher perturbation bounds. In order to overcome these shortcomings, we propose a novel defense, Oracle-Aligned Adversarial Training (OA-AT), to align the predictions of the network with that of an Oracle during adversarial training. The proposed approach achieves state-of-the-art performance at large epsilon bounds (such as an L-inf bound of 16/255 on CIFAR-10) while outperforming existing defenses (AWP, TRADES, PGD-AT) at standard bounds (8/255) as well.



## **19. Provably Robust Detection of Out-of-distribution Data (almost) for free**

cs.LG

**SubmitDate**: 2022-10-18    [abs](http://arxiv.org/abs/2106.04260v2) [paper-pdf](http://arxiv.org/pdf/2106.04260v2)

**Authors**: Alexander Meinke, Julian Bitterwolf, Matthias Hein

**Abstract**: The application of machine learning in safety-critical systems requires a reliable assessment of uncertainty. However, deep neural networks are known to produce highly overconfident predictions on out-of-distribution (OOD) data. Even if trained to be non-confident on OOD data, one can still adversarially manipulate OOD data so that the classifier again assigns high confidence to the manipulated samples. We show that two previously published defenses can be broken by better adapted attacks, highlighting the importance of robustness guarantees around OOD data. Since the existing method for this task is hard to train and significantly limits accuracy, we construct a classifier that can simultaneously achieve provably adversarially robust OOD detection and high clean accuracy. Moreover, by slightly modifying the classifier's architecture our method provably avoids the asymptotic overconfidence problem of standard neural networks. We provide code for all our experiments.



## **20. ROSE: Robust Selective Fine-tuning for Pre-trained Language Models**

cs.CL

Accepted to EMNLP 2022. Code is available at  https://github.com/jiangllan/ROSE

**SubmitDate**: 2022-10-18    [abs](http://arxiv.org/abs/2210.09658v1) [paper-pdf](http://arxiv.org/pdf/2210.09658v1)

**Authors**: Lan Jiang, Hao Zhou, Yankai Lin, Peng Li, Jie Zhou, Rui Jiang

**Abstract**: Even though the large-scale language models have achieved excellent performances, they suffer from various adversarial attacks. A large body of defense methods has been proposed. However, they are still limited due to redundant attack search spaces and the inability to defend against various types of attacks. In this work, we present a novel fine-tuning approach called \textbf{RO}bust \textbf{SE}letive fine-tuning (\textbf{ROSE}) to address this issue. ROSE conducts selective updates when adapting pre-trained models to downstream tasks, filtering out invaluable and unrobust updates of parameters. Specifically, we propose two strategies: the first-order and second-order ROSE for selecting target robust parameters. The experimental results show that ROSE achieves significant improvements in adversarial robustness on various downstream NLP tasks, and the ensemble method even surpasses both variants above. Furthermore, ROSE can be easily incorporated into existing fine-tuning methods to improve their adversarial robustness further. The empirical analysis confirms that ROSE eliminates unrobust spurious updates during fine-tuning, leading to solutions corresponding to flatter and wider optima than the conventional method. Code is available at \url{https://github.com/jiangllan/ROSE}.



## **21. Analysis of Master Vein Attacks on Finger Vein Recognition Systems**

cs.CV

Accepted to be Published in Proceedings of the IEEE/CVF Winter  Conference on Applications of Computer Vision (WACV) 2023

**SubmitDate**: 2022-10-18    [abs](http://arxiv.org/abs/2210.10667v1) [paper-pdf](http://arxiv.org/pdf/2210.10667v1)

**Authors**: Huy H. Nguyen, Trung-Nghia Le, Junichi Yamagishi, Isao Echizen

**Abstract**: Finger vein recognition (FVR) systems have been commercially used, especially in ATMs, for customer verification. Thus, it is essential to measure their robustness against various attack methods, especially when a hand-crafted FVR system is used without any countermeasure methods. In this paper, we are the first in the literature to introduce master vein attacks in which we craft a vein-looking image so that it can falsely match with as many identities as possible by the FVR systems. We present two methods for generating master veins for use in attacking these systems. The first uses an adaptation of the latent variable evolution algorithm with a proposed generative model (a multi-stage combination of beta-VAE and WGAN-GP models). The second uses an adversarial machine learning attack method to attack a strong surrogate CNN-based recognition system. The two methods can be easily combined to boost their attack ability. Experimental results demonstrated that the proposed methods alone and together achieved false acceptance rates up to 73.29% and 88.79%, respectively, against Miura's hand-crafted FVR system. We also point out that Miura's system is easily compromised by non-vein-looking samples generated by a WGAN-GP model with false acceptance rates up to 94.21%. The results raise the alarm about the robustness of such systems and suggest that master vein attacks should be considered an important security measure.



## **22. Making Split Learning Resilient to Label Leakage by Potential Energy Loss**

cs.CR

**SubmitDate**: 2022-10-18    [abs](http://arxiv.org/abs/2210.09617v1) [paper-pdf](http://arxiv.org/pdf/2210.09617v1)

**Authors**: Fei Zheng, Chaochao Chen, Binhui Yao, Xiaolin Zheng

**Abstract**: As a practical privacy-preserving learning method, split learning has drawn much attention in academia and industry. However, its security is constantly being questioned since the intermediate results are shared during training and inference. In this paper, we focus on the privacy leakage problem caused by the trained split model, i.e., the attacker can use a few labeled samples to fine-tune the bottom model, and gets quite good performance. To prevent such kind of privacy leakage, we propose the potential energy loss to make the output of the bottom model become a more `complicated' distribution, by pushing outputs of the same class towards the decision boundary. Therefore, the adversary suffers a large generalization error when fine-tuning the bottom model with only a few leaked labeled samples. Experiment results show that our method significantly lowers the attacker's fine-tuning accuracy, making the split model more resilient to label leakage.



## **23. Friendly Noise against Adversarial Noise: A Powerful Defense against Data Poisoning Attacks**

cs.CR

**SubmitDate**: 2022-10-18    [abs](http://arxiv.org/abs/2208.10224v3) [paper-pdf](http://arxiv.org/pdf/2208.10224v3)

**Authors**: Tian Yu Liu, Yu Yang, Baharan Mirzasoleiman

**Abstract**: A powerful category of (invisible) data poisoning attacks modify a subset of training examples by small adversarial perturbations to change the prediction of certain test-time data. Existing defense mechanisms are not desirable to deploy in practice, as they often either drastically harm the generalization performance, or are attack-specific, and prohibitively slow to apply. Here, we propose a simple but highly effective approach that unlike existing methods breaks various types of invisible poisoning attacks with the slightest drop in the generalization performance. We make the key observation that attacks introduce local sharp regions of high training loss, which when minimized, results in learning the adversarial perturbations and makes the attack successful. To break poisoning attacks, our key idea is to alleviate the sharp loss regions introduced by poisons. To do so, our approach comprises two components: an optimized friendly noise that is generated to maximally perturb examples without degrading the performance, and a randomly varying noise component. The combination of both components builds a very light-weight but extremely effective defense against the most powerful triggerless targeted and hidden-trigger backdoor poisoning attacks, including Gradient Matching, Bulls-eye Polytope, and Sleeper Agent. We show that our friendly noise is transferable to other architectures, and adaptive attacks cannot break our defense due to its random noise component.



## **24. Make Some Noise: Reliable and Efficient Single-Step Adversarial Training**

cs.LG

Published in NeurIPS 2022

**SubmitDate**: 2022-10-17    [abs](http://arxiv.org/abs/2202.01181v3) [paper-pdf](http://arxiv.org/pdf/2202.01181v3)

**Authors**: Pau de Jorge, Adel Bibi, Riccardo Volpi, Amartya Sanyal, Philip H. S. Torr, Grégory Rogez, Puneet K. Dokania

**Abstract**: Recently, Wong et al. showed that adversarial training with single-step FGSM leads to a characteristic failure mode named Catastrophic Overfitting (CO), in which a model becomes suddenly vulnerable to multi-step attacks. Experimentally they showed that simply adding a random perturbation prior to FGSM (RS-FGSM) could prevent CO. However, Andriushchenko and Flammarion observed that RS-FGSM still leads to CO for larger perturbations, and proposed a computationally expensive regularizer (GradAlign) to avoid it. In this work, we methodically revisit the role of noise and clipping in single-step adversarial training. Contrary to previous intuitions, we find that using a stronger noise around the clean sample combined with \textit{not clipping} is highly effective in avoiding CO for large perturbation radii. We then propose Noise-FGSM (N-FGSM) that, while providing the benefits of single-step adversarial training, does not suffer from CO. Empirical analyses on a large suite of experiments show that N-FGSM is able to match or surpass the performance of previous state-of-the-art GradAlign, while achieving 3x speed-up. Code can be found in https://github.com/pdejorge/N-FGSM



## **25. Deepfake Text Detection: Limitations and Opportunities**

cs.CR

Accepted to IEEE S&P 2023; First two authors contributed equally to  this work; 18 pages, 7 figures

**SubmitDate**: 2022-10-17    [abs](http://arxiv.org/abs/2210.09421v1) [paper-pdf](http://arxiv.org/pdf/2210.09421v1)

**Authors**: Jiameng Pu, Zain Sarwar, Sifat Muhammad Abdullah, Abdullah Rehman, Yoonjin Kim, Parantapa Bhattacharya, Mobin Javed, Bimal Viswanath

**Abstract**: Recent advances in generative models for language have enabled the creation of convincing synthetic text or deepfake text. Prior work has demonstrated the potential for misuse of deepfake text to mislead content consumers. Therefore, deepfake text detection, the task of discriminating between human and machine-generated text, is becoming increasingly critical. Several defenses have been proposed for deepfake text detection. However, we lack a thorough understanding of their real-world applicability. In this paper, we collect deepfake text from 4 online services powered by Transformer-based tools to evaluate the generalization ability of the defenses on content in the wild. We develop several low-cost adversarial attacks, and investigate the robustness of existing defenses against an adaptive attacker. We find that many defenses show significant degradation in performance under our evaluation scenarios compared to their original claimed performance. Our evaluation shows that tapping into the semantic information in the text content is a promising approach for improving the robustness and generalization performance of deepfake text detection schemes.



## **26. Towards Generating Adversarial Examples on Mixed-type Data**

cs.LG

**SubmitDate**: 2022-10-17    [abs](http://arxiv.org/abs/2210.09405v1) [paper-pdf](http://arxiv.org/pdf/2210.09405v1)

**Authors**: Han Xu, Menghai Pan, Zhimeng Jiang, Huiyuan Chen, Xiaoting Li, Mahashweta Das, Hao Yang

**Abstract**: The existence of adversarial attacks (or adversarial examples) brings huge concern about the machine learning (ML) model's safety issues. For many safety-critical ML tasks, such as financial forecasting, fraudulent detection, and anomaly detection, the data samples are usually mixed-type, which contain plenty of numerical and categorical features at the same time. However, how to generate adversarial examples with mixed-type data is still seldom studied. In this paper, we propose a novel attack algorithm M-Attack, which can effectively generate adversarial examples in mixed-type data. Based on M-Attack, attackers can attempt to mislead the targeted classification model's prediction, by only slightly perturbing both the numerical and categorical features in the given data samples. More importantly, by adding designed regularizations, our generated adversarial examples can evade potential detection models, which makes the attack indeed insidious. Through extensive empirical studies, we validate the effectiveness and efficiency of our attack method and evaluate the robustness of existing classification models against our proposed attack. The experimental results highlight the feasibility of generating adversarial examples toward machine learning models in real-world applications.



## **27. Probabilistic Categorical Adversarial Attack & Adversarial Training**

cs.LG

**SubmitDate**: 2022-10-17    [abs](http://arxiv.org/abs/2210.09364v1) [paper-pdf](http://arxiv.org/pdf/2210.09364v1)

**Authors**: Penghei He, Han Xu, Jie Ren, Yuxuan Wan, Zitao Liu, Jiliang Tang

**Abstract**: The existence of adversarial examples brings huge concern for people to apply Deep Neural Networks (DNNs) in safety-critical tasks. However, how to generate adversarial examples with categorical data is an important problem but lack of extensive exploration. Previously established methods leverage greedy search method, which can be very time-consuming to conduct successful attack. This also limits the development of adversarial training and potential defenses for categorical data. To tackle this problem, we propose Probabilistic Categorical Adversarial Attack (PCAA), which transfers the discrete optimization problem to a continuous problem that can be solved efficiently by Projected Gradient Descent. In our paper, we theoretically analyze its optimality and time complexity to demonstrate its significant advantage over current greedy based attacks. Moreover, based on our attack, we propose an efficient adversarial training framework. Through a comprehensive empirical study, we justify the effectiveness of our proposed attack and defense algorithms.



## **28. Marksman Backdoor: Backdoor Attacks with Arbitrary Target Class**

cs.CR

Accepted to NeurIPS 2022

**SubmitDate**: 2022-10-17    [abs](http://arxiv.org/abs/2210.09194v1) [paper-pdf](http://arxiv.org/pdf/2210.09194v1)

**Authors**: Khoa D. Doan, Yingjie Lao, Ping Li

**Abstract**: In recent years, machine learning models have been shown to be vulnerable to backdoor attacks. Under such attacks, an adversary embeds a stealthy backdoor into the trained model such that the compromised models will behave normally on clean inputs but will misclassify according to the adversary's control on maliciously constructed input with a trigger. While these existing attacks are very effective, the adversary's capability is limited: given an input, these attacks can only cause the model to misclassify toward a single pre-defined or target class. In contrast, this paper exploits a novel backdoor attack with a much more powerful payload, denoted as Marksman, where the adversary can arbitrarily choose which target class the model will misclassify given any input during inference. To achieve this goal, we propose to represent the trigger function as a class-conditional generative model and to inject the backdoor in a constrained optimization framework, where the trigger function learns to generate an optimal trigger pattern to attack any target class at will while simultaneously embedding this generative backdoor into the trained model. Given the learned trigger-generation function, during inference, the adversary can specify an arbitrary backdoor attack target class, and an appropriate trigger causing the model to classify toward this target class is created accordingly. We show empirically that the proposed framework achieves high attack performance while preserving the clean-data performance in several benchmark datasets, including MNIST, CIFAR10, GTSRB, and TinyImageNet. The proposed Marksman backdoor attack can also easily bypass existing backdoor defenses that were originally designed against backdoor attacks with a single target class. Our work takes another significant step toward understanding the extensive risks of backdoor attacks in practice.



## **29. Adversarial Robustness is at Odds with Lazy Training**

cs.LG

NeurIPS 2022

**SubmitDate**: 2022-10-17    [abs](http://arxiv.org/abs/2207.00411v2) [paper-pdf](http://arxiv.org/pdf/2207.00411v2)

**Authors**: Yunjuan Wang, Enayat Ullah, Poorya Mianjy, Raman Arora

**Abstract**: Recent works show that adversarial examples exist for random neural networks [Daniely and Schacham, 2020] and that these examples can be found using a single step of gradient ascent [Bubeck et al., 2021]. In this work, we extend this line of work to "lazy training" of neural networks -- a dominant model in deep learning theory in which neural networks are provably efficiently learnable. We show that over-parametrized neural networks that are guaranteed to generalize well and enjoy strong computational guarantees remain vulnerable to attacks generated using a single step of gradient ascent.



## **30. DE-CROP: Data-efficient Certified Robustness for Pretrained Classifiers**

cs.LG

WACV 2023. Project page: https://sites.google.com/view/decrop

**SubmitDate**: 2022-10-17    [abs](http://arxiv.org/abs/2210.08929v1) [paper-pdf](http://arxiv.org/pdf/2210.08929v1)

**Authors**: Gaurav Kumar Nayak, Ruchit Rawal, Anirban Chakraborty

**Abstract**: Certified defense using randomized smoothing is a popular technique to provide robustness guarantees for deep neural networks against l2 adversarial attacks. Existing works use this technique to provably secure a pretrained non-robust model by training a custom denoiser network on entire training data. However, access to the training set may be restricted to a handful of data samples due to constraints such as high transmission cost and the proprietary nature of the data. Thus, we formulate a novel problem of "how to certify the robustness of pretrained models using only a few training samples". We observe that training the custom denoiser directly using the existing techniques on limited samples yields poor certification. To overcome this, our proposed approach (DE-CROP) generates class-boundary and interpolated samples corresponding to each training sample, ensuring high diversity in the feature space of the pretrained classifier. We train the denoiser by maximizing the similarity between the denoised output of the generated sample and the original training sample in the classifier's logit space. We also perform distribution level matching using domain discriminator and maximum mean discrepancy that yields further benefit. In white box setup, we obtain significant improvements over the baseline on multiple benchmark datasets and also report similar performance under the challenging black box setup.



## **31. Beyond Model Interpretability: On the Faithfulness and Adversarial Robustness of Contrastive Textual Explanations**

cs.CL

**SubmitDate**: 2022-10-17    [abs](http://arxiv.org/abs/2210.08902v1) [paper-pdf](http://arxiv.org/pdf/2210.08902v1)

**Authors**: Julia El Zini, Mariette Awad

**Abstract**: Contrastive explanation methods go beyond transparency and address the contrastive aspect of explanations. Such explanations are emerging as an attractive option to provide actionable change to scenarios adversely impacted by classifiers' decisions. However, their extension to textual data is under-explored and there is little investigation on their vulnerabilities and limitations.   This work motivates textual counterfactuals by laying the ground for a novel evaluation scheme inspired by the faithfulness of explanations. Accordingly, we extend the computation of three metrics, proximity,connectedness and stability, to textual data and we benchmark two successful contrastive methods, POLYJUICE and MiCE, on our suggested metrics. Experiments on sentiment analysis data show that the connectedness of counterfactuals to their original counterparts is not obvious in both models. More interestingly, the generated contrastive texts are more attainable with POLYJUICE which highlights the significance of latent representations in counterfactual search. Finally, we perform the first semantic adversarial attack on textual recourse methods. The results demonstrate the robustness of POLYJUICE and the role that latent input representations play in robustness and reliability.



## **32. Differential Evolution based Dual Adversarial Camouflage: Fooling Human Eyes and Object Detectors**

cs.CV

**SubmitDate**: 2022-10-17    [abs](http://arxiv.org/abs/2210.08870v1) [paper-pdf](http://arxiv.org/pdf/2210.08870v1)

**Authors**: Jialiang Sun

**Abstract**: Recent studies reveal that deep neural network (DNN) based object detectors are vulnerable to adversarial attacks in the form of adding the perturbation to the images, leading to the wrong output of object detectors. Most current existing works focus on generating perturbed images, also called adversarial examples, to fool object detectors. Though the generated adversarial examples themselves can remain a certain naturalness, most of them can still be easily observed by human eyes, which limits their further application in the real world. To alleviate this problem, we propose a differential evolution based dual adversarial camouflage (DE_DAC) method, composed of two stages to fool human eyes and object detectors simultaneously. Specifically, we try to obtain the camouflage texture, which can be rendered over the surface of the object. In the first stage, we optimize the global texture to minimize the discrepancy between the rendered object and the scene images, making human eyes difficult to distinguish. In the second stage, we design three loss functions to optimize the local texture, making object detectors ineffective. In addition, we introduce the differential evolution algorithm to search for the near-optimal areas of the object to attack, improving the adversarial performance under certain attack area limitations. Besides, we also study the performance of adaptive DE_DAC, which can be adapted to the environment. Experiments show that our proposed method could obtain a good trade-off between the fooling human eyes and object detectors under multiple specific scenes and objects.



## **33. ODG-Q: Robust Quantization via Online Domain Generalization**

cs.LG

**SubmitDate**: 2022-10-17    [abs](http://arxiv.org/abs/2210.08701v1) [paper-pdf](http://arxiv.org/pdf/2210.08701v1)

**Authors**: Chaofan Tao, Ngai Wong

**Abstract**: Quantizing neural networks to low-bitwidth is important for model deployment on resource-limited edge hardware. Although a quantized network has a smaller model size and memory footprint, it is fragile to adversarial attacks. However, few methods study the robustness and training efficiency of quantized networks. To this end, we propose a new method by recasting robust quantization as an online domain generalization problem, termed ODG-Q, which generates diverse adversarial data at a low cost during training. ODG-Q consistently outperforms existing works against various adversarial attacks. For example, on CIFAR-10 dataset, ODG-Q achieves 49.2% average improvements under five common white-box attacks and 21.7% average improvements under five common black-box attacks, with a training cost similar to that of natural training (viz. without adversaries). To our best knowledge, this work is the first work that trains both quantized and binary neural networks on ImageNet that consistently improve robustness under different attacks. We also provide a theoretical insight of ODG-Q that accounts for the bound of model risk on attacked data.



## **34. A2: Efficient Automated Attacker for Boosting Adversarial Training**

cs.CV

Accepted by NeurIPS2022

**SubmitDate**: 2022-10-17    [abs](http://arxiv.org/abs/2210.03543v2) [paper-pdf](http://arxiv.org/pdf/2210.03543v2)

**Authors**: Zhuoer Xu, Guanghui Zhu, Changhua Meng, Shiwen Cui, Zhenzhe Ying, Weiqiang Wang, Ming GU, Yihua Huang

**Abstract**: Based on the significant improvement of model robustness by AT (Adversarial Training), various variants have been proposed to further boost the performance. Well-recognized methods have focused on different components of AT (e.g., designing loss functions and leveraging additional unlabeled data). It is generally accepted that stronger perturbations yield more robust models. However, how to generate stronger perturbations efficiently is still missed. In this paper, we propose an efficient automated attacker called A2 to boost AT by generating the optimal perturbations on-the-fly during training. A2 is a parameterized automated attacker to search in the attacker space for the best attacker against the defense model and examples. Extensive experiments across different datasets demonstrate that A2 generates stronger perturbations with low extra cost and reliably improves the robustness of various AT methods against different attacks.



## **35. Reliability and Robustness analysis of Machine Learning based Phishing URL Detectors**

cs.CR

Accepted in Transactions of Dependable and Secure Computing  (SI-Reliability and Robustness in AI-Based Cybersecurity Solutions)

**SubmitDate**: 2022-10-17    [abs](http://arxiv.org/abs/2005.08454v2) [paper-pdf](http://arxiv.org/pdf/2005.08454v2)

**Authors**: Bushra Sabir, M. Ali Babar, Raj Gaire, Alsharif Abuadbba

**Abstract**: ML-based Phishing URL (MLPU) detectors serve as the first level of defence to protect users and organisations from being victims of phishing attacks. Lately, few studies have launched successful adversarial attacks against specific MLPU detectors raising questions about their practical reliability and usage. Nevertheless, the robustness of these systems has not been extensively investigated. Therefore, the security vulnerabilities of these systems, in general, remain primarily unknown which calls for testing the robustness of these systems. In this article, we have proposed a methodology to investigate the reliability and robustness of 50 representative state-of-the-art MLPU models. Firstly, we have proposed a cost-effective Adversarial URL generator URLBUG that created an Adversarial URL dataset. Subsequently, we reproduced 50 MLPU (traditional ML and Deep learning) systems and recorded their baseline performance. Lastly, we tested the considered MLPU systems on Adversarial Dataset and analyzed their robustness and reliability using box plots and heat maps. Our results showed that the generated adversarial URLs have valid syntax and can be registered at a median annual price of \$11.99. Out of 13\% of the already registered adversarial URLs, 63.94\% were used for malicious purposes. Moreover, the considered MLPU models Matthew Correlation Coefficient (MCC) dropped from a median 0.92 to 0.02 when tested against $Adv_\mathrm{data}$, indicating that the baseline MLPU models are unreliable in their current form. Further, our findings identified several security vulnerabilities of these systems and provided future directions for researchers to design dependable and secure MLPU systems.



## **36. Post-breach Recovery: Protection against White-box Adversarial Examples for Leaked DNN Models**

cs.CR

**SubmitDate**: 2022-10-16    [abs](http://arxiv.org/abs/2205.10686v2) [paper-pdf](http://arxiv.org/pdf/2205.10686v2)

**Authors**: Shawn Shan, Wenxin Ding, Emily Wenger, Haitao Zheng, Ben Y. Zhao

**Abstract**: Server breaches are an unfortunate reality on today's Internet. In the context of deep neural network (DNN) models, they are particularly harmful, because a leaked model gives an attacker "white-box" access to generate adversarial examples, a threat model that has no practical robust defenses. For practitioners who have invested years and millions into proprietary DNNs, e.g. medical imaging, this seems like an inevitable disaster looming on the horizon.   In this paper, we consider the problem of post-breach recovery for DNN models. We propose Neo, a new system that creates new versions of leaked models, alongside an inference time filter that detects and removes adversarial examples generated on previously leaked models. The classification surfaces of different model versions are slightly offset (by introducing hidden distributions), and Neo detects the overfitting of attacks to the leaked model used in its generation. We show that across a variety of tasks and attack methods, Neo is able to filter out attacks from leaked models with very high accuracy, and provides strong protection (7--10 recoveries) against attackers who repeatedly breach the server. Neo performs well against a variety of strong adaptive attacks, dropping slightly in # of breaches recoverable, and demonstrates potential as a complement to DNN defenses in the wild.



## **37. Robust Feature-Level Adversaries are Interpretability Tools**

cs.LG

Code available at  https://github.com/thestephencasper/feature_level_adv

**SubmitDate**: 2022-10-16    [abs](http://arxiv.org/abs/2110.03605v5) [paper-pdf](http://arxiv.org/pdf/2110.03605v5)

**Authors**: Stephen Casper, Max Nadeau, Dylan Hadfield-Menell, Gabriel Kreiman

**Abstract**: The literature on adversarial attacks in computer vision typically focuses on pixel-level perturbations. These tend to be very difficult to interpret. Recent work that manipulates the latent representations of image generators to create "feature-level" adversarial perturbations gives us an opportunity to explore perceptible, interpretable adversarial attacks. We make three contributions. First, we observe that feature-level attacks provide useful classes of inputs for studying representations in models. Second, we show that these adversaries are versatile and highly robust. We demonstrate that they can be used to produce targeted, universal, disguised, physically-realizable, and black-box attacks at the ImageNet scale. Third, we show how these adversarial images can be used as a practical interpretability tool for identifying bugs in networks. We use these adversaries to make predictions about spurious associations between features and classes which we then test by designing "copy/paste" attacks in which one natural image is pasted into another to cause a targeted misclassification. Our results suggest that feature-level attacks are a promising approach for rigorous interpretability research. They support the design of tools to better understand what a model has learned and diagnose brittle feature associations. Code is available at https://github.com/thestephencasper/feature_level_adv



## **38. Nowhere to Hide: A Lightweight Unsupervised Detector against Adversarial Examples**

cs.LG

**SubmitDate**: 2022-10-16    [abs](http://arxiv.org/abs/2210.08579v1) [paper-pdf](http://arxiv.org/pdf/2210.08579v1)

**Authors**: Hui Liu, Bo Zhao, Kehuan Zhang, Peng Liu

**Abstract**: Although deep neural networks (DNNs) have shown impressive performance on many perceptual tasks, they are vulnerable to adversarial examples that are generated by adding slight but maliciously crafted perturbations to benign images. Adversarial detection is an important technique for identifying adversarial examples before they are entered into target DNNs. Previous studies to detect adversarial examples either targeted specific attacks or required expensive computation. How design a lightweight unsupervised detector is still a challenging problem. In this paper, we propose an AutoEncoder-based Adversarial Examples (AEAE) detector, that can guard DNN models by detecting adversarial examples with low computation in an unsupervised manner. The AEAE includes only a shallow autoencoder but plays two roles. First, a well-trained autoencoder has learned the manifold of benign examples. This autoencoder can produce a large reconstruction error for adversarial images with large perturbations, so we can detect significantly perturbed adversarial examples based on the reconstruction error. Second, the autoencoder can filter out the small noise and change the DNN's prediction on adversarial examples with small perturbations. It helps to detect slightly perturbed adversarial examples based on the prediction distance. To cover these two cases, we utilize the reconstruction error and prediction distance from benign images to construct a two-tuple feature set and train an adversarial detector using the isolation forest algorithm. We show empirically that the AEAE is unsupervised and inexpensive against the most state-of-the-art attacks. Through the detection in these two cases, there is nowhere to hide adversarial examples.



## **39. Object-Attentional Untargeted Adversarial Attack**

cs.CV

**SubmitDate**: 2022-10-16    [abs](http://arxiv.org/abs/2210.08472v1) [paper-pdf](http://arxiv.org/pdf/2210.08472v1)

**Authors**: Chao Zhou, Yuan-Gen Wang, Guopu Zhu

**Abstract**: Deep neural networks are facing severe threats from adversarial attacks. Most existing black-box attacks fool target model by generating either global perturbations or local patches. However, both global perturbations and local patches easily cause annoying visual artifacts in adversarial example. Compared with some smooth regions of an image, the object region generally has more edges and a more complex texture. Thus small perturbations on it will be more imperceptible. On the other hand, the object region is undoubtfully the decisive part of an image to classification tasks. Motivated by these two facts, we propose an object-attentional adversarial attack method for untargeted attack. Specifically, we first generate an object region by intersecting the object detection region from YOLOv4 with the salient object detection (SOD) region from HVPNet. Furthermore, we design an activation strategy to avoid the reaction caused by the incomplete SOD. Then, we perform an adversarial attack only on the detected object region by leveraging Simple Black-box Adversarial Attack (SimBA). To verify the proposed method, we create a unique dataset by extracting all the images containing the object defined by COCO from ImageNet-1K, named COCO-Reduced-ImageNet in this paper. Experimental results on ImageNet-1K and COCO-Reduced-ImageNet show that under various system settings, our method yields the adversarial example with better perceptual quality meanwhile saving the query budget up to 24.16\% compared to the state-of-the-art approaches including SimBA.



## **40. RoS-KD: A Robust Stochastic Knowledge Distillation Approach for Noisy Medical Imaging**

cs.CV

Accepted in ICDM 2022

**SubmitDate**: 2022-10-15    [abs](http://arxiv.org/abs/2210.08388v1) [paper-pdf](http://arxiv.org/pdf/2210.08388v1)

**Authors**: Ajay Jaiswal, Kumar Ashutosh, Justin F Rousseau, Yifan Peng, Zhangyang Wang, Ying Ding

**Abstract**: AI-powered Medical Imaging has recently achieved enormous attention due to its ability to provide fast-paced healthcare diagnoses. However, it usually suffers from a lack of high-quality datasets due to high annotation cost, inter-observer variability, human annotator error, and errors in computer-generated labels. Deep learning models trained on noisy labelled datasets are sensitive to the noise type and lead to less generalization on the unseen samples. To address this challenge, we propose a Robust Stochastic Knowledge Distillation (RoS-KD) framework which mimics the notion of learning a topic from multiple sources to ensure deterrence in learning noisy information. More specifically, RoS-KD learns a smooth, well-informed, and robust student manifold by distilling knowledge from multiple teachers trained on overlapping subsets of training data. Our extensive experiments on popular medical imaging classification tasks (cardiopulmonary disease and lesion classification) using real-world datasets, show the performance benefit of RoS-KD, its ability to distill knowledge from many popular large networks (ResNet-50, DenseNet-121, MobileNet-V2) in a comparatively small network, and its robustness to adversarial attacks (PGD, FSGM). More specifically, RoS-KD achieves >2% and >4% improvement on F1-score for lesion classification and cardiopulmonary disease classification tasks, respectively, when the underlying student is ResNet-18 against recent competitive knowledge distillation baseline. Additionally, on cardiopulmonary disease classification task, RoS-KD outperforms most of the SOTA baselines by ~1% gain in AUC score.



## **41. GAMA: Generative Adversarial Multi-Object Scene Attacks**

cs.CV

Accepted at NeurIPS 2022; First two authors contributed equally;  Includes Supplementary Material

**SubmitDate**: 2022-10-15    [abs](http://arxiv.org/abs/2209.09502v2) [paper-pdf](http://arxiv.org/pdf/2209.09502v2)

**Authors**: Abhishek Aich, Calvin-Khang Ta, Akash Gupta, Chengyu Song, Srikanth V. Krishnamurthy, M. Salman Asif, Amit K. Roy-Chowdhury

**Abstract**: The majority of methods for crafting adversarial attacks have focused on scenes with a single dominant object (e.g., images from ImageNet). On the other hand, natural scenes include multiple dominant objects that are semantically related. Thus, it is crucial to explore designing attack strategies that look beyond learning on single-object scenes or attack single-object victim classifiers. Due to their inherent property of strong transferability of perturbations to unknown models, this paper presents the first approach of using generative models for adversarial attacks on multi-object scenes. In order to represent the relationships between different objects in the input scene, we leverage upon the open-sourced pre-trained vision-language model CLIP (Contrastive Language-Image Pre-training), with the motivation to exploit the encoded semantics in the language space along with the visual space. We call this attack approach Generative Adversarial Multi-object scene Attacks (GAMA). GAMA demonstrates the utility of the CLIP model as an attacker's tool to train formidable perturbation generators for multi-object scenes. Using the joint image-text features to train the generator, we show that GAMA can craft potent transferable perturbations in order to fool victim classifiers in various attack settings. For example, GAMA triggers ~16% more misclassification than state-of-the-art generative approaches in black-box settings where both the classifier architecture and data distribution of the attacker are different from the victim. Our code is available here: https://abhishekaich27.github.io/gama.html



## **42. A Scalable Reinforcement Learning Approach for Attack Allocation in Swarm to Swarm Engagement Problems**

cs.RO

submitted to ICRA 2023

**SubmitDate**: 2022-10-15    [abs](http://arxiv.org/abs/2210.08319v1) [paper-pdf](http://arxiv.org/pdf/2210.08319v1)

**Authors**: Umut Demir, Nazim Kemal Ure

**Abstract**: In this work we propose a reinforcement learning (RL) framework that controls the density of a large-scale swarm for engaging with adversarial swarm attacks. Although there is a significant amount of existing work in applying artificial intelligence methods to swarm control, analysis of interactions between two adversarial swarms is a rather understudied area. Most of the existing work in this subject develop strategies by making hard assumptions regarding the strategy and dynamics of the adversarial swarm. Our main contribution is the formulation of the swarm to swarm engagement problem as a Markov Decision Process and development of RL algorithms that can compute engagement strategies without the knowledge of strategy/dynamics of the adversarial swarm. Simulation results show that the developed framework can handle a wide array of large-scale engagement scenarios in an efficient manner.



## **43. Robust Binary Models by Pruning Randomly-initialized Networks**

cs.LG

Accepted as NeurIPS 2022 paper

**SubmitDate**: 2022-10-15    [abs](http://arxiv.org/abs/2202.01341v2) [paper-pdf](http://arxiv.org/pdf/2202.01341v2)

**Authors**: Chen Liu, Ziqi Zhao, Sabine Süsstrunk, Mathieu Salzmann

**Abstract**: Robustness to adversarial attacks was shown to require a larger model capacity, and thus a larger memory footprint. In this paper, we introduce an approach to obtain robust yet compact models by pruning randomly-initialized binary networks. Unlike adversarial training, which learns the model parameters, we initialize the model parameters as either +1 or -1, keep them fixed, and find a subnetwork structure that is robust to attacks. Our method confirms the Strong Lottery Ticket Hypothesis in the presence of adversarial attacks, and extends this to binary networks. Furthermore, it yields more compact networks with competitive performance than existing works by 1) adaptively pruning different network layers; 2) exploiting an effective binary initialization scheme; 3) incorporating a last batch normalization layer to improve training stability. Our experiments demonstrate that our approach not only always outperforms the state-of-the-art robust binary networks, but also can achieve accuracy better than full-precision ones on some datasets. Finally, we show the structured patterns of our pruned binary networks.



## **44. Overparameterization from Computational Constraints**

cs.LG

**SubmitDate**: 2022-10-15    [abs](http://arxiv.org/abs/2208.12926v2) [paper-pdf](http://arxiv.org/pdf/2208.12926v2)

**Authors**: Sanjam Garg, Somesh Jha, Saeed Mahloujifar, Mohammad Mahmoody, Mingyuan Wang

**Abstract**: Overparameterized models with millions of parameters have been hugely successful. In this work, we ask: can the need for large models be, at least in part, due to the \emph{computational} limitations of the learner? Additionally, we ask, is this situation exacerbated for \emph{robust} learning? We show that this indeed could be the case. We show learning tasks for which computationally bounded learners need \emph{significantly more} model parameters than what information-theoretic learners need. Furthermore, we show that even more model parameters could be necessary for robust learning. In particular, for computationally bounded learners, we extend the recent result of Bubeck and Sellke [NeurIPS'2021] which shows that robust models might need more parameters, to the computational regime and show that bounded learners could provably need an even larger number of parameters. Then, we address the following related question: can we hope to remedy the situation for robust computationally bounded learning by restricting \emph{adversaries} to also be computationally bounded for sake of obtaining models with fewer parameters? Here again, we show that this could be possible. Specifically, building on the work of Garg, Jha, Mahloujifar, and Mahmoody [ALT'2020], we demonstrate a learning task that can be learned efficiently and robustly against a computationally bounded attacker, while to be robust against an information-theoretic attacker requires the learner to utilize significantly more parameters.



## **45. Distributionally Robust Multiclass Classification and Applications in Deep Image Classifiers**

cs.CV

arXiv admin note: substantial text overlap with arXiv:2109.12772

**SubmitDate**: 2022-10-15    [abs](http://arxiv.org/abs/2210.08198v1) [paper-pdf](http://arxiv.org/pdf/2210.08198v1)

**Authors**: Ruidi Chen, Boran Hao, Ioannis Ch. Paschalidis

**Abstract**: We develop a Distributionally Robust Optimization (DRO) formulation for Multiclass Logistic Regression (MLR), which could tolerate data contaminated by outliers. The DRO framework uses a probabilistic ambiguity set defined as a ball of distributions that are close to the empirical distribution of the training set in the sense of the Wasserstein metric. We relax the DRO formulation into a regularized learning problem whose regularizer is a norm of the coefficient matrix. We establish out-of-sample performance guarantees for the solutions to our model, offering insights on the role of the regularizer in controlling the prediction error. We apply the proposed method in rendering deep Vision Transformer (ViT)-based image classifiers robust to random and adversarial attacks. Specifically, using the MNIST and CIFAR-10 datasets, we demonstrate reductions in test error rate by up to 83.5% and loss by up to 91.3% compared with baseline methods, by adopting a novel random training method.



## **46. Dynamics-aware Adversarial Attack of Adaptive Neural Networks**

cs.CV

**SubmitDate**: 2022-10-15    [abs](http://arxiv.org/abs/2210.08159v1) [paper-pdf](http://arxiv.org/pdf/2210.08159v1)

**Authors**: An Tao, Yueqi Duan, Yingqi Wang, Jiwen Lu, Jie Zhou

**Abstract**: In this paper, we investigate the dynamics-aware adversarial attack problem of adaptive neural networks. Most existing adversarial attack algorithms are designed under a basic assumption -- the network architecture is fixed throughout the attack process. However, this assumption does not hold for many recently proposed adaptive neural networks, which adaptively deactivate unnecessary execution units based on inputs to improve computational efficiency. It results in a serious issue of lagged gradient, making the learned attack at the current step ineffective due to the architecture change afterward. To address this issue, we propose a Leaded Gradient Method (LGM) and show the significant effects of the lagged gradient. More specifically, we reformulate the gradients to be aware of the potential dynamic changes of network architectures, so that the learned attack better "leads" the next step than the dynamics-unaware methods when network architecture changes dynamically. Extensive experiments on representative types of adaptive neural networks for both 2D images and 3D point clouds show that our LGM achieves impressive adversarial attack performance compared with the dynamic-unaware attack methods.



## **47. Certified Robustness Against Natural Language Attacks by Causal Intervention**

cs.LG

**SubmitDate**: 2022-10-14    [abs](http://arxiv.org/abs/2205.12331v3) [paper-pdf](http://arxiv.org/pdf/2205.12331v3)

**Authors**: Haiteng Zhao, Chang Ma, Xinshuai Dong, Anh Tuan Luu, Zhi-Hong Deng, Hanwang Zhang

**Abstract**: Deep learning models have achieved great success in many fields, yet they are vulnerable to adversarial examples. This paper follows a causal perspective to look into the adversarial vulnerability and proposes Causal Intervention by Semantic Smoothing (CISS), a novel framework towards robustness against natural language attacks. Instead of merely fitting observational data, CISS learns causal effects p(y|do(x)) by smoothing in the latent semantic space to make robust predictions, which scales to deep architectures and avoids tedious construction of noise customized for specific attacks. CISS is provably robust against word substitution attacks, as well as empirically robust even when perturbations are strengthened by unknown attack algorithms. For example, on YELP, CISS surpasses the runner-up by 6.7% in terms of certified robustness against word substitutions, and achieves 79.4% empirical robustness when syntactic attacks are integrated.



## **48. SealClub: Computer-aided Paper Document Authentication**

cs.CR

**SubmitDate**: 2022-10-14    [abs](http://arxiv.org/abs/2210.07884v1) [paper-pdf](http://arxiv.org/pdf/2210.07884v1)

**Authors**: Martín Ochoa, Jorge Toro-Pozo, David Basin

**Abstract**: Digital authentication is a mature field, offering a range of solutions with rigorous mathematical guarantees. Nevertheless, paper documents, where cryptographic techniques are not directly applicable, are still widely utilized due to usability and legal reasons. We propose a novel approach to authenticating paper documents using smartphones by taking short videos of them. Our solution combines cryptographic and image comparison techniques to detect and highlight subtle semantic-changing attacks on rich documents, containing text and graphics, that could go unnoticed by humans. We rigorously analyze our approach, proving that it is secure against strong adversaries capable of compromising different system components. We also measure its accuracy empirically on a set of 128 videos of paper documents, half containing subtle forgeries. Our algorithm finds all forgeries accurately (no false alarms) after analyzing 5.13 frames on average (corresponding to 1.28 seconds of video). Highlighted regions are large enough to be visible to users, but small enough to precisely locate forgeries. Thus, our approach provides a promising way for users to authenticate paper documents using conventional smartphones under realistic conditions.



## **49. Pre-trained Adversarial Perturbations**

cs.CV

**SubmitDate**: 2022-10-14    [abs](http://arxiv.org/abs/2210.03372v2) [paper-pdf](http://arxiv.org/pdf/2210.03372v2)

**Authors**: Yuanhao Ban, Yinpeng Dong

**Abstract**: Self-supervised pre-training has drawn increasing attention in recent years due to its superior performance on numerous downstream tasks after fine-tuning. However, it is well-known that deep learning models lack the robustness to adversarial examples, which can also invoke security issues to pre-trained models, despite being less explored. In this paper, we delve into the robustness of pre-trained models by introducing Pre-trained Adversarial Perturbations (PAPs), which are universal perturbations crafted for the pre-trained models to maintain the effectiveness when attacking fine-tuned ones without any knowledge of the downstream tasks. To this end, we propose a Low-Level Layer Lifting Attack (L4A) method to generate effective PAPs by lifting the neuron activations of low-level layers of the pre-trained models. Equipped with an enhanced noise augmentation strategy, L4A is effective at generating more transferable PAPs against fine-tuned models. Extensive experiments on typical pre-trained vision models and ten downstream tasks demonstrate that our method improves the attack success rate by a large margin compared with state-of-the-art methods.



## **50. Generative Adversarial Learning for Trusted and Secure Clustering in Industrial Wireless Sensor Networks**

cs.NI

**SubmitDate**: 2022-10-14    [abs](http://arxiv.org/abs/2210.07707v1) [paper-pdf](http://arxiv.org/pdf/2210.07707v1)

**Authors**: Liu Yang, Simon X. Yang, Yun Li, Yinzhi Lu, Tan Guo

**Abstract**: Traditional machine learning techniques have been widely used to establish the trust management systems. However, the scale of training dataset can significantly affect the security performances of the systems, while it is a great challenge to detect malicious nodes due to the absence of labeled data regarding novel attacks. To address this issue, this paper presents a generative adversarial network (GAN) based trust management mechanism for Industrial Wireless Sensor Networks (IWSNs). First, type-2 fuzzy logic is adopted to evaluate the reputation of sensor nodes while alleviating the uncertainty problem. Then, trust vectors are collected to train a GAN-based codec structure, which is used for further malicious node detection. Moreover, to avoid normal nodes being isolated from the network permanently due to error detections, a GAN-based trust redemption model is constructed to enhance the resilience of trust management. Based on the latest detection results, a trust model update method is developed to adapt to the dynamic industrial environment. The proposed trust management mechanism is finally applied to secure clustering for reliable and real-time data transmission, and simulation results show that it achieves a high detection rate up to 96%, as well as a low false positive rate below 8%.



