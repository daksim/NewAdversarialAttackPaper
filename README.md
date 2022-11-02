# Latest Adversarial Attack Papers
**update at 2022-11-03 06:31:34**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. Improving Hyperspectral Adversarial Robustness using Ensemble Networks in the Presences of Multiple Attacks**

cs.LG

6 pages, 2 figures, 1 table, 1 algorithm

**SubmitDate**: 2022-11-01    [abs](http://arxiv.org/abs/2210.16346v2) [paper-pdf](http://arxiv.org/pdf/2210.16346v2)

**Authors**: Nicholas Soucy, Salimeh Yasaei Sekeh

**Abstract**: Semantic segmentation of hyperspectral images (HSI) has seen great strides in recent years by incorporating knowledge from deep learning RGB classification models. Similar to their classification counterparts, semantic segmentation models are vulnerable to adversarial examples and need adversarial training to counteract them. Traditional approaches to adversarial robustness focus on training or retraining a single network on attacked data, however, in the presence of multiple attacks these approaches decrease the performance compared to networks trained individually on each attack. To combat this issue we propose an Adversarial Discriminator Ensemble Network (ADE-Net) which focuses on attack type detection and adversarial robustness under a unified model to preserve per data-type weight optimally while robustifiying the overall network. In the proposed method, a discriminator network is used to separate data by attack type into their specific attack-expert ensemble network. Our approach allows for the presence of multiple attacks mixed together while also labeling attack types during testing. We experimentally show that ADE-Net outperforms the baseline, which is a single network adversarially trained under a mix of multiple attacks, for HSI Indian Pines, Kennedy Space, and Houston datasets.



## **2. A Unified Evaluation of Textual Backdoor Learning: Frameworks and Benchmarks**

cs.LG

NeurIPS 2022 Datasets & Benchmarks; Toolkits avaliable at  https://github.com/thunlp/OpenBackdoor

**SubmitDate**: 2022-11-01    [abs](http://arxiv.org/abs/2206.08514v2) [paper-pdf](http://arxiv.org/pdf/2206.08514v2)

**Authors**: Ganqu Cui, Lifan Yuan, Bingxiang He, Yangyi Chen, Zhiyuan Liu, Maosong Sun

**Abstract**: Textual backdoor attacks are a kind of practical threat to NLP systems. By injecting a backdoor in the training phase, the adversary could control model predictions via predefined triggers. As various attack and defense models have been proposed, it is of great significance to perform rigorous evaluations. However, we highlight two issues in previous backdoor learning evaluations: (1) The differences between real-world scenarios (e.g. releasing poisoned datasets or models) are neglected, and we argue that each scenario has its own constraints and concerns, thus requires specific evaluation protocols; (2) The evaluation metrics only consider whether the attacks could flip the models' predictions on poisoned samples and retain performances on benign samples, but ignore that poisoned samples should also be stealthy and semantic-preserving. To address these issues, we categorize existing works into three practical scenarios in which attackers release datasets, pre-trained models, and fine-tuned models respectively, then discuss their unique evaluation methodologies. On metrics, to completely evaluate poisoned samples, we use grammar error increase and perplexity difference for stealthiness, along with text similarity for validity. After formalizing the frameworks, we develop an open-source toolkit OpenBackdoor to foster the implementations and evaluations of textual backdoor learning. With this toolkit, we perform extensive experiments to benchmark attack and defense models under the suggested paradigm. To facilitate the underexplored defenses against poisoned datasets, we further propose CUBE, a simple yet strong clustering-based defense baseline. We hope that our frameworks and benchmarks could serve as the cornerstones for future model development and evaluations.



## **3. A Comprehensive Evaluation Framework for Deep Model Robustness**

cs.CV

Submitted to Pattern Recognition

**SubmitDate**: 2022-11-01    [abs](http://arxiv.org/abs/2101.09617v2) [paper-pdf](http://arxiv.org/pdf/2101.09617v2)

**Authors**: Jun Guo, Wei Bao, Jiakai Wang, Yuqing Ma, Xinghai Gao, Gang Xiao, Aishan Liu, Jian Dong, Xianglong Liu, Wenjun Wu

**Abstract**: Deep neural networks (DNNs) have achieved remarkable performance across a wide range of applications, while they are vulnerable to adversarial examples, which motivates the evaluation and benchmark of model robustness. However, current evaluations usually use simple metrics to study the performance of defenses, which are far from understanding the limitation and weaknesses of these defense methods. Thus, most proposed defenses are quickly shown to be attacked successfully, which results in the ``arm race'' phenomenon between attack and defense. To mitigate this problem, we establish a model robustness evaluation framework containing 23 comprehensive and rigorous metrics, which consider two key perspectives of adversarial learning (i.e., data and model). Through neuron coverage and data imperceptibility, we use data-oriented metrics to measure the integrity of test examples; by delving into model structure and behavior, we exploit model-oriented metrics to further evaluate robustness in the adversarial setting. To fully demonstrate the effectiveness of our framework, we conduct large-scale experiments on multiple datasets including CIFAR-10, SVHN, and ImageNet using different models and defenses with our open-source platform. Overall, our paper provides a comprehensive evaluation framework, where researchers could conduct comprehensive and fast evaluations using the open-source toolkit, and the analytical results could inspire deeper understanding and further improvement to the model robustness.



## **4. The Perils of Learning From Unlabeled Data: Backdoor Attacks on Semi-supervised Learning**

cs.CR

**SubmitDate**: 2022-11-01    [abs](http://arxiv.org/abs/2211.00453v1) [paper-pdf](http://arxiv.org/pdf/2211.00453v1)

**Authors**: Virat Shejwalkar, Lingjuan Lyu, Amir Houmansadr

**Abstract**: Semi-supervised machine learning (SSL) is gaining popularity as it reduces the cost of training ML models. It does so by using very small amounts of (expensive, well-inspected) labeled data and large amounts of (cheap, non-inspected) unlabeled data. SSL has shown comparable or even superior performances compared to conventional fully-supervised ML techniques.   In this paper, we show that the key feature of SSL that it can learn from (non-inspected) unlabeled data exposes SSL to strong poisoning attacks. In fact, we argue that, due to its reliance on non-inspected unlabeled data, poisoning is a much more severe problem in SSL than in conventional fully-supervised ML.   Specifically, we design a backdoor poisoning attack on SSL that can be conducted by a weak adversary with no knowledge of target SSL pipeline. This is unlike prior poisoning attacks in fully-supervised settings that assume strong adversaries with practically-unrealistic capabilities. We show that by poisoning only 0.2% of the unlabeled training data, our attack can cause misclassification of more than 80% of test inputs (when they contain the adversary's backdoor trigger). Our attacks remain effective across twenty combinations of benchmark datasets and SSL algorithms, and even circumvent the state-of-the-art defenses against backdoor attacks. Our work raises significant concerns about the practical utility of existing SSL algorithms.



## **5. Relative Attention-based One-Class Adversarial Autoencoder for Continuous Authentication of Smartphone Users**

cs.HC

**SubmitDate**: 2022-11-01    [abs](http://arxiv.org/abs/2210.16819v2) [paper-pdf](http://arxiv.org/pdf/2210.16819v2)

**Authors**: Mingming Hu, Kun Zhang, Ruibang You, Bibo Tu

**Abstract**: Behavioral biometrics-based continuous authentication is a promising authentication scheme, which uses behavioral biometrics recorded by built-in sensors to authenticate smartphone users throughout the session. However, current continuous authentication methods suffer some limitations: 1) behavioral biometrics from impostors are needed to train continuous authentication models. Since the distribution of negative samples from diverse attackers are unknown, it is a difficult problem to solve in real-world scenarios; 2) most deep learning-based continuous authentication methods need to train two models to improve authentication performance. A deep learning model for deep feature extraction, and a machine learning-based classifier for classification; 3) weak capability of capturing users' behavioral patterns leads to poor authentication performance. To solve these issues, we propose a relative attention-based one-class adversarial autoencoder for continuous authentication of smartphone users. First, we propose a one-class adversarial autoencoder to learn latent representations of legitimate users' behavioral patterns, which is trained only with legitimate smartphone users' behavioral biometrics. Second, we present the relative attention layer to capture richer contextual semantic representation of users' behavioral patterns, which modifies the standard self-attention mechanism using convolution projection instead of linear projection to perform the attention maps. Experimental results demonstrate that we can achieve superior performance of 1.05% EER, 1.09% EER, and 1.08% EER with a high authentication frequency (0.7s) on three public datasets.



## **6. Universal Perturbation Attack on Differentiable No-Reference Image- and Video-Quality Metrics**

cs.CV

BMVC 2022

**SubmitDate**: 2022-11-01    [abs](http://arxiv.org/abs/2211.00366v1) [paper-pdf](http://arxiv.org/pdf/2211.00366v1)

**Authors**: Ekaterina Shumitskaya, Anastasia Antsiferova, Dmitriy Vatolin

**Abstract**: Universal adversarial perturbation attacks are widely used to analyze image classifiers that employ convolutional neural networks. Nowadays, some attacks can deceive image- and video-quality metrics. So sustainability analysis of these metrics is important. Indeed, if an attack can confuse the metric, an attacker can easily increase quality scores. When developers of image- and video-algorithms can boost their scores through detached processing, algorithm comparisons are no longer fair. Inspired by the idea of universal adversarial perturbation for classifiers, we suggest a new method to attack differentiable no-reference quality metrics through universal perturbation. We applied this method to seven no-reference image- and video-quality metrics (PaQ-2-PiQ, Linearity, VSFA, MDTVSFA, KonCept512, Nima and SPAQ). For each one, we trained a universal perturbation that increases the respective scores. We also propose a method for assessing metric stability and identify the metrics that are the most vulnerable and the most resistant to our attack. The existence of successful universal perturbations appears to diminish the metric's ability to provide reliable scores. We therefore recommend our proposed method as an additional verification of metric reliability to complement traditional subjective tests and benchmarks.



## **7. FRSUM: Towards Faithful Abstractive Summarization via Enhancing Factual Robustness**

cs.CL

Findings of EMNLP 2022

**SubmitDate**: 2022-11-01    [abs](http://arxiv.org/abs/2211.00294v1) [paper-pdf](http://arxiv.org/pdf/2211.00294v1)

**Authors**: Wenhao Wu, Wei Li, Jiachen Liu, Xinyan Xiao, Ziqiang Cao, Sujian Li, Hua Wu

**Abstract**: Despite being able to generate fluent and grammatical text, current Seq2Seq summarization models still suffering from the unfaithful generation problem. In this paper, we study the faithfulness of existing systems from a new perspective of factual robustness which is the ability to correctly generate factual information over adversarial unfaithful information. We first measure a model's factual robustness by its success rate to defend against adversarial attacks when generating factual information. The factual robustness analysis on a wide range of current systems shows its good consistency with human judgments on faithfulness. Inspired by these findings, we propose to improve the faithfulness of a model by enhancing its factual robustness. Specifically, we propose a novel training strategy, namely FRSUM, which teaches the model to defend against both explicit adversarial samples and implicit factual adversarial perturbations. Extensive automatic and human evaluation results show that FRSUM consistently improves the faithfulness of various Seq2Seq models, such as T5, BART.



## **8. Adversarial Training with Complementary Labels: On the Benefit of Gradually Informative Attacks**

cs.LG

NeurIPS 2022

**SubmitDate**: 2022-11-01    [abs](http://arxiv.org/abs/2211.00269v1) [paper-pdf](http://arxiv.org/pdf/2211.00269v1)

**Authors**: Jianan Zhou, Jianing Zhu, Jingfeng Zhang, Tongliang Liu, Gang Niu, Bo Han, Masashi Sugiyama

**Abstract**: Adversarial training (AT) with imperfect supervision is significant but receives limited attention. To push AT towards more practical scenarios, we explore a brand new yet challenging setting, i.e., AT with complementary labels (CLs), which specify a class that a data sample does not belong to. However, the direct combination of AT with existing methods for CLs results in consistent failure, but not on a simple baseline of two-stage training. In this paper, we further explore the phenomenon and identify the underlying challenges of AT with CLs as intractable adversarial optimization and low-quality adversarial examples. To address the above problems, we propose a new learning strategy using gradually informative attacks, which consists of two critical components: 1) Warm-up Attack (Warm-up) gently raises the adversarial perturbation budgets to ease the adversarial optimization with CLs; 2) Pseudo-Label Attack (PLA) incorporates the progressively informative model predictions into a corrected complementary loss. Extensive experiments are conducted to demonstrate the effectiveness of our method on a range of benchmarked datasets. The code is publicly available at: https://github.com/RoyalSkye/ATCL.



## **9. Adversarial Policies Beat Professional-Level Go AIs**

cs.LG

21 pages, 11 figures

**SubmitDate**: 2022-11-01    [abs](http://arxiv.org/abs/2211.00241v1) [paper-pdf](http://arxiv.org/pdf/2211.00241v1)

**Authors**: Tony Tong Wang, Adam Gleave, Nora Belrose, Tom Tseng, Joseph Miller, Michael D Dennis, Yawen Duan, Viktor Pogrebniak, Sergey Levine, Stuart Russell

**Abstract**: We attack the state-of-the-art Go-playing AI system, KataGo, by training an adversarial policy that plays against a frozen KataGo victim. Our attack achieves a >99% win-rate against KataGo without search, and a >50% win-rate when KataGo uses enough search to be near-superhuman. To the best of our knowledge, this is the first successful end-to-end attack against a Go AI playing at the level of a top human professional. Notably, the adversary does not win by learning to play Go better than KataGo -- in fact, the adversary is easily beaten by human amateurs. Instead, the adversary wins by tricking KataGo into ending the game prematurely at a point that is favorable to the adversary. Our results demonstrate that even professional-level AI systems may harbor surprising failure modes. See https://goattack.alignmentfund.org/ for example games.



## **10. Synthetic ID Card Image Generation for Improving Presentation Attack Detection**

cs.CV

**SubmitDate**: 2022-10-31    [abs](http://arxiv.org/abs/2211.00098v1) [paper-pdf](http://arxiv.org/pdf/2211.00098v1)

**Authors**: Daniel Benalcazar, Juan E. Tapia, Sebastian Gonzalez, Christoph Busch

**Abstract**: Currently, it is ever more common to access online services for activities which formerly required physical attendance. From banking operations to visa applications, a significant number of processes have been digitised, especially since the advent of the COVID-19 pandemic, requiring remote biometric authentication of the user. On the downside, some subjects intend to interfere with the normal operation of remote systems for personal profit by using fake identity documents, such as passports and ID cards. Deep learning solutions to detect such frauds have been presented in the literature. However, due to privacy concerns and the sensitive nature of personal identity documents, developing a dataset with the necessary number of examples for training deep neural networks is challenging. This work explores three methods for synthetically generating ID card images to increase the amount of data while training fraud-detection networks. These methods include computer vision algorithms and Generative Adversarial Networks. Our results indicate that databases can be supplemented with synthetic images without any loss in performance for the print/scan Presentation Attack Instrument Species (PAIS) and a loss in performance of 1% for the screen capture PAIS.



## **11. Lessons Learned: How (Not) to Defend Against Property Inference Attacks**

cs.CR

**SubmitDate**: 2022-10-31    [abs](http://arxiv.org/abs/2205.08821v3) [paper-pdf](http://arxiv.org/pdf/2205.08821v3)

**Authors**: Joshua Stock, Jens Wettlaufer, Daniel Demmler, Hannes Federrath

**Abstract**: This work investigates and evaluates multiple defense strategies against property inference attacks (PIAs), a privacy attack against machine learning models. Given a trained machine learning model, PIAs aim to extract statistical properties of its underlying training data, e.g., reveal the ratio of men and women in a medical training data set. While for other privacy attacks like membership inference, a lot of research on defense mechanisms has been published, this is the first work focusing on defending against PIAs. With the primary goal of developing a generic mitigation strategy against white-box PIAs, we propose the novel approach property unlearning. Extensive experiments with property unlearning show that while it is very effective when defending target models against specific adversaries, property unlearning is not able to generalize, i.e., protect against a whole class of PIAs. To investigate the reasons behind this limitation, we present the results of experiments with the explainable AI tool LIME. They show how state-of-the-art property inference adversaries with the same objective focus on different parts of the target model. We further elaborate on this with a follow-up experiment, in which we use the visualization technique t-SNE to exhibit how severely statistical training data properties are manifested in machine learning models. Based on this, we develop the conjecture that post-training techniques like property unlearning might not suffice to provide the desirable generic protection against PIAs. As an alternative, we investigate the effects of simpler training data preprocessing methods like adding Gaussian noise to images of a training data set on the success rate of PIAs. We conclude with a discussion of the different defense approaches, summarize the lessons learned and provide directions for future work.



## **12. Sardino: Ultra-Fast Dynamic Ensemble for Secure Visual Sensing at Mobile Edge**

cs.CV

**SubmitDate**: 2022-10-31    [abs](http://arxiv.org/abs/2204.08189v4) [paper-pdf](http://arxiv.org/pdf/2204.08189v4)

**Authors**: Qun Song, Zhenyu Yan, Wenjie Luo, Rui Tan

**Abstract**: Adversarial example attack endangers the mobile edge systems such as vehicles and drones that adopt deep neural networks for visual sensing. This paper presents {\em Sardino}, an active and dynamic defense approach that renews the inference ensemble at run time to develop security against the adaptive adversary who tries to exfiltrate the ensemble and construct the corresponding effective adversarial examples. By applying consistency check and data fusion on the ensemble's predictions, Sardino can detect and thwart adversarial inputs. Compared with the training-based ensemble renewal, we use HyperNet to achieve {\em one million times} acceleration and per-frame ensemble renewal that presents the highest level of difficulty to the prerequisite exfiltration attacks. We design a run-time planner that maximizes the ensemble size in favor of security while maintaining the processing frame rate. Beyond adversarial examples, Sardino can also address the issue of out-of-distribution inputs effectively. This paper presents extensive evaluation of Sardino's performance in counteracting adversarial examples and applies it to build a real-time car-borne traffic sign recognition system. Live on-road tests show the built system's effectiveness in maintaining frame rate and detecting out-of-distribution inputs due to the false positives of a preceding YOLO-based traffic sign detector.



## **13. Symmetries, flat minima, and the conserved quantities of gradient flow**

cs.LG

Preliminary version; comments welcome

**SubmitDate**: 2022-10-31    [abs](http://arxiv.org/abs/2210.17216v1) [paper-pdf](http://arxiv.org/pdf/2210.17216v1)

**Authors**: Bo Zhao, Iordan Ganev, Robin Walters, Rose Yu, Nima Dehmamy

**Abstract**: Empirical studies of the loss landscape of deep networks have revealed that many local minima are connected through low-loss valleys. Ensemble models sampling different parts of a low-loss valley have reached SOTA performance. Yet, little is known about the theoretical origin of such valleys. We present a general framework for finding continuous symmetries in the parameter space, which carve out low-loss valleys. Importantly, we introduce a novel set of nonlinear, data-dependent symmetries for neural networks. These symmetries can transform a trained model such that it performs similarly on new samples. We then show that conserved quantities associated with linear symmetries can be used to define coordinates along low-loss valleys. The conserved quantities help reveal that using common initialization methods, gradient flow only explores a small part of the global minimum. By relating conserved quantities to convergence rate and sharpness of the minimum, we provide insights on how initialization impacts convergence and generalizability. We also find the nonlinear action to be viable for ensemble building to improve robustness under certain adversarial attacks.



## **14. Defending Against Adversarial Attacks by Energy Storage Facility**

cs.CR

5 pages, 5 main figures. Published in PESGM 2022

**SubmitDate**: 2022-10-31    [abs](http://arxiv.org/abs/2205.09522v2) [paper-pdf](http://arxiv.org/pdf/2205.09522v2)

**Authors**: Jiawei Li, Jianxiao Wang, Lin Chen, Yang Yu

**Abstract**: Adversarial attacks on data-driven algorithms applied in the power system will be a new type of threat to grid security. Literature has demonstrated that the adversarial attack on the deep-neural network can significantly mislead the load fore-cast of a power system. However, it is unclear how the new type of attack impacts the operation of the grid system. In this research, we manifest that the adversarial algorithm attack induces a significant cost-increase risk which will be exacerbated by the growing penetration of intermittent renewable energy. In Texas, a 5% adversarial attack can increase the total generation cost by 17% in a quarter, which accounts for around $20 million. When wind-energy penetration increases to over 40%, the 5% adversarial attack will inflate the genera-tion cost by 23%. Our research discovers a novel approach to defending against the adversarial attack: investing in the energy-storage system. All current literature focuses on developing algorithms to defend against adversarial attacks. We are the first research revealing the capability of using the facility in a physical system to defend against the adversarial algorithm attack in a system of the Internet of Things, such as a smart grid system.



## **15. Scoring Black-Box Models for Adversarial Robustness**

cs.LG

**SubmitDate**: 2022-10-31    [abs](http://arxiv.org/abs/2210.17140v1) [paper-pdf](http://arxiv.org/pdf/2210.17140v1)

**Authors**: Jian Vora, Pranay Reddy Samala

**Abstract**: Deep neural networks are susceptible to adversarial inputs and various methods have been proposed to defend these models against adversarial attacks under different perturbation models. The robustness of models to adversarial attacks has been analyzed by first constructing adversarial inputs for the model, and then testing the model performance on the constructed adversarial inputs. Most of these attacks require the model to be white-box, need access to data labels, and finding adversarial inputs can be computationally expensive. We propose a simple scoring method for black-box models which indicates their robustness to adversarial input. We show that adversarially more robust models have a smaller $l_1$-norm of LIME weights and sharper explanations.



## **16. Character-level White-Box Adversarial Attacks against Transformers via Attachable Subwords Substitution**

cs.CL

13 pages, 3 figures. EMNLP 2022

**SubmitDate**: 2022-10-31    [abs](http://arxiv.org/abs/2210.17004v1) [paper-pdf](http://arxiv.org/pdf/2210.17004v1)

**Authors**: Aiwei Liu, Honghai Yu, Xuming Hu, Shu'ang Li, Li Lin, Fukun Ma, Yawen Yang, Lijie Wen

**Abstract**: We propose the first character-level white-box adversarial attack method against transformer models. The intuition of our method comes from the observation that words are split into subtokens before being fed into the transformer models and the substitution between two close subtokens has a similar effect to the character modification. Our method mainly contains three steps. First, a gradient-based method is adopted to find the most vulnerable words in the sentence. Then we split the selected words into subtokens to replace the origin tokenization result from the transformer tokenizer. Finally, we utilize an adversarial loss to guide the substitution of attachable subtokens in which the Gumbel-softmax trick is introduced to ensure gradient propagation. Meanwhile, we introduce the visual and length constraint in the optimization process to achieve minimum character modifications. Extensive experiments on both sentence-level and token-level tasks demonstrate that our method could outperform the previous attack methods in terms of success rate and edit distance. Furthermore, human evaluation verifies our adversarial examples could preserve their origin labels.



## **17. A Comparative Study of Adversarial Attacks against Point Cloud Semantic Segmentation**

cs.CV

**SubmitDate**: 2022-10-30    [abs](http://arxiv.org/abs/2112.05871v3) [paper-pdf](http://arxiv.org/pdf/2112.05871v3)

**Authors**: Jiacen Xu, Zhe Zhou, Boyuan Feng, Yufei Ding, Zhou Li

**Abstract**: Recent research efforts on 3D point cloud semantic segmentation (PCSS) have achieved outstanding performance by adopting deep CNN (convolutional neural networks) and GCN (graph convolutional networks). However, the robustness of these complex models has not been systematically analyzed. Given that PCSS has been applied in many safety-critical applications (e.g., autonomous driving, geological sensing), it is important to fill this knowledge gap, in particular, how these models are affected under adversarial samples. While adversarial attacks against point clouds have been studied, we found many questions remain about the robustness of PCSS. For instance, all the prior attacks perturb the point coordinates of a point cloud, but the features associated with a point are also leveraged by some PCSS models, and whether they are good targets to attack is unknown yet.   We present a comparative study of PCSS robustness in this work. In particular, we formally define the attacker's objective under targeted attack and non-targeted attack and develop new attacks considering a variety of options, including feature-based and coordinate-based, norm-bounded and norm-unbounded, etc. We conduct evaluations with different combinations of attack options on two datasets (S3DIS and Semantic3D) and three PCSS models (PointNet++, DeepGCNs, and RandLA-Net). We found all of the PCSS models are vulnerable under both targeted and non-targeted attacks, and attacks against point features like color are more effective. With this study, we call the attention of the research community to develop new approaches to harden PCSS models against adversarial attacks.



## **18. Symmetric Saliency-based Adversarial Attack To Speaker Identification**

cs.SD

**SubmitDate**: 2022-10-30    [abs](http://arxiv.org/abs/2210.16777v1) [paper-pdf](http://arxiv.org/pdf/2210.16777v1)

**Authors**: Jiadi Yao, Xing Chen, Xiao-Lei Zhang, Wei-Qiang Zhang, Kunde Yang

**Abstract**: Adversarial attack approaches to speaker identification either need high computational cost or are not very effective, to our knowledge. To address this issue, in this paper, we propose a novel generation-network-based approach, called symmetric saliency-based encoder-decoder (SSED), to generate adversarial voice examples to speaker identification. It contains two novel components. First, it uses a novel saliency map decoder to learn the importance of speech samples to the decision of a targeted speaker identification system, so as to make the attacker focus on generating artificial noise to the important samples. It also proposes an angular loss function to push the speaker embedding far away from the source speaker. Our experimental results demonstrate that the proposed SSED yields the state-of-the-art performance, i.e. over 97% targeted attack success rate and a signal-to-noise level of over 39 dB on both the open-set and close-set speaker identification tasks, with a low computational cost.



## **19. Benchmarking Adversarial Patch Against Aerial Detection**

cs.CV

14 pages, 14 figures

**SubmitDate**: 2022-10-30    [abs](http://arxiv.org/abs/2210.16765v1) [paper-pdf](http://arxiv.org/pdf/2210.16765v1)

**Authors**: Jiawei Lian, Shaohui Mei, Shun Zhang, Mingyang Ma

**Abstract**: DNNs are vulnerable to adversarial examples, which poses great security concerns for security-critical systems. In this paper, a novel adaptive-patch-based physical attack (AP-PA) framework is proposed, which aims to generate adversarial patches that are adaptive in both physical dynamics and varying scales, and by which the particular targets can be hidden from being detected. Furthermore, the adversarial patch is also gifted with attack effectiveness against all targets of the same class with a patch outside the target (No need to smear targeted objects) and robust enough in the physical world. In addition, a new loss is devised to consider more available information of detected objects to optimize the adversarial patch, which can significantly improve the patch's attack efficacy (Average precision drop up to 87.86% and 85.48% in white-box and black-box settings, respectively) and optimizing efficiency. We also establish one of the first comprehensive, coherent, and rigorous benchmarks to evaluate the attack efficacy of adversarial patches on aerial detection tasks. Finally, several proportionally scaled experiments are performed physically to demonstrate that the elaborated adversarial patches can successfully deceive aerial detection algorithms in dynamic physical circumstances. The code is available at https://github.com/JiaweiLian/AP-PA.



## **20. RUSH: Robust Contrastive Learning via Randomized Smoothing**

cs.LG

incomplete validation, the defense strategy will fail when  considering Expectation Over Test (EOT)

**SubmitDate**: 2022-10-30    [abs](http://arxiv.org/abs/2207.05127v2) [paper-pdf](http://arxiv.org/pdf/2207.05127v2)

**Authors**: Yijiang Pang, Boyang Liu, Jiayu Zhou

**Abstract**: Recently, adversarial training has been incorporated in self-supervised contrastive pre-training to augment label efficiency with exciting adversarial robustness. However, the robustness came at a cost of expensive adversarial training. In this paper, we show a surprising fact that contrastive pre-training has an interesting yet implicit connection with robustness, and such natural robustness in the pre trained representation enables us to design a powerful robust algorithm against adversarial attacks, RUSH, that combines the standard contrastive pre-training and randomized smoothing. It boosts both standard accuracy and robust accuracy, and significantly reduces training costs as compared with adversarial training. We use extensive empirical studies to show that the proposed RUSH outperforms robust classifiers from adversarial training, by a significant margin on common benchmarks (CIFAR-10, CIFAR-100, and STL-10) under first-order attacks. In particular, under $\ell_{\infty}$-norm perturbations of size 8/255 PGD attack on CIFAR-10, our model using ResNet-18 as backbone reached 77.8% robust accuracy and 87.9% standard accuracy. Our work has an improvement of over 15% in robust accuracy and a slight improvement in standard accuracy, compared to the state-of-the-arts.



## **21. BERTops: Studying BERT Representations under a Topological Lens**

cs.LG

**SubmitDate**: 2022-10-29    [abs](http://arxiv.org/abs/2205.00953v2) [paper-pdf](http://arxiv.org/pdf/2205.00953v2)

**Authors**: Jatin Chauhan, Manohar Kaul

**Abstract**: Proposing scoring functions to effectively understand, analyze and learn various properties of high dimensional hidden representations of large-scale transformer models like BERT can be a challenging task. In this work, we explore a new direction by studying the topological features of BERT hidden representations using persistent homology (PH). We propose a novel scoring function named "persistence scoring function (PSF)" which: (i) accurately captures the homology of the high-dimensional hidden representations and correlates well with the test set accuracy of a wide range of datasets and outperforms existing scoring metrics, (ii) captures interesting post fine-tuning "per-class" level properties from both qualitative and quantitative viewpoints, (iii) is more stable to perturbations as compared to the baseline functions, which makes it a very robust proxy, and (iv) finally, also serves as a predictor of the attack success rates for a wide category of black-box and white-box adversarial attack methods. Our extensive correlation experiments demonstrate the practical utility of PSF on various NLP tasks relevant to BERT.



## **22. On the Need of Neuromorphic Twins to Detect Denial-of-Service Attacks on Communication Networks**

cs.IT

submitted for publication

**SubmitDate**: 2022-10-29    [abs](http://arxiv.org/abs/2210.16690v1) [paper-pdf](http://arxiv.org/pdf/2210.16690v1)

**Authors**: Holger Boche, Rafael F. Schaefer, H. Vincent Poor, Frank H. P. Fitzek

**Abstract**: As we are more and more dependent on the communication technologies, resilience against any attacks on communication networks is important to guarantee the digital sovereignty of our society. New developments of communication networks tackle the problem of resilience by in-network computing approaches for higher protocol layers, while the physical layer remains an open problem. This is particularly true for wireless communication systems which are inherently vulnerable to adversarial attacks due to the open nature of the wireless medium. In denial-of-service (DoS) attacks, an active adversary is able to completely disrupt the communication and it has been shown that Turing machines are incapable of detecting such attacks. As Turing machines provide the fundamental limits of digital information processing and therewith of digital twins, this implies that even the most powerful digital twins that preserve all information of the physical network error-free are not capable of detecting such attacks. This stimulates the question of how powerful the information processing hardware must be to enable the detection of DoS attacks. Therefore, in the paper the need of neuromorphic twins is advocated and by the use of Blum-Shub-Smale machines a first implementation that enables the detection of DoS attacks is shown. This result holds for both cases of with and without constraints on the input and jamming sequences of the adversary.



## **23. Security-Preserving Federated Learning via Byzantine-Sensitive Triplet Distance**

cs.LG

5 pages

**SubmitDate**: 2022-10-29    [abs](http://arxiv.org/abs/2210.16519v1) [paper-pdf](http://arxiv.org/pdf/2210.16519v1)

**Authors**: Youngjoon Lee, Sangwoo Park, Joonhyuk Kang

**Abstract**: While being an effective framework of learning a shared model across multiple edge devices, federated learning (FL) is generally vulnerable to Byzantine attacks from adversarial edge devices. While existing works on FL mitigate such compromised devices by only aggregating a subset of the local models at the server side, they still cannot successfully ignore the outliers due to imprecise scoring rule. In this paper, we propose an effective Byzantine-robust FL framework, namely dummy contrastive aggregation, by defining a novel scoring function that sensitively discriminates whether the model has been poisoned or not. Key idea is to extract essential information from every local models along with the previous global model to define a distance measure in a manner similar to triplet loss. Numerical results validate the advantage of the proposed approach by showing improved performance as compared to the state-of-the-art Byzantine-resilient aggregation methods, e.g., Krum, Trimmed-mean, and Fang.



## **24. Robust Boosting Forests with Richer Deep Feature Hierarchy**

cs.CV

**SubmitDate**: 2022-10-29    [abs](http://arxiv.org/abs/2210.16451v1) [paper-pdf](http://arxiv.org/pdf/2210.16451v1)

**Authors**: Jianqiao Wangni

**Abstract**: We propose a robust variant of boosting forest to the various adversarial defense methods, and apply it to enhance the robustness of the deep neural network. We retain the deep network architecture, weights, and middle layer features, then install gradient boosting forest to select the features from each layer of the deep network, and predict the target. For training each decision tree, we propose a novel conservative and greedy trade-off, with consideration for less misprediction instead of pure gain functions, therefore being suboptimal and conservative. We actively increase tree depth to remedy the accuracy with splits in more features, being more greedy in growing tree depth. We propose a new task on 3D face model, whose robustness has not been carefully studied, despite the great security and privacy concerns related to face analytics. We tried a simple attack method on a pure convolutional neural network (CNN) face shape estimator, making it degenerate to only output average face shape with invisible perturbation. Our conservative-greedy boosting forest (CGBF) on face landmark datasets showed a great improvement over original pure deep learning methods under the adversarial attacks.



## **25. MAZE: Data-Free Model Stealing Attack Using Zeroth-Order Gradient Estimation**

stat.ML

**SubmitDate**: 2022-10-28    [abs](http://arxiv.org/abs/2005.03161v2) [paper-pdf](http://arxiv.org/pdf/2005.03161v2)

**Authors**: Sanjay Kariyappa, Atul Prakash, Moinuddin Qureshi

**Abstract**: Model Stealing (MS) attacks allow an adversary with black-box access to a Machine Learning model to replicate its functionality, compromising the confidentiality of the model. Such attacks train a clone model by using the predictions of the target model for different inputs. The effectiveness of such attacks relies heavily on the availability of data necessary to query the target model. Existing attacks either assume partial access to the dataset of the target model or availability of an alternate dataset with semantic similarities. This paper proposes MAZE -- a data-free model stealing attack using zeroth-order gradient estimation. In contrast to prior works, MAZE does not require any data and instead creates synthetic data using a generative model. Inspired by recent works in data-free Knowledge Distillation (KD), we train the generative model using a disagreement objective to produce inputs that maximize disagreement between the clone and the target model. However, unlike the white-box setting of KD, where the gradient information is available, training a generator for model stealing requires performing black-box optimization, as it involves accessing the target model under attack. MAZE relies on zeroth-order gradient estimation to perform this optimization and enables a highly accurate MS attack. Our evaluation with four datasets shows that MAZE provides a normalized clone accuracy in the range of 0.91x to 0.99x, and outperforms even the recent attacks that rely on partial data (JBDA, clone accuracy 0.13x to 0.69x) and surrogate data (KnockoffNets, clone accuracy 0.52x to 0.97x). We also study an extension of MAZE in the partial-data setting and develop MAZE-PD, which generates synthetic data closer to the target distribution. MAZE-PD further improves the clone accuracy (0.97x to 1.0x) and reduces the query required for the attack by 2x-24x.



## **26. Distributed Black-box Attack against Image Classification Cloud Services**

cs.LG

10 pages, 11 figures

**SubmitDate**: 2022-10-28    [abs](http://arxiv.org/abs/2210.16371v1) [paper-pdf](http://arxiv.org/pdf/2210.16371v1)

**Authors**: Han Wu, Sareh Rowlands, Johan Wahlstrom

**Abstract**: Black-box adversarial attacks can fool image classifiers into misclassifying images without requiring access to model structure and weights. Recently proposed black-box attacks can achieve a success rate of more than 95\% after less than 1,000 queries. The question then arises of whether black-box attacks have become a real threat against IoT devices that rely on cloud APIs to achieve image classification. To shed some light on this, note that prior research has primarily focused on increasing the success rate and reducing the number of required queries. However, another crucial factor for black-box attacks against cloud APIs is the time required to perform the attack. This paper applies black-box attacks directly to cloud APIs rather than to local models, thereby avoiding multiple mistakes made in prior research. Further, we exploit load balancing to enable distributed black-box attacks that can reduce the attack time by a factor of about five for both local search and gradient estimation methods.



## **27. Universalization of any adversarial attack using very few test examples**

cs.LG

Appeared in ACM CODS-COMAD 2022 (Research Track)

**SubmitDate**: 2022-10-28    [abs](http://arxiv.org/abs/2005.08632v2) [paper-pdf](http://arxiv.org/pdf/2005.08632v2)

**Authors**: Sandesh Kamath, Amit Deshpande, K V Subrahmanyam, Vineeth N Balasubramanian

**Abstract**: Deep learning models are known to be vulnerable not only to input-dependent adversarial attacks but also to input-agnostic or universal adversarial attacks. Dezfooli et al. \cite{Dezfooli17,Dezfooli17anal} construct universal adversarial attack on a given model by looking at a large number of training data points and the geometry of the decision boundary near them. Subsequent work \cite{Khrulkov18} constructs universal attack by looking only at test examples and intermediate layers of the given model. In this paper, we propose a simple universalization technique to take any input-dependent adversarial attack and construct a universal attack by only looking at very few adversarial test examples. We do not require details of the given model and have negligible computational overhead for universalization. We theoretically justify our universalization technique by a spectral property common to many input-dependent adversarial perturbations, e.g., gradients, Fast Gradient Sign Method (FGSM) and DeepFool. Using matrix concentration inequalities and spectral perturbation bounds, we show that the top singular vector of input-dependent adversarial directions on a small test sample gives an effective and simple universal adversarial attack. For VGG16 and VGG19 models trained on ImageNet, our simple universalization of Gradient, FGSM, and DeepFool perturbations using a test sample of 64 images gives fooling rates comparable to state-of-the-art universal attacks \cite{Dezfooli17,Khrulkov18} for reasonable norms of perturbation. Code available at https://github.com/ksandeshk/svd-uap .



## **28. Local Model Reconstruction Attacks in Federated Learning and their Uses**

cs.LG

**SubmitDate**: 2022-10-28    [abs](http://arxiv.org/abs/2210.16205v1) [paper-pdf](http://arxiv.org/pdf/2210.16205v1)

**Authors**: Ilias Driouich, Chuan Xu, Giovanni Neglia, Frederic Giroire, Eoin Thomas

**Abstract**: In this paper, we initiate the study of local model reconstruction attacks for federated learning, where a honest-but-curious adversary eavesdrops the messages exchanged between a targeted client and the server, and then reconstructs the local/personalized model of the victim. The local model reconstruction attack allows the adversary to trigger other classical attacks in a more effective way, since the local model only depends on the client's data and can leak more private information than the global model learned by the server. Additionally, we propose a novel model-based attribute inference attack in federated learning leveraging the local model reconstruction attack. We provide an analytical lower-bound for this attribute inference attack. Empirical results using real world datasets confirm that our local reconstruction attack works well for both regression and classification tasks. Moreover, we benchmark our novel attribute inference attack against the state-of-the-art attacks in federated learning. Our attack results in higher reconstruction accuracy especially when the clients' datasets are heterogeneous. Our work provides a new angle for designing powerful and explainable attacks to effectively quantify the privacy risk in FL.



## **29. Improving Transferability of Adversarial Examples on Face Recognition with Beneficial Perturbation Feature Augmentation**

cs.CV

**SubmitDate**: 2022-10-28    [abs](http://arxiv.org/abs/2210.16117v1) [paper-pdf](http://arxiv.org/pdf/2210.16117v1)

**Authors**: Fengfan Zhou, Hefei Ling, Yuxuan Shi, Jiazhong Chen, Zongyi Li, Qian Wang

**Abstract**: Face recognition (FR) models can be easily fooled by adversarial examples, which are crafted by adding imperceptible perturbations on benign face images. To improve the transferability of adversarial examples on FR models, we propose a novel attack method called Beneficial Perturbation Feature Augmentation Attack (BPFA), which reduces the overfitting of the adversarial examples to surrogate FR models by the adversarial strategy. Specifically, in the backpropagation step, BPFA records the gradients on pre-selected features and uses the gradient on the input image to craft adversarial perturbation to be added on the input image. In the next forward propagation step, BPFA leverages the recorded gradients to add perturbations(i.e., beneficial perturbations) that can be pitted against the adversarial perturbation added on the input image on their corresponding features. The above two steps are repeated until the last backpropagation step before the maximum number of iterations is reached. The optimization process of the adversarial perturbation added on the input image and the optimization process of the beneficial perturbations added on the features correspond to a minimax two-player game. Extensive experiments demonstrate that BPFA outperforms the state-of-the-art gradient-based adversarial attacks on FR.



## **30. Watermarking Graph Neural Networks based on Backdoor Attacks**

cs.LG

18 pages, 9 figures

**SubmitDate**: 2022-10-28    [abs](http://arxiv.org/abs/2110.11024v4) [paper-pdf](http://arxiv.org/pdf/2110.11024v4)

**Authors**: Jing Xu, Stefanos Koffas, Oguzhan Ersoy, Stjepan Picek

**Abstract**: Graph Neural Networks (GNNs) have achieved promising performance in various real-world applications. Building a powerful GNN model is not a trivial task, as it requires a large amount of training data, powerful computing resources, and human expertise in fine-tuning the model. Moreover, with the development of adversarial attacks, e.g., model stealing attacks, GNNs raise challenges to model authentication. To avoid copyright infringement on GNNs, verifying the ownership of the GNN models is necessary.   This paper presents a watermarking framework for GNNs for both graph and node classification tasks. We 1) design two strategies to generate watermarked data for the graph classification task and one for the node classification task, 2) embed the watermark into the host model through training to obtain the watermarked GNN model, and 3) verify the ownership of the suspicious model in a black-box setting. The experiments show that our framework can verify the ownership of GNN models with a very high probability (up to $99\%$) for both tasks. Finally, we experimentally show that our watermarking approach is robust against a state-of-the-art model extraction technique and four state-of-the-art defenses against backdoor attacks.



## **31. RoChBert: Towards Robust BERT Fine-tuning for Chinese**

cs.CL

Accepted by Findings of EMNLP 2022

**SubmitDate**: 2022-10-28    [abs](http://arxiv.org/abs/2210.15944v1) [paper-pdf](http://arxiv.org/pdf/2210.15944v1)

**Authors**: Zihan Zhang, Jinfeng Li, Ning Shi, Bo Yuan, Xiangyu Liu, Rong Zhang, Hui Xue, Donghong Sun, Chao Zhang

**Abstract**: Despite of the superb performance on a wide range of tasks, pre-trained language models (e.g., BERT) have been proved vulnerable to adversarial texts. In this paper, we present RoChBERT, a framework to build more Robust BERT-based models by utilizing a more comprehensive adversarial graph to fuse Chinese phonetic and glyph features into pre-trained representations during fine-tuning. Inspired by curriculum learning, we further propose to augment the training dataset with adversarial texts in combination with intermediate samples. Extensive experiments demonstrate that RoChBERT outperforms previous methods in significant ways: (i) robust -- RoChBERT greatly improves the model robustness without sacrificing accuracy on benign texts. Specifically, the defense lowers the success rates of unlimited and limited attacks by 59.43% and 39.33% respectively, while remaining accuracy of 93.30%; (ii) flexible -- RoChBERT can easily extend to various language models to solve different downstream tasks with excellent performance; and (iii) efficient -- RoChBERT can be directly applied to the fine-tuning stage without pre-training language model from scratch, and the proposed data augmentation method is also low-cost.



## **32. DICTION: DynamIC robusT whIte bOx watermarkiNg scheme**

cs.CR

18 pages, 5 figures, PrePrint

**SubmitDate**: 2022-10-27    [abs](http://arxiv.org/abs/2210.15745v1) [paper-pdf](http://arxiv.org/pdf/2210.15745v1)

**Authors**: Reda Bellafqira, Gouenou Coatrieux

**Abstract**: Deep neural network (DNN) watermarking is a suitable method for protecting the ownership of deep learning (DL) models derived from computationally intensive processes and painstakingly compiled and annotated datasets. It secretly embeds an identifier (watermark) within the model, which can be retrieved by the owner to prove ownership. In this paper, we first provide a unified framework for white box DNN watermarking schemes. It includes current state-of-the art methods outlining their theoretical inter-connections. In second, we introduce DICTION, a new white-box Dynamic Robust watermarking scheme, we derived from this framework. Its main originality stands on a generative adversarial network (GAN) strategy where the watermark extraction function is a DNN trained as a GAN discriminator, and the target model to watermark as a GAN generator taking a GAN latent space as trigger set input. DICTION can be seen as a generalization of DeepSigns which, to the best of knowledge, is the only other Dynamic white-box watermarking scheme from the literature. Experiments conducted on the same model test set as Deepsigns demonstrate that our scheme achieves much better performance. Especially, and contrarily to DeepSigns, with DICTION one can increase the watermark capacity while preserving at best the model accuracy and ensuring simultaneously a strong robustness against a wide range of watermark removal and detection attacks.



## **33. TAD: Transfer Learning-based Multi-Adversarial Detection of Evasion Attacks against Network Intrusion Detection Systems**

cs.CR

This is a preprint of an already published journal paper

**SubmitDate**: 2022-10-27    [abs](http://arxiv.org/abs/2210.15700v1) [paper-pdf](http://arxiv.org/pdf/2210.15700v1)

**Authors**: Islam Debicha, Richard Bauwens, Thibault Debatty, Jean-Michel Dricot, Tayeb Kenaza, Wim Mees

**Abstract**: Nowadays, intrusion detection systems based on deep learning deliver state-of-the-art performance. However, recent research has shown that specially crafted perturbations, called adversarial examples, are capable of significantly reducing the performance of these intrusion detection systems. The objective of this paper is to design an efficient transfer learning-based adversarial detector and then to assess the effectiveness of using multiple strategically placed adversarial detectors compared to a single adversarial detector for intrusion detection systems. In our experiments, we implement existing state-of-the-art models for intrusion detection. We then attack those models with a set of chosen evasion attacks. In an attempt to detect those adversarial attacks, we design and implement multiple transfer learning-based adversarial detectors, each receiving a subset of the information passed through the IDS. By combining their respective decisions, we illustrate that combining multiple detectors can further improve the detectability of adversarial traffic compared to a single detector in the case of a parallel IDS design.



## **34. Learning Location from Shared Elevation Profiles in Fitness Apps: A Privacy Perspective**

cs.CR

16 pages, 12 figures, 10 tables; accepted for publication in IEEE  Transactions on Mobile Computing (October 2022). arXiv admin note:  substantial text overlap with arXiv:1910.09041

**SubmitDate**: 2022-10-27    [abs](http://arxiv.org/abs/2210.15529v1) [paper-pdf](http://arxiv.org/pdf/2210.15529v1)

**Authors**: Ulku Meteriz-Yildiran, Necip Fazil Yildiran, Joongheon Kim, David Mohaisen

**Abstract**: The extensive use of smartphones and wearable devices has facilitated many useful applications. For example, with Global Positioning System (GPS)-equipped smart and wearable devices, many applications can gather, process, and share rich metadata, such as geolocation, trajectories, elevation, and time. For example, fitness applications, such as Runkeeper and Strava, utilize the information for activity tracking and have recently witnessed a boom in popularity. Those fitness tracker applications have their own web platforms and allow users to share activities on such platforms or even with other social network platforms. To preserve the privacy of users while allowing sharing, several of those platforms may allow users to disclose partial information, such as the elevation profile for an activity, which supposedly would not leak the location of the users. In this work, and as a cautionary tale, we create a proof of concept where we examine the extent to which elevation profiles can be used to predict the location of users. To tackle this problem, we devise three plausible threat settings under which the city or borough of the targets can be predicted. Those threat settings define the amount of information available to the adversary to launch the prediction attacks. Establishing that simple features of elevation profiles, e.g., spectral features, are insufficient, we devise both natural language processing (NLP)-inspired text-like representation and computer vision-inspired image-like representation of elevation profiles, and we convert the problem at hand into text and image classification problem. We use both traditional machine learning- and deep learning-based techniques and achieve a prediction success rate ranging from 59.59\% to 99.80\%. The findings are alarming, highlighting that sharing elevation information may have significant location privacy risks.



## **35. An Analysis of Robustness of Non-Lipschitz Networks**

cs.LG

42 pages, 9 figures

**SubmitDate**: 2022-10-27    [abs](http://arxiv.org/abs/2010.06154v3) [paper-pdf](http://arxiv.org/pdf/2010.06154v3)

**Authors**: Maria-Florina Balcan, Avrim Blum, Dravyansh Sharma, Hongyang Zhang

**Abstract**: Despite significant advances, deep networks remain highly susceptible to adversarial attack. One fundamental challenge is that small input perturbations can often produce large movements in the network's final-layer feature space. In this paper, we define an attack model that abstracts this challenge, to help understand its intrinsic properties. In our model, the adversary may move data an arbitrary distance in feature space but only in random low-dimensional subspaces. We prove such adversaries can be quite powerful: defeating any algorithm that must classify any input it is given. However, by allowing the algorithm to abstain on unusual inputs, we show such adversaries can be overcome when classes are reasonably well-separated in feature space. We further provide strong theoretical guarantees for setting algorithm parameters to optimize over accuracy-abstention trade-offs using data-driven methods. Our results provide new robustness guarantees for nearest-neighbor style algorithms, and also have application to contrastive learning, where we empirically demonstrate the ability of such algorithms to obtain high robust accuracy with low abstention rates. Our model is also motivated by strategic classification, where entities being classified aim to manipulate their observable features to produce a preferred classification, and we provide new insights into that area as well.



## **36. LeNo: Adversarial Robust Salient Object Detection Networks with Learnable Noise**

cs.CV

8 pages, 5 figures, submitted to AAAI

**SubmitDate**: 2022-10-27    [abs](http://arxiv.org/abs/2210.15392v1) [paper-pdf](http://arxiv.org/pdf/2210.15392v1)

**Authors**: He Tang, He Wang

**Abstract**: Pixel-wise predction with deep neural network has become an effective paradigm for salient object detection (SOD) and achieved remakable performance. However, very few SOD models are robust against adversarial attacks which are visually imperceptible for human visual attention. The previous work robust salient object detection against adversarial attacks (ROSA) shuffles the pre-segmented superpixels and then refines the coarse saliency map by the densely connected CRF. Different from ROSA that rely on various pre- and post-processings, this paper proposes a light-weight Learnble Noise (LeNo) to against adversarial attacks for SOD models. LeNo preserves accuracy of SOD models on both adversarial and clean images, as well as inference speed. In general, LeNo consists of a simple shallow noise and noise estimation that embedded in the encoder and decoder of arbitrary SOD networks respectively. Inspired by the center prior of human visual attention mechanism, we initialize the shallow noise with a cross-shaped gaussian distribution for better defense against adversarial attacks. Instead of adding additional network components for post-processing, the proposed noise estimation modifies only one channel of the decoder. With the deeply-supervised noise-decoupled training on state-of-the-art RGB and RGB-D SOD networks, LeNo outperforms previous works not only on adversarial images but also clean images, which contributes stronger robustness for SOD.



## **37. Isometric 3D Adversarial Examples in the Physical World**

cs.CV

NeurIPS 2022

**SubmitDate**: 2022-10-27    [abs](http://arxiv.org/abs/2210.15291v1) [paper-pdf](http://arxiv.org/pdf/2210.15291v1)

**Authors**: Yibo Miao, Yinpeng Dong, Jun Zhu, Xiao-Shan Gao

**Abstract**: 3D deep learning models are shown to be as vulnerable to adversarial examples as 2D models. However, existing attack methods are still far from stealthy and suffer from severe performance degradation in the physical world. Although 3D data is highly structured, it is difficult to bound the perturbations with simple metrics in the Euclidean space. In this paper, we propose a novel $\epsilon$-isometric ($\epsilon$-ISO) attack to generate natural and robust 3D adversarial examples in the physical world by considering the geometric properties of 3D objects and the invariance to physical transformations. For naturalness, we constrain the adversarial example to be $\epsilon$-isometric to the original one by adopting the Gaussian curvature as a surrogate metric guaranteed by a theoretical analysis. For invariance to physical transformations, we propose a maxima over transformation (MaxOT) method that actively searches for the most harmful transformations rather than random ones to make the generated adversarial example more robust in the physical world. Experiments on typical point cloud recognition models validate that our approach can significantly improve the attack success rate and naturalness of the generated 3D adversarial examples than the state-of-the-art attack methods.



## **38. TASA: Deceiving Question Answering Models by Twin Answer Sentences Attack**

cs.CL

Accepted by EMNLP 2022 (long), 9 pages main + 2 pages references + 7  pages appendix

**SubmitDate**: 2022-10-27    [abs](http://arxiv.org/abs/2210.15221v1) [paper-pdf](http://arxiv.org/pdf/2210.15221v1)

**Authors**: Yu Cao, Dianqi Li, Meng Fang, Tianyi Zhou, Jun Gao, Yibing Zhan, Dacheng Tao

**Abstract**: We present Twin Answer Sentences Attack (TASA), an adversarial attack method for question answering (QA) models that produces fluent and grammatical adversarial contexts while maintaining gold answers. Despite phenomenal progress on general adversarial attacks, few works have investigated the vulnerability and attack specifically for QA models. In this work, we first explore the biases in the existing models and discover that they mainly rely on keyword matching between the question and context, and ignore the relevant contextual relations for answer prediction. Based on two biases above, TASA attacks the target model in two folds: (1) lowering the model's confidence on the gold answer with a perturbed answer sentence; (2) misguiding the model towards a wrong answer with a distracting answer sentence. Equipped with designed beam search and filtering methods, TASA can generate more effective attacks than existing textual attack methods while sustaining the quality of contexts, in extensive experiments on five QA datasets and human evaluations.



## **39. V-Cloak: Intelligibility-, Naturalness- & Timbre-Preserving Real-Time Voice Anonymization**

cs.SD

Accepted by USENIX Security Symposium 2023

**SubmitDate**: 2022-10-27    [abs](http://arxiv.org/abs/2210.15140v1) [paper-pdf](http://arxiv.org/pdf/2210.15140v1)

**Authors**: Jiangyi Deng, Fei Teng, Yanjiao Chen, Xiaofu Chen, Zhaohui Wang, Wenyuan Xu

**Abstract**: Voice data generated on instant messaging or social media applications contains unique user voiceprints that may be abused by malicious adversaries for identity inference or identity theft. Existing voice anonymization techniques, e.g., signal processing and voice conversion/synthesis, suffer from degradation of perceptual quality. In this paper, we develop a voice anonymization system, named V-Cloak, which attains real-time voice anonymization while preserving the intelligibility, naturalness and timbre of the audio. Our designed anonymizer features a one-shot generative model that modulates the features of the original audio at different frequency levels. We train the anonymizer with a carefully-designed loss function. Apart from the anonymity loss, we further incorporate the intelligibility loss and the psychoacoustics-based naturalness loss. The anonymizer can realize untargeted and targeted anonymization to achieve the anonymity goals of unidentifiability and unlinkability.   We have conducted extensive experiments on four datasets, i.e., LibriSpeech (English), AISHELL (Chinese), CommonVoice (French) and CommonVoice (Italian), five Automatic Speaker Verification (ASV) systems (including two DNN-based, two statistical and one commercial ASV), and eleven Automatic Speech Recognition (ASR) systems (for different languages). Experiment results confirm that V-Cloak outperforms five baselines in terms of anonymity performance. We also demonstrate that V-Cloak trained only on the VoxCeleb1 dataset against ECAPA-TDNN ASV and DeepSpeech2 ASR has transferable anonymity against other ASVs and cross-language intelligibility for other ASRs. Furthermore, we verify the robustness of V-Cloak against various de-noising techniques and adaptive attacks. Hopefully, V-Cloak may provide a cloak for us in a prism world.



## **40. Adaptive Test-Time Defense with the Manifold Hypothesis**

cs.LG

**SubmitDate**: 2022-10-27    [abs](http://arxiv.org/abs/2210.14404v2) [paper-pdf](http://arxiv.org/pdf/2210.14404v2)

**Authors**: Zhaoyuan Yang, Zhiwei Xu, Jing Zhang, Richard Hartley, Peter Tu

**Abstract**: In this work, we formulate a novel framework of adversarial robustness using the manifold hypothesis. Our framework provides sufficient conditions for defending against adversarial examples. We develop a test-time defense method with our formulation and variational inference. The developed approach combines manifold learning with the Bayesian framework to provide adversarial robustness without the need for adversarial training. We show that our proposed approach can provide adversarial robustness even if attackers are aware of existence of test-time defense. In additions, our approach can also serve as a test-time defense mechanism for variational autoencoders.



## **41. Improving Adversarial Robustness with Self-Paced Hard-Class Pair Reweighting**

cs.CV

**SubmitDate**: 2022-10-26    [abs](http://arxiv.org/abs/2210.15068v1) [paper-pdf](http://arxiv.org/pdf/2210.15068v1)

**Authors**: Pengyue Hou, Jie Han, Xingyu Li

**Abstract**: Deep Neural Networks are vulnerable to adversarial attacks. Among many defense strategies, adversarial training with untargeted attacks is one of the most recognized methods. Theoretically, the predicted labels of untargeted attacks should be unpredictable and uniformly-distributed overall false classes. However, we find that the naturally imbalanced inter-class semantic similarity makes those hard-class pairs to become the virtual targets of each other. This study investigates the impact of such closely-coupled classes on adversarial attacks and develops a self-paced reweighting strategy in adversarial training accordingly. Specifically, we propose to upweight hard-class pair loss in model optimization, which prompts learning discriminative features from hard classes. We further incorporate a term to quantify hard-class pair consistency in adversarial training, which greatly boost model robustness. Extensive experiments show that the proposed adversarial training method achieves superior robustness performance over state-of-the-art defenses against a wide range of adversarial attacks.



## **42. Using Deception in Markov Game to Understand Adversarial Behaviors through a Capture-The-Flag Environment**

cs.GT

Accepted at GameSec 2022

**SubmitDate**: 2022-10-26    [abs](http://arxiv.org/abs/2210.15011v1) [paper-pdf](http://arxiv.org/pdf/2210.15011v1)

**Authors**: Siddhant Bhambri, Purv Chauhan, Frederico Araujo, Adam Doupé, Subbarao Kambhampati

**Abstract**: Identifying the actual adversarial threat against a system vulnerability has been a long-standing challenge for cybersecurity research. To determine an optimal strategy for the defender, game-theoretic based decision models have been widely used to simulate the real-world attacker-defender scenarios while taking the defender's constraints into consideration. In this work, we focus on understanding human attacker behaviors in order to optimize the defender's strategy. To achieve this goal, we model attacker-defender engagements as Markov Games and search for their Bayesian Stackelberg Equilibrium. We validate our modeling approach and report our empirical findings using a Capture-The-Flag (CTF) setup, and we conduct user studies on adversaries with varying skill-levels. Our studies show that application-level deceptions are an optimal mitigation strategy against targeted attacks -- outperforming classic cyber-defensive maneuvers, such as patching or blocking network requests. We use this result to further hypothesize over the attacker's behaviors when trapped in an embedded honeypot environment and present a detailed analysis of the same.



## **43. Model-Free Prediction of Adversarial Drop Points in 3D Point Clouds**

cs.CV

10 pages, 6 figures

**SubmitDate**: 2022-10-26    [abs](http://arxiv.org/abs/2210.14164v2) [paper-pdf](http://arxiv.org/pdf/2210.14164v2)

**Authors**: Hanieh Naderi, Chinthaka Dinesh, Ivan V. Bajic, Shohreh Kasaei

**Abstract**: Adversarial attacks pose serious challenges for deep neural network (DNN)-based analysis of various input signals. In the case of 3D point clouds, methods have been developed to identify points that play a key role in the network decision, and these become crucial in generating existing adversarial attacks. For example, a saliency map approach is a popular method for identifying adversarial drop points, whose removal would significantly impact the network decision. Generally, methods for identifying adversarial points rely on the deep model itself in order to determine which points are critically important for the model's decision. This paper aims to provide a novel viewpoint on this problem, in which adversarial points can be predicted independently of the model. To this end, we define 14 point cloud features and use multiple linear regression to examine whether these features can be used for model-free adversarial point prediction, and which combination of features is best suited for this purpose. Experiments show that a suitable combination of features is able to predict adversarial points of three different networks -- PointNet, PointNet++, and DGCNN -- significantly better than a random guess. The results also provide further insight into DNNs for point cloud analysis, by showing which features play key roles in their decision-making process.



## **44. Disentangled Text Representation Learning with Information-Theoretic Perspective for Adversarial Robustness**

cs.CL

**SubmitDate**: 2022-10-26    [abs](http://arxiv.org/abs/2210.14957v1) [paper-pdf](http://arxiv.org/pdf/2210.14957v1)

**Authors**: Jiahao Zhao, Wenji Mao

**Abstract**: Adversarial vulnerability remains a major obstacle to constructing reliable NLP systems. When imperceptible perturbations are added to raw input text, the performance of a deep learning model may drop dramatically under attacks. Recent work argues the adversarial vulnerability of the model is caused by the non-robust features in supervised training. Thus in this paper, we tackle the adversarial robustness challenge from the view of disentangled representation learning, which is able to explicitly disentangle robust and non-robust features in text. Specifically, inspired by the variation of information (VI) in information theory, we derive a disentangled learning objective composed of mutual information to represent both the semantic representativeness of latent embeddings and differentiation of robust and non-robust features. On the basis of this, we design a disentangled learning network to estimate these mutual information. Experiments on text classification and entailment tasks show that our method significantly outperforms the representative methods under adversarial attacks, indicating that discarding non-robust features is critical for improving adversarial robustness.



## **45. On the Versatile Uses of Partial Distance Correlation in Deep Learning**

cs.CV

This paper has been selected as best paper award for ECCV 2022!

**SubmitDate**: 2022-10-26    [abs](http://arxiv.org/abs/2207.09684v2) [paper-pdf](http://arxiv.org/pdf/2207.09684v2)

**Authors**: Xingjian Zhen, Zihang Meng, Rudrasis Chakraborty, Vikas Singh

**Abstract**: Comparing the functional behavior of neural network models, whether it is a single network over time or two (or more networks) during or post-training, is an essential step in understanding what they are learning (and what they are not), and for identifying strategies for regularization or efficiency improvements. Despite recent progress, e.g., comparing vision transformers to CNNs, systematic comparison of function, especially across different networks, remains difficult and is often carried out layer by layer. Approaches such as canonical correlation analysis (CCA) are applicable in principle, but have been sparingly used so far. In this paper, we revisit a (less widely known) from statistics, called distance correlation (and its partial variant), designed to evaluate correlation between feature spaces of different dimensions. We describe the steps necessary to carry out its deployment for large scale models -- this opens the door to a surprising array of applications ranging from conditioning one deep model w.r.t. another, learning disentangled representations as well as optimizing diverse models that would directly be more robust to adversarial attacks. Our experiments suggest a versatile regularizer (or constraint) with many advantages, which avoids some of the common difficulties one faces in such analyses. Code is at https://github.com/zhenxingjian/Partial_Distance_Correlation.



## **46. Identifying Threats, Cybercrime and Digital Forensic Opportunities in Smart City Infrastructure via Threat Modeling**

cs.CR

**SubmitDate**: 2022-10-26    [abs](http://arxiv.org/abs/2210.14692v1) [paper-pdf](http://arxiv.org/pdf/2210.14692v1)

**Authors**: Yee Ching Tok, Sudipta Chattopadhyay

**Abstract**: Technological advances have enabled multiple countries to consider implementing Smart City Infrastructure to provide in-depth insights into different data points and enhance the lives of citizens. Unfortunately, these new technological implementations also entice adversaries and cybercriminals to execute cyber-attacks and commit criminal acts on these modern infrastructures. Given the borderless nature of cyber attacks, varying levels of understanding of smart city infrastructure and ongoing investigation workloads, law enforcement agencies and investigators would be hard-pressed to respond to these kinds of cybercrime. Without an investigative capability by investigators, these smart infrastructures could become new targets favored by cybercriminals.   To address the challenges faced by investigators, we propose a common definition of smart city infrastructure. Based on the definition, we utilize the STRIDE threat modeling methodology and the Microsoft Threat Modeling Tool to identify threats present in the infrastructure and create a threat model which can be further customized or extended by interested parties. Next, we map offences, possible evidence sources and types of threats identified to help investigators understand what crimes could have been committed and what evidence would be required in their investigation work. Finally, noting that Smart City Infrastructure investigations would be a global multi-faceted challenge, we discuss technical and legal opportunities in digital forensics on Smart City Infrastructure.



## **47. Certified Robustness in Federated Learning**

cs.LG

Accepted at Workshop on Federated Learning: Recent Advances and New  Challenges, NeurIPS 2022

**SubmitDate**: 2022-10-26    [abs](http://arxiv.org/abs/2206.02535v2) [paper-pdf](http://arxiv.org/pdf/2206.02535v2)

**Authors**: Motasem Alfarra, Juan C. Pérez, Egor Shulgin, Peter Richtárik, Bernard Ghanem

**Abstract**: Federated learning has recently gained significant attention and popularity due to its effectiveness in training machine learning models on distributed data privately. However, as in the single-node supervised learning setup, models trained in federated learning suffer from vulnerability to imperceptible input transformations known as adversarial attacks, questioning their deployment in security-related applications. In this work, we study the interplay between federated training, personalization, and certified robustness. In particular, we deploy randomized smoothing, a widely-used and scalable certification method, to certify deep networks trained on a federated setup against input perturbations and transformations. We find that the simple federated averaging technique is effective in building not only more accurate, but also more certifiably-robust models, compared to training solely on local data. We further analyze personalization, a popular technique in federated training that increases the model's bias towards local data, on robustness. We show several advantages of personalization over both~(that is, only training on local data and federated training) in building more robust models with faster training. Finally, we explore the robustness of mixtures of global and local~(i.e. personalized) models, and find that the robustness of local models degrades as they diverge from the global model



## **48. Short Paper: Static and Microarchitectural ML-Based Approaches For Detecting Spectre Vulnerabilities and Attacks**

cs.CR

5 pages, 2 figures. Accepted to the Hardware and Architectural  Support for Security and Privacy (HASP'22), in conjunction with the 55th  IEEE/ACM International Symposium on Microarchitecture (MICRO'22)

**SubmitDate**: 2022-10-26    [abs](http://arxiv.org/abs/2210.14452v1) [paper-pdf](http://arxiv.org/pdf/2210.14452v1)

**Authors**: Chidera Biringa, Gaspard Baye, Gökhan Kul

**Abstract**: Spectre intrusions exploit speculative execution design vulnerabilities in modern processors. The attacks violate the principles of isolation in programs to gain unauthorized private user information. Current state-of-the-art detection techniques utilize micro-architectural features or vulnerable speculative code to detect these threats. However, these techniques are insufficient as Spectre attacks have proven to be more stealthy with recently discovered variants that bypass current mitigation mechanisms. Side-channels generate distinct patterns in processor cache, and sensitive information leakage is dependent on source code vulnerable to Spectre attacks, where an adversary uses these vulnerabilities, such as branch prediction, which causes a data breach. Previous studies predominantly approach the detection of Spectre attacks using the microarchitectural analysis, a reactive approach. Hence, in this paper, we present the first comprehensive evaluation of static and microarchitectural analysis-assisted machine learning approaches to detect Spectre vulnerable code snippets (preventive) and Spectre attacks (reactive). We evaluate the performance trade-offs in employing classifiers for detecting Spectre vulnerabilities and attacks.



## **49. LP-BFGS attack: An adversarial attack based on the Hessian with limited pixels**

cs.CR

5 pages, 4 figures

**SubmitDate**: 2022-10-26    [abs](http://arxiv.org/abs/2210.15446v1) [paper-pdf](http://arxiv.org/pdf/2210.15446v1)

**Authors**: Jiebao Zhang, Wenhua Qian, Rencan Nie, Jinde Cao, Dan Xu

**Abstract**: Deep neural networks are vulnerable to adversarial attacks. Most white-box attacks are based on the gradient of models to the input. Since the computation and memory budget, adversarial attacks based on the Hessian information are not paid enough attention. In this work, we study the attack performance and computation cost of the attack method based on the Hessian with a limited perturbation pixel number. Specifically, we propose the Limited Pixel BFGS (LP-BFGS) attack method by incorporating the BFGS algorithm. Some pixels are selected as perturbation pixels by the Integrated Gradient algorithm, which are regarded as optimization variables of the LP-BFGS attack. Experimental results across different networks and datasets with various perturbation pixel numbers demonstrate our approach has a comparable attack with an acceptable computation compared with existing solutions.



## **50. Improving Adversarial Robustness via Joint Classification and Multiple Explicit Detection Classes**

cs.CV

21 pages, 6 figures

**SubmitDate**: 2022-10-26    [abs](http://arxiv.org/abs/2210.14410v1) [paper-pdf](http://arxiv.org/pdf/2210.14410v1)

**Authors**: Sina Baharlouei, Fatemeh Sheikholeslami, Meisam Razaviyayn, Zico Kolter

**Abstract**: This work concerns the development of deep networks that are certifiably robust to adversarial attacks. Joint robust classification-detection was recently introduced as a certified defense mechanism, where adversarial examples are either correctly classified or assigned to the "abstain" class. In this work, we show that such a provable framework can benefit by extension to networks with multiple explicit abstain classes, where the adversarial examples are adaptively assigned to those. We show that naively adding multiple abstain classes can lead to "model degeneracy", then we propose a regularization approach and a training method to counter this degeneracy by promoting full use of the multiple abstain classes. Our experiments demonstrate that the proposed approach consistently achieves favorable standard vs. robust verified accuracy tradeoffs, outperforming state-of-the-art algorithms for various choices of number of abstain classes.



