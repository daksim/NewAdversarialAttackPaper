# Latest Adversarial Attack Papers
**update at 2022-07-23 06:31:30**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. Careful What You Wish For: on the Extraction of Adversarially Trained Models**

cs.LG

To be published in the proceedings of the 19th Annual International  Conference on Privacy, Security & Trust (PST 2022). The conference  proceedings will be included in IEEE Xplore as in previous editions of the  conference

**SubmitDate**: 2022-07-21    [paper-pdf](http://arxiv.org/pdf/2207.10561v1)

**Authors**: Kacem Khaled, Gabriela Nicolescu, Felipe Gohring de Magalhães

**Abstracts**: Recent attacks on Machine Learning (ML) models such as evasion attacks with adversarial examples and models stealing through extraction attacks pose several security and privacy threats. Prior work proposes to use adversarial training to secure models from adversarial examples that can evade the classification of a model and deteriorate its performance. However, this protection technique affects the model's decision boundary and its prediction probabilities, hence it might raise model privacy risks. In fact, a malicious user using only a query access to the prediction output of a model can extract it and obtain a high-accuracy and high-fidelity surrogate model. To have a greater extraction, these attacks leverage the prediction probabilities of the victim model. Indeed, all previous work on extraction attacks do not take into consideration the changes in the training process for security purposes. In this paper, we propose a framework to assess extraction attacks on adversarially trained models with vision datasets. To the best of our knowledge, our work is the first to perform such evaluation. Through an extensive empirical study, we demonstrate that adversarially trained models are more vulnerable to extraction attacks than models obtained under natural training circumstances. They can achieve up to $\times1.2$ higher accuracy and agreement with a fraction lower than $\times0.75$ of the queries. We additionally find that the adversarial robustness capability is transferable through extraction attacks, i.e., extracted Deep Neural Networks (DNNs) from robust models show an enhanced accuracy to adversarial examples compared to extracted DNNs from naturally trained (i.e. standard) models.



## **2. Triangle Attack: A Query-efficient Decision-based Adversarial Attack**

cs.CV

Accepted by ECCV 2022, code is available at  https://github.com/xiaosen-wang/TA

**SubmitDate**: 2022-07-21    [paper-pdf](http://arxiv.org/pdf/2112.06569v3)

**Authors**: Xiaosen Wang, Zeliang Zhang, Kangheng Tong, Dihong Gong, Kun He, Zhifeng Li, Wei Liu

**Abstracts**: Decision-based attack poses a severe threat to real-world applications since it regards the target model as a black box and only accesses the hard prediction label. Great efforts have been made recently to decrease the number of queries; however, existing decision-based attacks still require thousands of queries in order to generate good quality adversarial examples. In this work, we find that a benign sample, the current and the next adversarial examples can naturally construct a triangle in a subspace for any iterative attacks. Based on the law of sines, we propose a novel Triangle Attack (TA) to optimize the perturbation by utilizing the geometric information that the longer side is always opposite the larger angle in any triangle. However, directly applying such information on the input image is ineffective because it cannot thoroughly explore the neighborhood of the input sample in the high dimensional space. To address this issue, TA optimizes the perturbation in the low frequency space for effective dimensionality reduction owing to the generality of such geometric property. Extensive evaluations on ImageNet dataset show that TA achieves a much higher attack success rate within 1,000 queries and needs a much less number of queries to achieve the same attack success rate under various perturbation budgets than existing decision-based attacks. With such high efficiency, we further validate the applicability of TA on real-world API, i.e., Tencent Cloud API.



## **3. Knowledge-enhanced Black-box Attacks for Recommendations**

cs.LG

Accepted in the KDD'22

**SubmitDate**: 2022-07-21    [paper-pdf](http://arxiv.org/pdf/2207.10307v1)

**Authors**: Jingfan Chen, Wenqi Fan, Guanghui Zhu, Xiangyu Zhao, Chunfeng Yuan, Qing Li, Yihua Huang

**Abstracts**: Recent studies have shown that deep neural networks-based recommender systems are vulnerable to adversarial attacks, where attackers can inject carefully crafted fake user profiles (i.e., a set of items that fake users have interacted with) into a target recommender system to achieve malicious purposes, such as promote or demote a set of target items. Due to the security and privacy concerns, it is more practical to perform adversarial attacks under the black-box setting, where the architecture/parameters and training data of target systems cannot be easily accessed by attackers. However, generating high-quality fake user profiles under black-box setting is rather challenging with limited resources to target systems. To address this challenge, in this work, we introduce a novel strategy by leveraging items' attribute information (i.e., items' knowledge graph), which can be publicly accessible and provide rich auxiliary knowledge to enhance the generation of fake user profiles. More specifically, we propose a knowledge graph-enhanced black-box attacking framework (KGAttack) to effectively learn attacking policies through deep reinforcement learning techniques, in which knowledge graph is seamlessly integrated into hierarchical policy networks to generate fake user profiles for performing adversarial black-box attacks. Comprehensive experiments on various real-world datasets demonstrate the effectiveness of the proposed attacking framework under the black-box setting.



## **4. Image Generation Network for Covert Transmission in Online Social Network**

cs.CV

ACMMM2022 Poster

**SubmitDate**: 2022-07-21    [paper-pdf](http://arxiv.org/pdf/2207.10292v1)

**Authors**: Zhengxin You, Qichao Ying, Sheng Li, Zhenxing Qian, Xinpeng Zhang

**Abstracts**: Online social networks have stimulated communications over the Internet more than ever, making it possible for secret message transmission over such noisy channels. In this paper, we propose a Coverless Image Steganography Network, called CIS-Net, that synthesizes a high-quality image directly conditioned on the secret message to transfer. CIS-Net is composed of four modules, namely, the Generation, Adversarial, Extraction, and Noise Module. The receiver can extract the hidden message without any loss even the images have been distorted by JPEG compression attacks. To disguise the behaviour of steganography, we collected images in the context of profile photos and stickers and train our network accordingly. As such, the generated images are more inclined to escape from malicious detection and attack. The distinctions from previous image steganography methods are majorly the robustness and losslessness against diverse attacks. Experiments over diverse public datasets have manifested the superior ability of anti-steganalysis.



## **5. Switching One-Versus-the-Rest Loss to Increase the Margin of Logits for Adversarial Robustness**

cs.LG

20 pages, 16 figures

**SubmitDate**: 2022-07-21    [paper-pdf](http://arxiv.org/pdf/2207.10283v1)

**Authors**: Sekitoshi Kanai, Shin'ya Yamaguchi, Masanori Yamada, Hiroshi Takahashi, Yasutoshi Ida

**Abstracts**: Defending deep neural networks against adversarial examples is a key challenge for AI safety. To improve the robustness effectively, recent methods focus on important data points near the decision boundary in adversarial training. However, these methods are vulnerable to Auto-Attack, which is an ensemble of parameter-free attacks for reliable evaluation. In this paper, we experimentally investigate the causes of their vulnerability and find that existing methods reduce margins between logits for the true label and the other labels while keeping their gradient norms non-small values. Reduced margins and non-small gradient norms cause their vulnerability since the largest logit can be easily flipped by the perturbation. Our experiments also show that the histogram of the logit margins has two peaks, i.e., small and large logit margins. From the observations, we propose switching one-versus-the-rest loss (SOVR), which uses one-versus-the-rest loss when data have small logit margins so that it increases the margins. We find that SOVR increases logit margins more than existing methods while keeping gradient norms small and outperforms them in terms of the robustness against Auto-Attack.



## **6. FOCUS: Fairness via Agent-Awareness for Federated Learning on Heterogeneous Data**

cs.LG

**SubmitDate**: 2022-07-21    [paper-pdf](http://arxiv.org/pdf/2207.10265v1)

**Authors**: Wenda Chu, Chulin Xie, Boxin Wang, Linyi Li, Lang Yin, Han Zhao, Bo Li

**Abstracts**: Federated learning (FL) provides an effective paradigm to train machine learning models over distributed data with privacy protection. However, recent studies show that FL is subject to various security, privacy, and fairness threats due to the potentially malicious and heterogeneous local agents. For instance, it is vulnerable to local adversarial agents who only contribute low-quality data, with the goal of harming the performance of those with high-quality data. This kind of attack hence breaks existing definitions of fairness in FL that mainly focus on a certain notion of performance parity. In this work, we aim to address this limitation and propose a formal definition of fairness via agent-awareness for FL (FAA), which takes the heterogeneous data contributions of local agents into account. In addition, we propose a fair FL training algorithm based on agent clustering (FOCUS) to achieve FAA. Theoretically, we prove the convergence and optimality of FOCUS under mild conditions for linear models and general convex loss functions with bounded smoothness. We also prove that FOCUS always achieves higher fairness measured by FAA compared with standard FedAvg protocol under both linear models and general convex loss functions. Empirically, we evaluate FOCUS on four datasets, including synthetic data, images, and texts under different settings, and we show that FOCUS achieves significantly higher fairness based on FAA while maintaining similar or even higher prediction accuracy compared with FedAvg.



## **7. Illusionary Attacks on Sequential Decision Makers and Countermeasures**

cs.AI

**SubmitDate**: 2022-07-20    [paper-pdf](http://arxiv.org/pdf/2207.10170v1)

**Authors**: Tim Franzmeyer, João F. Henriques, Jakob N. Foerster, Philip H. S. Torr, Adel Bibi, Christian Schroeder de Witt

**Abstracts**: Autonomous intelligent agents deployed to the real-world need to be robust against adversarial attacks on sensory inputs. Existing work in reinforcement learning focuses on minimum-norm perturbation attacks, which were originally introduced to mimic a notion of perceptual invariance in computer vision. In this paper, we note that such minimum-norm perturbation attacks can be trivially detected by victim agents, as these result in observation sequences that are not consistent with the victim agent's actions. Furthermore, many real-world agents, such as physical robots, commonly operate under human supervisors, which are not susceptible to such perturbation attacks. As a result, we propose to instead focus on illusionary attacks, a novel form of attack that is consistent with the world model of the victim agent. We provide a formal definition of this novel attack framework, explore its characteristics under a variety of conditions, and conclude that agents must seek realism feedback to be robust to illusionary attacks.



## **8. PFMC: a parallel symbolic model checker for security protocol verification**

cs.LO

**SubmitDate**: 2022-07-20    [paper-pdf](http://arxiv.org/pdf/2207.09895v1)

**Authors**: Alex James, Alwen Tiu, Nisansala Yatapanage

**Abstracts**: We present an investigation into the design and implementation of a parallel model checker for security protocol verification that is based on a symbolic model of the adversary, where instantiations of concrete terms and messages are avoided until needed to resolve a particular assertion. We propose to build on this naturally lazy approach to parallelise this symbolic state exploration and evaluation. We utilise the concept of strategies in Haskell, which abstracts away from the low-level details of thread management and modularly adds parallel evaluation strategies (encapsulated as a monad in Haskell). We build on an existing symbolic model checker, OFMC, which is already implemented in Haskell. We show that there is a very significant speed up of around 3-5 times improvement when moving from the original single-threaded implementation of OFMC to our multi-threaded version, for both the Dolev-Yao attacker model and more general algebraic attacker models. We identify several issues in parallelising the model checker: among others, controlling growth of memory consumption, balancing lazy vs strict evaluation, and achieving an optimal granularity of parallelism.



## **9. Adaptive Image Transformations for Transfer-based Adversarial Attack**

cs.CV

34 pages, 7 figures, 11 tables. Accepted by ECCV2022

**SubmitDate**: 2022-07-20    [paper-pdf](http://arxiv.org/pdf/2111.13844v3)

**Authors**: Zheng Yuan, Jie Zhang, Shiguang Shan

**Abstracts**: Adversarial attacks provide a good way to study the robustness of deep learning models. One category of methods in transfer-based black-box attack utilizes several image transformation operations to improve the transferability of adversarial examples, which is effective, but fails to take the specific characteristic of the input image into consideration. In this work, we propose a novel architecture, called Adaptive Image Transformation Learner (AITL), which incorporates different image transformation operations into a unified framework to further improve the transferability of adversarial examples. Unlike the fixed combinational transformations used in existing works, our elaborately designed transformation learner adaptively selects the most effective combination of image transformations specific to the input image. Extensive experiments on ImageNet demonstrate that our method significantly improves the attack success rates on both normally trained models and defense models under various settings.



## **10. On the Robustness of Quality Measures for GANs**

cs.LG

Accepted at the European Conference in Computer Vision (ECCV 2022)

**SubmitDate**: 2022-07-20    [paper-pdf](http://arxiv.org/pdf/2201.13019v2)

**Authors**: Motasem Alfarra, Juan C. Pérez, Anna Frühstück, Philip H. S. Torr, Peter Wonka, Bernard Ghanem

**Abstracts**: This work evaluates the robustness of quality measures of generative models such as Inception Score (IS) and Fr\'echet Inception Distance (FID). Analogous to the vulnerability of deep models against a variety of adversarial attacks, we show that such metrics can also be manipulated by additive pixel perturbations. Our experiments indicate that one can generate a distribution of images with very high scores but low perceptual quality. Conversely, one can optimize for small imperceptible perturbations that, when added to real world images, deteriorate their scores. We further extend our evaluation to generative models themselves, including the state of the art network StyleGANv2. We show the vulnerability of both the generative model and the FID against additive perturbations in the latent space. Finally, we show that the FID can be robustified by simply replacing the standard Inception with a robust Inception. We validate the effectiveness of the robustified metric through extensive experiments, showing it is more robust against manipulation.



## **11. On the Versatile Uses of Partial Distance Correlation in Deep Learning**

cs.CV

**SubmitDate**: 2022-07-20    [paper-pdf](http://arxiv.org/pdf/2207.09684v1)

**Authors**: Xingjian Zhen, Zihang Meng, Rudrasis Chakraborty, Vikas Singh

**Abstracts**: Comparing the functional behavior of neural network models, whether it is a single network over time or two (or more networks) during or post-training, is an essential step in understanding what they are learning (and what they are not), and for identifying strategies for regularization or efficiency improvements. Despite recent progress, e.g., comparing vision transformers to CNNs, systematic comparison of function, especially across different networks, remains difficult and is often carried out layer by layer. Approaches such as canonical correlation analysis (CCA) are applicable in principle, but have been sparingly used so far. In this paper, we revisit a (less widely known) from statistics, called distance correlation (and its partial variant), designed to evaluate correlation between feature spaces of different dimensions. We describe the steps necessary to carry out its deployment for large scale models -- this opens the door to a surprising array of applications ranging from conditioning one deep model w.r.t. another, learning disentangled representations as well as optimizing diverse models that would directly be more robust to adversarial attacks. Our experiments suggest a versatile regularizer (or constraint) with many advantages, which avoids some of the common difficulties one faces in such analyses. Code is at https://github.com/zhenxingjian/Partial_Distance_Correlation.



## **12. Detecting Textual Adversarial Examples through Randomized Substitution and Vote**

cs.CL

Accepted by UAI 2022, code is avaliable at  https://github.com/JHL-HUST/RSV

**SubmitDate**: 2022-07-20    [paper-pdf](http://arxiv.org/pdf/2109.05698v2)

**Authors**: Xiaosen Wang, Yifeng Xiong, Kun He

**Abstracts**: A line of work has shown that natural text processing models are vulnerable to adversarial examples. Correspondingly, various defense methods are proposed to mitigate the threat of textual adversarial examples, eg, adversarial training, input transformations, detection, etc. In this work, we treat the optimization process for synonym substitution based textual adversarial attacks as a specific sequence of word replacement, in which each word mutually influences other words. We identify that we could destroy such mutual interaction and eliminate the adversarial perturbation by randomly substituting a word with its synonyms. Based on this observation, we propose a novel textual adversarial example detection method, termed Randomized Substitution and Vote (RS&V), which votes the prediction label by accumulating the logits of k samples generated by randomly substituting the words in the input text with synonyms. The proposed RS&V is generally applicable to any existing neural networks without modification on the architecture or extra training, and it is orthogonal to prior work on making the classification network itself more robust. Empirical evaluations on three benchmark datasets demonstrate that our RS&V could detect the textual adversarial examples more successfully than the existing detection methods while maintaining the high classification accuracy on benign samples.



## **13. Diversified Adversarial Attacks based on Conjugate Gradient Method**

cs.LG

Proceedings of the 39th International Conference on Machine Learning  (ICML 2022)

**SubmitDate**: 2022-07-19    [paper-pdf](http://arxiv.org/pdf/2206.09628v2)

**Authors**: Keiichiro Yamamura, Haruki Sato, Nariaki Tateiwa, Nozomi Hata, Toru Mitsutake, Issa Oe, Hiroki Ishikura, Katsuki Fujisawa

**Abstracts**: Deep learning models are vulnerable to adversarial examples, and adversarial attacks used to generate such examples have attracted considerable research interest. Although existing methods based on the steepest descent have achieved high attack success rates, ill-conditioned problems occasionally reduce their performance. To address this limitation, we utilize the conjugate gradient (CG) method, which is effective for this type of problem, and propose a novel attack algorithm inspired by the CG method, named the Auto Conjugate Gradient (ACG) attack. The results of large-scale evaluation experiments conducted on the latest robust models show that, for most models, ACG was able to find more adversarial examples with fewer iterations than the existing SOTA algorithm Auto-PGD (APGD). We investigated the difference in search performance between ACG and APGD in terms of diversification and intensification, and define a measure called Diversity Index (DI) to quantify the degree of diversity. From the analysis of the diversity using this index, we show that the more diverse search of the proposed method remarkably improves its attack success rate.



## **14. Towards Robust Multivariate Time-Series Forecasting: Adversarial Attacks and Defense Mechanisms**

cs.LG

**SubmitDate**: 2022-07-19    [paper-pdf](http://arxiv.org/pdf/2207.09572v1)

**Authors**: Linbo Liu, Youngsuk Park, Trong Nghia Hoang, Hilaf Hasson, Jun Huan

**Abstracts**: As deep learning models have gradually become the main workhorse of time series forecasting, the potential vulnerability under adversarial attacks to forecasting and decision system accordingly has emerged as a main issue in recent years. Albeit such behaviors and defense mechanisms started to be investigated for the univariate time series forecasting, there are still few studies regarding the multivariate forecasting which is often preferred due to its capacity to encode correlations between different time series. In this work, we study and design adversarial attack on multivariate probabilistic forecasting models, taking into consideration attack budget constraints and the correlation architecture between multiple time series. Specifically, we investigate a sparse indirect attack that hurts the prediction of an item (time series) by only attacking the history of a small number of other items to save attacking cost. In order to combat these attacks, we also develop two defense strategies. First, we adopt randomized smoothing to multivariate time series scenario and verify its effectiveness via empirical experiments. Second, we leverage a sparse attacker to enable end-to-end adversarial training that delivers robust probabilistic forecasters. Extensive experiments on real dataset confirm that our attack schemes are powerful and our defend algorithms are more effective compared with other baseline defense mechanisms.



## **15. Increasing the Cost of Model Extraction with Calibrated Proof of Work**

cs.CR

Published as a conference paper at ICLR 2022 (Spotlight - 5% of  submitted papers)

**SubmitDate**: 2022-07-19    [paper-pdf](http://arxiv.org/pdf/2201.09243v2)

**Authors**: Adam Dziedzic, Muhammad Ahmad Kaleem, Yu Shen Lu, Nicolas Papernot

**Abstracts**: In model extraction attacks, adversaries can steal a machine learning model exposed via a public API by repeatedly querying it and adjusting their own model based on obtained predictions. To prevent model stealing, existing defenses focus on detecting malicious queries, truncating, or distorting outputs, thus necessarily introducing a tradeoff between robustness and model utility for legitimate users. Instead, we propose to impede model extraction by requiring users to complete a proof-of-work before they can read the model's predictions. This deters attackers by greatly increasing (even up to 100x) the computational effort needed to leverage query access for model extraction. Since we calibrate the effort required to complete the proof-of-work to each query, this only introduces a slight overhead for regular users (up to 2x). To achieve this, our calibration applies tools from differential privacy to measure the information revealed by a query. Our method requires no modification of the victim model and can be applied by machine learning practitioners to guard their publicly exposed models against being easily stolen.



## **16. Assaying Out-Of-Distribution Generalization in Transfer Learning**

cs.LG

**SubmitDate**: 2022-07-19    [paper-pdf](http://arxiv.org/pdf/2207.09239v1)

**Authors**: Florian Wenzel, Andrea Dittadi, Peter Vincent Gehler, Carl-Johann Simon-Gabriel, Max Horn, Dominik Zietlow, David Kernert, Chris Russell, Thomas Brox, Bernt Schiele, Bernhard Schölkopf, Francesco Locatello

**Abstracts**: Since out-of-distribution generalization is a generally ill-posed problem, various proxy targets (e.g., calibration, adversarial robustness, algorithmic corruptions, invariance across shifts) were studied across different research programs resulting in different recommendations. While sharing the same aspirational goal, these approaches have never been tested under the same experimental conditions on real data. In this paper, we take a unified view of previous work, highlighting message discrepancies that we address empirically, and providing recommendations on how to measure the robustness of a model and how to improve it. To this end, we collect 172 publicly available dataset pairs for training and out-of-distribution evaluation of accuracy, calibration error, adversarial attacks, environment invariance, and synthetic corruptions. We fine-tune over 31k networks, from nine different architectures in the many- and few-shot setting. Our findings confirm that in- and out-of-distribution accuracies tend to increase jointly, but show that their relation is largely dataset-dependent, and in general more nuanced and more complex than posited by previous, smaller scale studies.



## **17. MUD-PQFed: Towards Malicious User Detection in Privacy-Preserving Quantized Federated Learning**

cs.CR

13 pages,13 figures

**SubmitDate**: 2022-07-19    [paper-pdf](http://arxiv.org/pdf/2207.09080v1)

**Authors**: Hua Ma, Qun Li, Yifeng Zheng, Zhi Zhang, Xiaoning Liu, Yansong Gao, Said F. Al-Sarawi, Derek Abbott

**Abstracts**: Federated Learning (FL), a distributed machine learning paradigm, has been adapted to mitigate privacy concerns for customers. Despite their appeal, there are various inference attacks that can exploit shared-plaintext model updates to embed traces of customer private information, leading to serious privacy concerns. To alleviate this privacy issue, cryptographic techniques such as Secure Multi-Party Computation and Homomorphic Encryption have been used for privacy-preserving FL. However, such security issues in privacy-preserving FL are poorly elucidated and underexplored. This work is the first attempt to elucidate the triviality of performing model corruption attacks on privacy-preserving FL based on lightweight secret sharing. We consider scenarios in which model updates are quantized to reduce communication overhead in this case, where an adversary can simply provide local parameters outside the legal range to corrupt the model. We then propose the MUD-PQFed protocol, which can precisely detect malicious clients performing attacks and enforce fair penalties. By removing the contributions of detected malicious clients, the global model utility is preserved to be comparable to the baseline global model without the attack. Extensive experiments validate effectiveness in maintaining baseline accuracy and detecting malicious clients in a fine-grained manner



## **18. $\ell_\infty$-Robustness and Beyond: Unleashing Efficient Adversarial Training**

cs.LG

Accepted to the 17th European Conference on Computer Vision (ECCV  2022)

**SubmitDate**: 2022-07-19    [paper-pdf](http://arxiv.org/pdf/2112.00378v2)

**Authors**: Hadi M. Dolatabadi, Sarah Erfani, Christopher Leckie

**Abstracts**: Neural networks are vulnerable to adversarial attacks: adding well-crafted, imperceptible perturbations to their input can modify their output. Adversarial training is one of the most effective approaches in training robust models against such attacks. However, it is much slower than vanilla training of neural networks since it needs to construct adversarial examples for the entire training data at every iteration, hampering its effectiveness. Recently, Fast Adversarial Training (FAT) was proposed that can obtain robust models efficiently. However, the reasons behind its success are not fully understood, and more importantly, it can only train robust models for $\ell_\infty$-bounded attacks as it uses FGSM during training. In this paper, by leveraging the theory of coreset selection, we show how selecting a small subset of training data provides a general, more principled approach toward reducing the time complexity of robust training. Unlike existing methods, our approach can be adapted to a wide variety of training objectives, including TRADES, $\ell_p$-PGD, and Perceptual Adversarial Training (PAT). Our experimental results indicate that our approach speeds up adversarial training by 2-3 times while experiencing a slight reduction in the clean and robust accuracy.



## **19. Decorrelative Network Architecture for Robust Electrocardiogram Classification**

cs.LG

12 pages, 6 figures

**SubmitDate**: 2022-07-19    [paper-pdf](http://arxiv.org/pdf/2207.09031v1)

**Authors**: Christopher Wiedeman, Ge Wang

**Abstracts**: Artificial intelligence has made great progresses in medical data analysis, but the lack of robustness and interpretability has kept these methods from being widely deployed. In particular, data-driven models are vulnerable to adversarial attacks, which are small, targeted perturbations that dramatically degrade model performance. As a recent example, while deep learning has shown impressive performance in electrocardiogram (ECG) classification, Han et al. crafted realistic perturbations that fooled the network 74% of the time [2020]. Current adversarial defense paradigms are computationally intensive and impractical for many high dimensional problems. Previous research indicates that a network vulnerability is related to the features learned during training. We propose a novel approach based on ensemble decorrelation and Fourier partitioning for training parallel network arms into a decorrelated architecture to learn complementary features, significantly reducing the chance of a perturbation fooling all arms of the deep learning model. We test our approach in ECG classification, demonstrating a much-improved 77.2% chance of at least one correct network arm on the strongest adversarial attack tested, in contrast to a 21.7% chance from a comparable ensemble. Our approach does not require expensive optimization with adversarial samples, and thus can be scaled to large problems. These methods can easily be applied to other tasks for improved network robustness.



## **20. Multi-step domain adaptation by adversarial attack to $\mathcal{H} Δ\mathcal{H}$-divergence**

cs.LG

**SubmitDate**: 2022-07-18    [paper-pdf](http://arxiv.org/pdf/2207.08948v1)

**Authors**: Arip Asadulaev, Alexander Panfilov, Andrey Filchenkov

**Abstracts**: Adversarial examples are transferable between different models. In our paper, we propose to use this property for multi-step domain adaptation. In unsupervised domain adaptation settings, we demonstrate that replacing the source domain with adversarial examples to $\mathcal{H} \Delta \mathcal{H}$-divergence can improve source classifier accuracy on the target domain. Our method can be connected to most domain adaptation techniques. We conducted a range of experiments and achieved improvement in accuracy on Digits and Office-Home datasets.



## **21. Benchmarking Machine Learning Robustness in Covid-19 Genome Sequence Classification**

q-bio.GN

**SubmitDate**: 2022-07-18    [paper-pdf](http://arxiv.org/pdf/2207.08898v1)

**Authors**: Sarwan Ali, Bikram Sahoo, Alexander Zelikovskiy, Pin-Yu Chen, Murray Patterson

**Abstracts**: The rapid spread of the COVID-19 pandemic has resulted in an unprecedented amount of sequence data of the SARS-CoV-2 genome -- millions of sequences and counting. This amount of data, while being orders of magnitude beyond the capacity of traditional approaches to understanding the diversity, dynamics, and evolution of viruses is nonetheless a rich resource for machine learning (ML) approaches as alternatives for extracting such important information from these data. It is of hence utmost importance to design a framework for testing and benchmarking the robustness of these ML models.   This paper makes the first effort (to our knowledge) to benchmark the robustness of ML models by simulating biological sequences with errors. In this paper, we introduce several ways to perturb SARS-CoV-2 genome sequences to mimic the error profiles of common sequencing platforms such as Illumina and PacBio. We show from experiments on a wide array of ML models that some simulation-based approaches are more robust (and accurate) than others for specific embedding methods to certain adversarial attacks to the input sequences. Our benchmarking framework may assist researchers in properly assessing different ML models and help them understand the behavior of the SARS-CoV-2 virus or avoid possible future pandemics.



## **22. Prior-Guided Adversarial Initialization for Fast Adversarial Training**

cs.CV

ECCV 2022

**SubmitDate**: 2022-07-18    [paper-pdf](http://arxiv.org/pdf/2207.08859v1)

**Authors**: Xiaojun Jia, Yong Zhang, Xingxing Wei, Baoyuan Wu, Ke Ma, Jue Wang, Xiaochun Cao

**Abstracts**: Fast adversarial training (FAT) effectively improves the efficiency of standard adversarial training (SAT). However, initial FAT encounters catastrophic overfitting, i.e.,the robust accuracy against adversarial attacks suddenly and dramatically decreases. Though several FAT variants spare no effort to prevent overfitting, they sacrifice much calculation cost. In this paper, we explore the difference between the training processes of SAT and FAT and observe that the attack success rate of adversarial examples (AEs) of FAT gets worse gradually in the late training stage, resulting in overfitting. The AEs are generated by the fast gradient sign method (FGSM) with a zero or random initialization. Based on the observation, we propose a prior-guided FGSM initialization method to avoid overfitting after investigating several initialization strategies, improving the quality of the AEs during the whole training process. The initialization is formed by leveraging historically generated AEs without additional calculation cost. We further provide a theoretical analysis for the proposed initialization method. We also propose a simple yet effective regularizer based on the prior-guided initialization,i.e., the currently generated perturbation should not deviate too much from the prior-guided initialization. The regularizer adopts both historical and current adversarial perturbations to guide the model learning. Evaluations on four datasets demonstrate that the proposed method can prevent catastrophic overfitting and outperform state-of-the-art FAT methods. The code is released at https://github.com/jiaxiaojunQAQ/FGSM-PGI.



## **23. Adversarial Pixel Restoration as a Pretext Task for Transferable Perturbations**

cs.CV

**SubmitDate**: 2022-07-18    [paper-pdf](http://arxiv.org/pdf/2207.08803v1)

**Authors**: Hashmat Shadab Malik, Shahina K Kunhimon, Muzammal Naseer, Salman Khan, Fahad Shahbaz Khan

**Abstracts**: Transferable adversarial attacks optimize adversaries from a pretrained surrogate model and known label space to fool the unknown black-box models. Therefore, these attacks are restricted by the availability of an effective surrogate model. In this work, we relax this assumption and propose Adversarial Pixel Restoration as a self-supervised alternative to train an effective surrogate model from scratch under the condition of no labels and few data samples. Our training approach is based on a min-max objective which reduces overfitting via an adversarial objective and thus optimizes for a more generalizable surrogate model. Our proposed attack is complimentary to our adversarial pixel restoration and is independent of any task specific objective as it can be launched in a self-supervised manner. We successfully demonstrate the adversarial transferability of our approach to Vision Transformers as well as Convolutional Neural Networks for the tasks of classification, object detection, and video segmentation. Our codes & pre-trained surrogate models are available at: https://github.com/HashmatShadab/APR



## **24. Are Vision Transformers Robust to Patch Perturbations?**

cs.CV

**SubmitDate**: 2022-07-18    [paper-pdf](http://arxiv.org/pdf/2111.10659v2)

**Authors**: Jindong Gu, Volker Tresp, Yao Qin

**Abstracts**: Recent advances in Vision Transformer (ViT) have demonstrated its impressive performance in image classification, which makes it a promising alternative to Convolutional Neural Network (CNN). Unlike CNNs, ViT represents an input image as a sequence of image patches. The patch-based input image representation makes the following question interesting: How does ViT perform when individual input image patches are perturbed with natural corruptions or adversarial perturbations, compared to CNNs? In this work, we study the robustness of ViT to patch-wise perturbations. Surprisingly, we find that ViTs are more robust to naturally corrupted patches than CNNs, whereas they are more vulnerable to adversarial patches. Furthermore, we discover that the attention mechanism greatly affects the robustness of vision transformers. Specifically, the attention module can help improve the robustness of ViT by effectively ignoring natural corrupted patches. However, when ViTs are attacked by an adversary, the attention mechanism can be easily fooled to focus more on the adversarially perturbed patches and cause a mistake. Based on our analysis, we propose a simple temperature-scaling based method to improve the robustness of ViT against adversarial patches. Extensive qualitative and quantitative experiments are performed to support our findings, understanding, and improvement of ViT robustness to patch-wise perturbations across a set of transformer-based architectures.



## **25. Authentication Attacks on Projection-based Cancelable Biometric Schemes**

cs.CR

arXiv admin note: text overlap with arXiv:1910.01389 by other authors

**SubmitDate**: 2022-07-18    [paper-pdf](http://arxiv.org/pdf/2110.15163v6)

**Authors**: Axel Durbet, Pascal Lafourcade, Denis Migdal, Kevin Thiry-Atighehchi, Paul-Marie Grollemund

**Abstracts**: Cancelable biometric schemes aim at generating secure biometric templates by combining user specific tokens, such as password, stored secret or salt, along with biometric data. This type of transformation is constructed as a composition of a biometric transformation with a feature extraction algorithm. The security requirements of cancelable biometric schemes concern the irreversibility, unlinkability and revocability of templates, without losing in accuracy of comparison. While several schemes were recently attacked regarding these requirements, full reversibility of such a composition in order to produce colliding biometric characteristics, and specifically presentation attacks, were never demonstrated to the best of our knowledge. In this paper, we formalize these attacks for a traditional cancelable scheme with the help of integer linear programming (ILP) and quadratically constrained quadratic programming (QCQP). Solving these optimization problems allows an adversary to slightly alter its fingerprint image in order to impersonate any individual. Moreover, in an even more severe scenario, it is possible to simultaneously impersonate several individuals.



## **26. Detection of Poisoning Attacks with Anomaly Detection in Federated Learning for Healthcare Applications: A Machine Learning Approach**

cs.LG

We will updated this article soon

**SubmitDate**: 2022-07-18    [paper-pdf](http://arxiv.org/pdf/2207.08486v1)

**Authors**: Ali Raza, Shujun Li, Kim-Phuc Tran, Ludovic Koehl

**Abstracts**: The application of Federated Learning (FL) is steadily increasing, especially in privacy-aware applications, such as healthcare. However, its applications have been limited by security concerns due to various adversarial attacks, such as poisoning attacks (model and data poisoning). Such attacks attempt to poison the local models and data to manipulate the global models in order to obtain undue benefits and malicious use. Traditional methods of data auditing to mitigate poisoning attacks find their limited applications in FL because the edge devices never share their raw data directly due to privacy concerns, and are globally distributed with no insight into their training data. Thereafter, it is challenging to develop appropriate strategies to address such attacks and minimize their impact on the global model in federated learning. In order to address such challenges in FL, we proposed a novel framework to detect poisoning attacks using deep neural networks and support vector machines, in the form of anomaly without acquiring any direct access or information about the underlying training data of local edge devices. We illustrate and evaluate the proposed framework using different state of art poisoning attacks for two different healthcare applications: Electrocardiograph classification and human activity recognition. Our experimental analysis shows that the proposed method can efficiently detect poisoning attacks and can remove the identified poisoned updated from the global aggregation. Thereafter can increase the performance of the federated global.



## **27. Towards Automated Classification of Attackers' TTPs by combining NLP with ML Techniques**

cs.CR

**SubmitDate**: 2022-07-18    [paper-pdf](http://arxiv.org/pdf/2207.08478v1)

**Authors**: Clemens Sauerwein, Alexander Pfohl

**Abstracts**: The increasingly sophisticated and growing number of threat actors along with the sheer speed at which cyber attacks unfold, make timely identification of attacks imperative to an organisations' security. Consequently, persons responsible for security employ a large variety of information sources concerning emerging attacks, attackers' course of actions or indicators of compromise. However, a vast amount of the needed security information is available in unstructured textual form, which complicates the automated and timely extraction of attackers' Tactics, Techniques and Procedures (TTPs). In order to address this problem we systematically evaluate and compare different Natural Language Processing (NLP) and machine learning techniques used for security information extraction in research. Based on our investigations we propose a data processing pipeline that automatically classifies unstructured text according to attackers' tactics and techniques derived from a knowledge base of adversary tactics, techniques and procedures.



## **28. A Perturbation-Constrained Adversarial Attack for Evaluating the Robustness of Optical Flow**

cs.CV

Accepted at the European Conference on Computer Vision (ECCV) 2022

**SubmitDate**: 2022-07-18    [paper-pdf](http://arxiv.org/pdf/2203.13214v2)

**Authors**: Jenny Schmalfuss, Philipp Scholze, Andrés Bruhn

**Abstracts**: Recent optical flow methods are almost exclusively judged in terms of accuracy, while their robustness is often neglected. Although adversarial attacks offer a useful tool to perform such an analysis, current attacks on optical flow methods focus on real-world attacking scenarios rather than a worst case robustness assessment. Hence, in this work, we propose a novel adversarial attack - the Perturbation-Constrained Flow Attack (PCFA) - that emphasizes destructivity over applicability as a real-world attack. PCFA is a global attack that optimizes adversarial perturbations to shift the predicted flow towards a specified target flow, while keeping the L2 norm of the perturbation below a chosen bound. Our experiments demonstrate PCFA's applicability in white- and black-box settings, and show it finds stronger adversarial samples than previous attacks. Based on these strong samples, we provide the first joint ranking of optical flow methods considering both prediction quality and adversarial robustness, which reveals state-of-the-art methods to be particularly vulnerable. Code is available at https://github.com/cv-stuttgart/PCFA.



## **29. Universal Adversarial Examples in Remote Sensing: Methodology and Benchmark**

cs.CV

**SubmitDate**: 2022-07-17    [paper-pdf](http://arxiv.org/pdf/2202.07054v3)

**Authors**: Yonghao Xu, Pedram Ghamisi

**Abstracts**: Deep neural networks have achieved great success in many important remote sensing tasks. Nevertheless, their vulnerability to adversarial examples should not be neglected. In this study, we systematically analyze the universal adversarial examples in remote sensing data for the first time, without any knowledge from the victim model. Specifically, we propose a novel black-box adversarial attack method, namely Mixup-Attack, and its simple variant Mixcut-Attack, for remote sensing data. The key idea of the proposed methods is to find common vulnerabilities among different networks by attacking the features in the shallow layer of a given surrogate model. Despite their simplicity, the proposed methods can generate transferable adversarial examples that deceive most of the state-of-the-art deep neural networks in both scene classification and semantic segmentation tasks with high success rates. We further provide the generated universal adversarial examples in the dataset named UAE-RS, which is the first dataset that provides black-box adversarial samples in the remote sensing field. We hope UAE-RS may serve as a benchmark that helps researchers to design deep neural networks with strong resistance toward adversarial attacks in the remote sensing field. Codes and the UAE-RS dataset are available online (https://github.com/YonghaoXu/UAE-RS).



## **30. Watermark Vaccine: Adversarial Attacks to Prevent Watermark Removal**

cs.CV

ECCV 2022

**SubmitDate**: 2022-07-17    [paper-pdf](http://arxiv.org/pdf/2207.08178v1)

**Authors**: Xinwei Liu, Jian Liu, Yang Bai, Jindong Gu, Tao Chen, Xiaojun Jia, Xiaochun Cao

**Abstracts**: As a common security tool, visible watermarking has been widely applied to protect copyrights of digital images. However, recent works have shown that visible watermarks can be removed by DNNs without damaging their host images. Such watermark-removal techniques pose a great threat to the ownership of images. Inspired by the vulnerability of DNNs on adversarial perturbations, we propose a novel defence mechanism by adversarial machine learning for good. From the perspective of the adversary, blind watermark-removal networks can be posed as our target models; then we actually optimize an imperceptible adversarial perturbation on the host images to proactively attack against watermark-removal networks, dubbed Watermark Vaccine. Specifically, two types of vaccines are proposed. Disrupting Watermark Vaccine (DWV) induces to ruin the host image along with watermark after passing through watermark-removal networks. In contrast, Inerasable Watermark Vaccine (IWV) works in another fashion of trying to keep the watermark not removed and still noticeable. Extensive experiments demonstrate the effectiveness of our DWV/IWV in preventing watermark removal, especially on various watermark removal networks.



## **31. Modeling Adversarial Noise for Adversarial Training**

cs.LG

**SubmitDate**: 2022-07-17    [paper-pdf](http://arxiv.org/pdf/2109.09901v5)

**Authors**: Dawei Zhou, Nannan Wang, Bo Han, Tongliang Liu

**Abstracts**: Deep neural networks have been demonstrated to be vulnerable to adversarial noise, promoting the development of defense against adversarial attacks. Motivated by the fact that adversarial noise contains well-generalizing features and that the relationship between adversarial data and natural data can help infer natural data and make reliable predictions, in this paper, we study to model adversarial noise by learning the transition relationship between adversarial labels (i.e. the flipped labels used to generate adversarial data) and natural labels (i.e. the ground truth labels of the natural data). Specifically, we introduce an instance-dependent transition matrix to relate adversarial labels and natural labels, which can be seamlessly embedded with the target model (enabling us to model stronger adaptive adversarial noise). Empirical evaluations demonstrate that our method could effectively improve adversarial accuracy.



## **32. Automated Repair of Neural Networks**

cs.LG

Code and results are available at  https://github.com/dorcoh/NNSynthesizer

**SubmitDate**: 2022-07-17    [paper-pdf](http://arxiv.org/pdf/2207.08157v1)

**Authors**: Dor Cohen, Ofer Strichman

**Abstracts**: Over the last decade, Neural Networks (NNs) have been widely used in numerous applications including safety-critical ones such as autonomous systems. Despite their emerging adoption, it is well known that NNs are susceptible to Adversarial Attacks. Hence, it is highly important to provide guarantees that such systems work correctly. To remedy these issues we introduce a framework for repairing unsafe NNs w.r.t. safety specification, that is by utilizing satisfiability modulo theories (SMT) solvers. Our method is able to search for a new, safe NN representation, by modifying only a few of its weight values. In addition, our technique attempts to maximize the similarity to original network with regard to its decision boundaries. We perform extensive experiments which demonstrate the capability of our proposed framework to yield safe NNs w.r.t. the Adversarial Robustness property, with only a mild loss of accuracy (in terms of similarity). Moreover, we compare our method with a naive baseline to empirically prove its effectiveness. To conclude, we provide an algorithm to automatically repair NNs given safety properties, and suggest a few heuristics to improve its computational performance. Currently, by following this approach we are capable of producing small-sized (i.e., with up to few hundreds of parameters) correct NNs, composed of the piecewise linear ReLU activation function. Nevertheless, our framework is general in the sense that it can synthesize NNs w.r.t. any decidable fragment of first-order logic specification.



## **33. Achieve Optimal Adversarial Accuracy for Adversarial Deep Learning using Stackelberg Game**

cs.LG

**SubmitDate**: 2022-07-17    [paper-pdf](http://arxiv.org/pdf/2207.08137v1)

**Authors**: Xiao-Shan Gao, Shuang Liu, Lijia Yu

**Abstracts**: Adversarial deep learning is to train robust DNNs against adversarial attacks, which is one of the major research focuses of deep learning. Game theory has been used to answer some of the basic questions about adversarial deep learning such as the existence of a classifier with optimal robustness and the existence of optimal adversarial samples for a given class of classifiers. In most previous work, adversarial deep learning was formulated as a simultaneous game and the strategy spaces are assumed to be certain probability distributions in order for the Nash equilibrium to exist. But, this assumption is not applicable to the practical situation. In this paper, we give answers to these basic questions for the practical case where the classifiers are DNNs with a given structure, by formulating the adversarial deep learning as sequential games. The existence of Stackelberg equilibria for these games are proved. Furthermore, it is shown that the equilibrium DNN has the largest adversarial accuracy among all DNNs with the same structure, when Carlini-Wagner's margin loss is used. Trade-off between robustness and accuracy in adversarial deep learning is also studied from game theoretical aspect.



## **34. Threat Model-Agnostic Adversarial Defense using Diffusion Models**

cs.CV

**SubmitDate**: 2022-07-17    [paper-pdf](http://arxiv.org/pdf/2207.08089v1)

**Authors**: Tsachi Blau, Roy Ganz, Bahjat Kawar, Alex Bronstein, Michael Elad

**Abstracts**: Deep Neural Networks (DNNs) are highly sensitive to imperceptible malicious perturbations, known as adversarial attacks. Following the discovery of this vulnerability in real-world imaging and vision applications, the associated safety concerns have attracted vast research attention, and many defense techniques have been developed. Most of these defense methods rely on adversarial training (AT) -- training the classification network on images perturbed according to a specific threat model, which defines the magnitude of the allowed modification. Although AT leads to promising results, training on a specific threat model fails to generalize to other types of perturbations. A different approach utilizes a preprocessing step to remove the adversarial perturbation from the attacked image. In this work, we follow the latter path and aim to develop a technique that leads to robust classifiers across various realizations of threat models. To this end, we harness the recent advances in stochastic generative modeling, and means to leverage these for sampling from conditional distributions. Our defense relies on an addition of Gaussian i.i.d noise to the attacked image, followed by a pretrained diffusion process -- an architecture that performs a stochastic iterative process over a denoising network, yielding a high perceptual quality denoised outcome. The obtained robustness with this stochastic preprocessing step is validated through extensive experiments on the CIFAR-10 dataset, showing that our method outperforms the leading defense methods under various threat models.



## **35. DIMBA: Discretely Masked Black-Box Attack in Single Object Tracking**

cs.CV

**SubmitDate**: 2022-07-17    [paper-pdf](http://arxiv.org/pdf/2207.08044v1)

**Authors**: Xiangyu Yin, Wenjie Ruan, Jonathan Fieldsend

**Abstracts**: The adversarial attack can force a CNN-based model to produce an incorrect output by craftily manipulating human-imperceptible input. Exploring such perturbations can help us gain a deeper understanding of the vulnerability of neural networks, and provide robustness to deep learning against miscellaneous adversaries. Despite extensive studies focusing on the robustness of image, audio, and NLP, works on adversarial examples of visual object tracking -- especially in a black-box manner -- are quite lacking. In this paper, we propose a novel adversarial attack method to generate noises for single object tracking under black-box settings, where perturbations are merely added on initial frames of tracking sequences, which is difficult to be noticed from the perspective of a whole video clip. Specifically, we divide our algorithm into three components and exploit reinforcement learning for localizing important frame patches precisely while reducing unnecessary computational queries overhead. Compared to existing techniques, our method requires fewer queries on initialized frames of a video to manipulate competitive or even better attack performance. We test our algorithm in both long-term and short-term datasets, including OTB100, VOT2018, UAV123, and LaSOT. Extensive experiments demonstrate the effectiveness of our method on three mainstream types of trackers: discrimination, Siamese-based, and reinforcement learning-based trackers.



## **36. Optimal Strategic Mining Against Cryptographic Self-Selection in Proof-of-Stake**

cs.CR

31 pages, ACM EC 2022

**SubmitDate**: 2022-07-16    [paper-pdf](http://arxiv.org/pdf/2207.07996v1)

**Authors**: Matheus V. X. Ferreira, Ye Lin Sally Hahn, S. Matthew Weinberg, Catherine Yu

**Abstracts**: Cryptographic Self-Selection is a subroutine used to select a leader for modern proof-of-stake consensus protocols, such as Algorand. In cryptographic self-selection, each round $r$ has a seed $Q_r$. In round $r$, each account owner is asked to digitally sign $Q_r$, hash their digital signature to produce a credential, and then broadcast this credential to the entire network. A publicly-known function scores each credential in a manner so that the distribution of the lowest scoring credential is identical to the distribution of stake owned by each account. The user who broadcasts the lowest-scoring credential is the leader for round $r$, and their credential becomes the seed $Q_{r+1}$. Such protocols leave open the possibility of a selfish-mining style attack: a user who owns multiple accounts that each produce low-scoring credentials in round $r$ can selectively choose which ones to broadcast in order to influence the seed for round $r+1$. Indeed, the user can pre-compute their credentials for round $r+1$ for each potential seed, and broadcast only the credential (among those with a low enough score to be the leader) that produces the most favorable seed.   We consider an adversary who wishes to maximize the expected fraction of rounds in which an account they own is the leader. We show such an adversary always benefits from deviating from the intended protocol, regardless of the fraction of the stake controlled. We characterize the optimal strategy; first by proving the existence of optimal positive recurrent strategies whenever the adversary owns last than $38\%$ of the stake. Then, we provide a Markov Decision Process formulation to compute the optimal strategy.



## **37. BOSS: Bidirectional One-Shot Synthesis of Adversarial Examples**

cs.LG

**SubmitDate**: 2022-07-16    [paper-pdf](http://arxiv.org/pdf/2108.02756v2)

**Authors**: Ismail R. Alkhouri, Alvaro Velasquez, George K. Atia

**Abstracts**: The design of additive imperceptible perturbations to the inputs of deep classifiers to maximize their misclassification rates is a central focus of adversarial machine learning. An alternative approach is to synthesize adversarial examples from scratch using GAN-like structures, albeit with the use of large amounts of training data. By contrast, this paper considers one-shot synthesis of adversarial examples; the inputs are synthesized from scratch to induce arbitrary soft predictions at the output of pre-trained models, while simultaneously maintaining high similarity to specified inputs. To this end, we present a problem that encodes objectives on the distance between the desired and output distributions of the trained model and the similarity between such inputs and the synthesized examples. We prove that the formulated problem is NP-complete. Then, we advance a generative approach to the solution in which the adversarial examples are obtained as the output of a generative network whose parameters are iteratively updated by optimizing surrogate loss functions for the dual-objective. We demonstrate the generality and versatility of the framework and approach proposed through applications to the design of targeted adversarial attacks, generation of decision boundary samples, and synthesis of low confidence classification inputs. The approach is further extended to an ensemble of models with different soft output specifications. The experimental results verify that the targeted and confidence reduction attack methods developed perform on par with state-of-the-art algorithms.



## **38. Certified Neural Network Watermarks with Randomized Smoothing**

cs.LG

ICML 2022

**SubmitDate**: 2022-07-16    [paper-pdf](http://arxiv.org/pdf/2207.07972v1)

**Authors**: Arpit Bansal, Ping-yeh Chiang, Michael Curry, Rajiv Jain, Curtis Wigington, Varun Manjunatha, John P Dickerson, Tom Goldstein

**Abstracts**: Watermarking is a commonly used strategy to protect creators' rights to digital images, videos and audio. Recently, watermarking methods have been extended to deep learning models -- in principle, the watermark should be preserved when an adversary tries to copy the model. However, in practice, watermarks can often be removed by an intelligent adversary. Several papers have proposed watermarking methods that claim to be empirically resistant to different types of removal attacks, but these new techniques often fail in the face of new or better-tuned adversaries. In this paper, we propose a certifiable watermarking method. Using the randomized smoothing technique proposed in Chiang et al., we show that our watermark is guaranteed to be unremovable unless the model parameters are changed by more than a certain l2 threshold. In addition to being certifiable, our watermark is also empirically more robust compared to previous watermarking methods. Our experiments can be reproduced with code at https://github.com/arpitbansal297/Certified_Watermarks



## **39. MixTailor: Mixed Gradient Aggregation for Robust Learning Against Tailored Attacks**

cs.LG

**SubmitDate**: 2022-07-16    [paper-pdf](http://arxiv.org/pdf/2207.07941v1)

**Authors**: Ali Ramezani-Kebrya, Iman Tabrizian, Fartash Faghri, Petar Popovski

**Abstracts**: Implementations of SGD on distributed and multi-GPU systems creates new vulnerabilities, which can be identified and misused by one or more adversarial agents. Recently, it has been shown that well-known Byzantine-resilient gradient aggregation schemes are indeed vulnerable to informed attackers that can tailor the attacks (Fang et al., 2020; Xie et al., 2020b). We introduce MixTailor, a scheme based on randomization of the aggregation strategies that makes it impossible for the attacker to be fully informed. Deterministic schemes can be integrated into MixTailor on the fly without introducing any additional hyperparameters. Randomization decreases the capability of a powerful adversary to tailor its attacks, while the resulting randomized aggregation scheme is still competitive in terms of performance. For both iid and non-iid settings, we establish almost sure convergence guarantees that are both stronger and more general than those available in the literature. Our empirical studies across various datasets, attacks, and settings, validate our hypothesis and show that MixTailor successfully defends when well-known Byzantine-tolerant schemes fail.



## **40. Masked Spatial-Spectral Autoencoders Are Excellent Hyperspectral Defenders**

cs.CV

14 pages, 9 figures

**SubmitDate**: 2022-07-16    [paper-pdf](http://arxiv.org/pdf/2207.07803v1)

**Authors**: Jiahao Qi, Zhiqiang Gong, Xingyue Liu, Kangcheng Bin, Chen Chen, Yongqian Li, Wei Xue, Yu Zhang, Ping Zhong

**Abstracts**: Deep learning methodology contributes a lot to the development of hyperspectral image (HSI) analysis community. However, it also makes HSI analysis systems vulnerable to adversarial attacks. To this end, we propose a masked spatial-spectral autoencoder (MSSA) in this paper under self-supervised learning theory, for enhancing the robustness of HSI analysis systems. First, a masked sequence attention learning module is conducted to promote the inherent robustness of HSI analysis systems along spectral channel. Then, we develop a graph convolutional network with learnable graph structure to establish global pixel-wise combinations.In this way, the attack effect would be dispersed by all the related pixels among each combination, and a better defense performance is achievable in spatial aspect.Finally, to improve the defense transferability and address the problem of limited labelled samples, MSSA employs spectra reconstruction as a pretext task and fits the datasets in a self-supervised manner.Comprehensive experiments over three benchmarks verify the effectiveness of MSSA in comparison with the state-of-the-art hyperspectral classification methods and representative adversarial defense strategies.



## **41. CARBEN: Composite Adversarial Robustness Benchmark**

cs.CV

IJCAI 2022 Demo Track; The demonstration is at  https://hsiung.cc/CARBEN/

**SubmitDate**: 2022-07-16    [paper-pdf](http://arxiv.org/pdf/2207.07797v1)

**Authors**: Lei Hsiung, Yun-Yun Tsai, Pin-Yu Chen, Tsung-Yi Ho

**Abstracts**: Prior literature on adversarial attack methods has mainly focused on attacking with and defending against a single threat model, e.g., perturbations bounded in Lp ball. However, multiple threat models can be combined into composite perturbations. One such approach, composite adversarial attack (CAA), not only expands the perturbable space of the image, but also may be overlooked by current modes of robustness evaluation. This paper demonstrates how CAA's attack order affects the resulting image, and provides real-time inferences of different models, which will facilitate users' configuration of the parameters of the attack level and their rapid evaluation of model prediction. A leaderboard to benchmark adversarial robustness against CAA is also introduced.



## **42. Towards the Desirable Decision Boundary by Moderate-Margin Adversarial Training**

cs.CV

**SubmitDate**: 2022-07-16    [paper-pdf](http://arxiv.org/pdf/2207.07793v1)

**Authors**: Xiaoyu Liang, Yaguan Qian, Jianchang Huang, Xiang Ling, Bin Wang, Chunming Wu, Wassim Swaileh

**Abstracts**: Adversarial training, as one of the most effective defense methods against adversarial attacks, tends to learn an inclusive decision boundary to increase the robustness of deep learning models. However, due to the large and unnecessary increase in the margin along adversarial directions, adversarial training causes heavy cross-over between natural examples and adversarial examples, which is not conducive to balancing the trade-off between robustness and natural accuracy. In this paper, we propose a novel adversarial training scheme to achieve a better trade-off between robustness and natural accuracy. It aims to learn a moderate-inclusive decision boundary, which means that the margins of natural examples under the decision boundary are moderate. We call this scheme Moderate-Margin Adversarial Training (MMAT), which generates finer-grained adversarial examples to mitigate the cross-over problem. We also take advantage of logits from a teacher model that has been well-trained to guide the learning of our model. Finally, MMAT achieves high natural accuracy and robustness under both black-box and white-box attacks. On SVHN, for example, state-of-the-art robustness and natural accuracy are achieved.



## **43. Demystifying the Adversarial Robustness of Random Transformation Defenses**

cs.CR

ICML 2022 (short presentation), AAAI 2022 AdvML Workshop (best paper,  oral presentation)

**SubmitDate**: 2022-07-15    [paper-pdf](http://arxiv.org/pdf/2207.03574v2)

**Authors**: Chawin Sitawarin, Zachary Golan-Strieb, David Wagner

**Abstracts**: Neural networks' lack of robustness against attacks raises concerns in security-sensitive settings such as autonomous vehicles. While many countermeasures may look promising, only a few withstand rigorous evaluation. Defenses using random transformations (RT) have shown impressive results, particularly BaRT (Raff et al., 2019) on ImageNet. However, this type of defense has not been rigorously evaluated, leaving its robustness properties poorly understood. Their stochastic properties make evaluation more challenging and render many proposed attacks on deterministic models inapplicable. First, we show that the BPDA attack (Athalye et al., 2018a) used in BaRT's evaluation is ineffective and likely overestimates its robustness. We then attempt to construct the strongest possible RT defense through the informed selection of transformations and Bayesian optimization for tuning their parameters. Furthermore, we create the strongest possible attack to evaluate our RT defense. Our new attack vastly outperforms the baseline, reducing the accuracy by 83% compared to the 19% reduction by the commonly used EoT attack ($4.3\times$ improvement). Our result indicates that the RT defense on the Imagenette dataset (a ten-class subset of ImageNet) is not robust against adversarial examples. Extending the study further, we use our new attack to adversarially train RT defense (called AdvRT), resulting in a large robustness gain. Code is available at https://github.com/wagner-group/demystify-random-transform.



## **44. CC-Fuzz: Genetic algorithm-based fuzzing for stress testing congestion control algorithms**

cs.NI

This version was submitted to Hotnets 2022

**SubmitDate**: 2022-07-15    [paper-pdf](http://arxiv.org/pdf/2207.07300v1)

**Authors**: Devdeep Ray, Srinivasan Seshan

**Abstracts**: Congestion control research has experienced a significant increase in interest in the past few years, with many purpose-built algorithms being designed with the needs of specific applications in mind. These algorithms undergo limited testing before being deployed on the Internet, where they interact with other congestion control algorithms and run across a variety of network conditions. This often results in unforeseen performance issues in the wild due to algorithmic inadequacies or implementation bugs, and these issues are often hard to identify since packet traces are not available.   In this paper, we present CC-Fuzz, an automated congestion control testing framework that uses a genetic search algorithm in order to stress test congestion control algorithms by generating adversarial network traces and traffic patterns. Initial results using this approach are promising - CC-Fuzz automatically found a bug in BBR that causes it to stall permanently, and is able to automatically discover the well-known low-rate TCP attack, among other things.



## **45. PASS: Parameters Audit-based Secure and Fair Federated Learning Scheme against Free Rider**

cs.CR

8 pages, 5 figures, 3 tables

**SubmitDate**: 2022-07-15    [paper-pdf](http://arxiv.org/pdf/2207.07292v1)

**Authors**: Jianhua Wang

**Abstracts**: Federated Learning (FL) as a secure distributed learning frame gains interest in Internet of Things (IoT) due to its capability of protecting private data of participants. However, traditional FL systems are vulnerable to attacks such as Free-Rider (FR) attack, which causes not only unfairness but also privacy leakage and inferior performance to FL systems. The existing defense mechanisms against FR attacks only concern the scenarios where the adversaries declare less than 50% of the total amount of clients. Moreover, they lose effectiveness in resisting selfish FR (SFR) attacks. In this paper, we propose a Parameter Audit-based Secure and fair federated learning Scheme (PASS) against FR attacks. The PASS has the following key features: (a) works well in the scenario where adversaries are more than 50% of the total amount of clients; (b) is effective in countering anonymous FR attacks and SFR attacks; (c) prevents from privacy leakage without accuracy loss. Extensive experimental results verify the data protecting capability in mean square error against privacy leakage and reveal the effectiveness of PASS in terms of a higher defense success rate and lower false positive rate against anonymous SFR attacks. Note in addition, PASS produces no effect on FL accuracy when there is no FR adversary.



## **46. Lipschitz Bound Analysis of Neural Networks**

cs.LG

5 pages, 7 figures

**SubmitDate**: 2022-07-14    [paper-pdf](http://arxiv.org/pdf/2207.07232v1)

**Authors**: Sarosij Bose

**Abstracts**: Lipschitz Bound Estimation is an effective method of regularizing deep neural networks to make them robust against adversarial attacks. This is useful in a variety of applications ranging from reinforcement learning to autonomous systems. In this paper, we highlight the significant gap in obtaining a non-trivial Lipschitz bound certificate for Convolutional Neural Networks (CNNs) and empirically support it with extensive graphical analysis. We also show that unrolling Convolutional layers or Toeplitz matrices can be employed to convert Convolutional Neural Networks (CNNs) to a Fully Connected Network. Further, we propose a simple algorithm to show the existing 20x-50x gap in a particular data distribution between the actual lipschitz constant and the obtained tight bound. We also ran sets of thorough experiments on various network architectures and benchmark them on datasets like MNIST and CIFAR-10. All these proposals are supported by extensive testing, graphs, histograms and comparative analysis.



## **47. Multi-Agent Deep Reinforcement Learning-Driven Mitigation of Adverse Effects of Cyber-Attacks on Electric Vehicle Charging Station**

eess.SY

Submitted to IEEE Transactions on Smart Grids

**SubmitDate**: 2022-07-14    [paper-pdf](http://arxiv.org/pdf/2207.07041v1)

**Authors**: M. Basnet, MH Ali

**Abstracts**: An electric vehicle charging station (EVCS) infrastructure is the backbone of transportation electrification. However, the EVCS has myriads of exploitable vulnerabilities in software, hardware, supply chain, and incumbent legacy technologies such as network, communication, and control. These standalone or networked EVCS open up large attack surfaces for the local or state-funded adversaries. The state-of-the-art approaches are not agile and intelligent enough to defend against and mitigate advanced persistent threats (APT). We propose the data-driven model-free distributed intelligence based on multiagent Deep Reinforcement Learning (MADRL)-- Twin Delayed Deep Deterministic Policy Gradient (TD3) -- that efficiently learns the control policy to mitigate the cyberattacks on the controllers of EVCS. Also, we have proposed two additional mitigation methods: the manual/Bruteforce mitigation and the controller clone-based mitigation. The attack model considers the APT designed to malfunction the duty cycles of the EVCS controllers with Type-I low-frequency attack and Type-II constant attack. The proposed model restores the EVCS operation under threat incidence in any/all controllers by correcting the control signals generated by the legacy controllers. Also, the TD3 algorithm provides higher granularity by learning nonlinear control policies as compared to the other two mitigation methods. Index Terms: Cyberattack, Deep Reinforcement Learning(DRL), Electric Vehicle Charging Station, Mitigation.



## **48. Adversarial Attacks on Monocular Pose Estimation**

cs.CV

Accepted at the 2022 IEEE/RSJ International Conference on Intelligent  Robots and Systems (IROS 2022)

**SubmitDate**: 2022-07-14    [paper-pdf](http://arxiv.org/pdf/2207.07032v1)

**Authors**: Hemang Chawla, Arnav Varma, Elahe Arani, Bahram Zonooz

**Abstracts**: Advances in deep learning have resulted in steady progress in computer vision with improved accuracy on tasks such as object detection and semantic segmentation. Nevertheless, deep neural networks are vulnerable to adversarial attacks, thus presenting a challenge in reliable deployment. Two of the prominent tasks in 3D scene-understanding for robotics and advanced drive assistance systems are monocular depth and pose estimation, often learned together in an unsupervised manner. While studies evaluating the impact of adversarial attacks on monocular depth estimation exist, a systematic demonstration and analysis of adversarial perturbations against pose estimation are lacking. We show how additive imperceptible perturbations can not only change predictions to increase the trajectory drift but also catastrophically alter its geometry. We also study the relation between adversarial perturbations targeting monocular depth and pose estimation networks, as well as the transferability of perturbations to other networks with different architectures and losses. Our experiments show how the generated perturbations lead to notable errors in relative rotation and translation predictions and elucidate vulnerabilities of the networks.



## **49. Susceptibility of Continual Learning Against Adversarial Attacks**

cs.LG

18 pages, 13 figures

**SubmitDate**: 2022-07-14    [paper-pdf](http://arxiv.org/pdf/2207.05225v3)

**Authors**: Hikmat Khan, Pir Masoom Shah, Syed Farhan Alam Zaidi, Saif ul Islam

**Abstracts**: The recent advances in continual (incremental or lifelong) learning have concentrated on the prevention of forgetting that can lead to catastrophic consequences, but there are two outstanding challenges that must be addressed. The first is the evaluation of the robustness of the proposed methods. The second is ensuring the security of learned tasks remains largely unexplored. This paper presents a comprehensive study of the susceptibility of the continually learned tasks (including both current and previously learned tasks) that are vulnerable to forgetting. Such vulnerability of tasks against adversarial attacks raises profound issues in data integrity and privacy. We consider all three scenarios (i.e, task-incremental leaning, domain-incremental learning and class-incremental learning) of continual learning and explore three regularization-based experiments, three replay-based experiments, and one hybrid technique based on the reply and exemplar approach. We examine the robustness of these methods. In particular, we consider cases where we demonstrate that any class belonging to the current or previously learned tasks is prone to misclassification. Our observations, we identify potential limitations in continual learning approaches against adversarial attacks. Our empirical study recommends that the research community consider the robustness of the proposed continual learning approaches and invest extensive efforts in mitigating catastrophic forgetting.



## **50. Adversarial Examples for Model-Based Control: A Sensitivity Analysis**

eess.SY

Submission to the 58th Annual Allerton Conference on Communication,  Control, and Computing

**SubmitDate**: 2022-07-14    [paper-pdf](http://arxiv.org/pdf/2207.06982v1)

**Authors**: Po-han Li, Ufuk Topcu, Sandeep P. Chinchali

**Abstracts**: We propose a method to attack controllers that rely on external timeseries forecasts as task parameters. An adversary can manipulate the costs, states, and actions of the controllers by forging the timeseries, in this case perturbing the real timeseries. Since the controllers often encode safety requirements or energy limits in their costs and constraints, we refer to such manipulation as an adversarial attack. We show that different attacks on model-based controllers can increase control costs, activate constraints, or even make the control optimization problem infeasible. We use the linear quadratic regulator and convex model predictive controllers as examples of how adversarial attacks succeed and demonstrate the impact of adversarial attacks on a battery storage control task for power grid operators. As a result, our method increases control cost by $8500\%$ and energy constraints by $13\%$ on real electricity demand timeseries.



