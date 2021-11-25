# Latest Adversarial Attack Papers
**update at 2021-11-25 23:56:48**

[中文版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. EAD: an ensemble approach to detect adversarial examples from the hidden features of deep neural networks**

cs.CV

**SubmitDate**: 2021-11-24    [paper-pdf](http://arxiv.org/pdf/2111.12631v1)

**Authors**: Francesco Craighero, Fabrizio Angaroni, Fabio Stella, Chiara Damiani, Marco Antoniotti, Alex Graudenzi

**Abstracts**: One of the key challenges in Deep Learning is the definition of effective strategies for the detection of adversarial examples. To this end, we propose a novel approach named Ensemble Adversarial Detector (EAD) for the identification of adversarial examples, in a standard multiclass classification scenario. EAD combines multiple detectors that exploit distinct properties of the input instances in the internal representation of a pre-trained Deep Neural Network (DNN). Specifically, EAD integrates the state-of-the-art detectors based on Mahalanobis distance and on Local Intrinsic Dimensionality (LID) with a newly introduced method based on One-class Support Vector Machines (OSVMs). Although all constituting methods assume that the greater the distance of a test instance from the set of correctly classified training instances, the higher its probability to be an adversarial example, they differ in the way such distance is computed. In order to exploit the effectiveness of the different methods in capturing distinct properties of data distributions and, accordingly, efficiently tackle the trade-off between generalization and overfitting, EAD employs detector-specific distance scores as features of a logistic regression classifier, after independent hyperparameters optimization. We evaluated the EAD approach on distinct datasets (CIFAR-10, CIFAR-100 and SVHN) and models (ResNet and DenseNet) and with regard to four adversarial attacks (FGSM, BIM, DeepFool and CW), also by comparing with competing approaches. Overall, we show that EAD achieves the best AUROC and AUPR in the large majority of the settings and comparable performance in the others. The improvement over the state-of-the-art, and the possibility to easily extend EAD to include any arbitrary set of detectors, pave the way to a widespread adoption of ensemble approaches in the broad field of adversarial example detection.



## **2. Learning to Break Deep Perceptual Hashing: The Use Case NeuralHash**

cs.LG

22 pages, 15 figures, 5 tables

**SubmitDate**: 2021-11-24    [paper-pdf](http://arxiv.org/pdf/2111.06628v2)

**Authors**: Lukas Struppek, Dominik Hintersdorf, Daniel Neider, Kristian Kersting

**Abstracts**: Apple recently revealed its deep perceptual hashing system NeuralHash to detect child sexual abuse material (CSAM) on user devices before files are uploaded to its iCloud service. Public criticism quickly arose regarding the protection of user privacy and the system's reliability. In this paper, we present the first comprehensive empirical analysis of deep perceptual hashing based on NeuralHash. Specifically, we show that current deep perceptual hashing may not be robust. An adversary can manipulate the hash values by applying slight changes in images, either induced by gradient-based approaches or simply by performing standard image transformations, forcing or preventing hash collisions. Such attacks permit malicious actors easily to exploit the detection system: from hiding abusive material to framing innocent users, everything is possible. Moreover, using the hash values, inferences can still be made about the data stored on user devices. In our view, based on our results, deep perceptual hashing in its current form is generally not ready for robust client-side scanning and should not be used from a privacy perspective.



## **3. REGroup: Rank-aggregating Ensemble of Generative Classifiers for Robust Predictions**

cs.CV

WACV,2022. Project Page : https://lokender.github.io/REGroup.html

**SubmitDate**: 2021-11-24    [paper-pdf](http://arxiv.org/pdf/2006.10679v2)

**Authors**: Lokender Tiwari, Anish Madan, Saket Anand, Subhashis Banerjee

**Abstracts**: Deep Neural Networks (DNNs) are often criticized for being susceptible to adversarial attacks. Most successful defense strategies adopt adversarial training or random input transformations that typically require retraining or fine-tuning the model to achieve reasonable performance. In this work, our investigations of intermediate representations of a pre-trained DNN lead to an interesting discovery pointing to intrinsic robustness to adversarial attacks. We find that we can learn a generative classifier by statistically characterizing the neural response of an intermediate layer to clean training samples. The predictions of multiple such intermediate-layer based classifiers, when aggregated, show unexpected robustness to adversarial attacks. Specifically, we devise an ensemble of these generative classifiers that rank-aggregates their predictions via a Borda count-based consensus. Our proposed approach uses a subset of the clean training data and a pre-trained model, and yet is agnostic to network architectures or the adversarial attack generation method. We show extensive experiments to establish that our defense strategy achieves state-of-the-art performance on the ImageNet validation set.



## **4. Thundernna: a white box adversarial attack**

cs.LG

10 pages, 5 figures

**SubmitDate**: 2021-11-24    [paper-pdf](http://arxiv.org/pdf/2111.12305v1)

**Authors**: Linfeng Ye

**Abstracts**: The existing work shows that the neural network trained by naive gradient-based optimization method is prone to adversarial attacks, adds small malicious on the ordinary input is enough to make the neural network wrong. At the same time, the attack against a neural network is the key to improving its robustness. The training against adversarial examples can make neural networks resist some kinds of adversarial attacks. At the same time, the adversarial attack against a neural network can also reveal some characteristics of the neural network, a complex high-dimensional non-linear function, as discussed in previous work.   In This project, we develop a first-order method to attack the neural network. Compare with other first-order attacks, our method has a much higher success rate. Furthermore, it is much faster than second-order attacks and multi-steps first-order attacks.



## **5. Subspace Adversarial Training**

cs.LG

**SubmitDate**: 2021-11-24    [paper-pdf](http://arxiv.org/pdf/2111.12229v1)

**Authors**: Tao Li, Yingwen Wu, Sizhe Chen, Kun Fang, Xiaolin Huang

**Abstracts**: Single-step adversarial training (AT) has received wide attention as it proved to be both efficient and robust. However, a serious problem of catastrophic overfitting exists, i.e., the robust accuracy against projected gradient descent (PGD) attack suddenly drops to $0\%$ during the training. In this paper, we understand this problem from a novel perspective of optimization and firstly reveal the close link between the fast-growing gradient of each sample and overfitting, which can also be applied to understand the robust overfitting phenomenon in multi-step AT. To control the growth of the gradient during the training, we propose a new AT method, subspace adversarial training (Sub-AT), which constrains the AT in a carefully extracted subspace. It successfully resolves both two kinds of overfitting and hence significantly boosts the robustness. In subspace, we also allow single-step AT with larger steps and larger radius, which further improves the robustness performance. As a result, we achieve the state-of-the-art single-step AT performance: our pure single-step AT can reach over $\mathbf{51}\%$ robust accuracy against strong PGD-50 attack with radius $8/255$ on CIFAR-10, even surpassing the standard multi-step PGD-10 AT with huge computational advantages. The code is released$\footnote{\url{https://github.com/nblt/Sub-AT}}$.



## **6. Fixed Points in Cyber Space: Rethinking Optimal Evasion Attacks in the Age of AI-NIDS**

cs.CR

**SubmitDate**: 2021-11-23    [paper-pdf](http://arxiv.org/pdf/2111.12197v1)

**Authors**: Christian Schroeder de Witt, Yongchao Huang, Philip H. S. Torr, Martin Strohmeier

**Abstracts**: Cyber attacks are increasing in volume, frequency, and complexity. In response, the security community is looking toward fully automating cyber defense systems using machine learning. However, so far the resultant effects on the coevolutionary dynamics of attackers and defenders have not been examined. In this whitepaper, we hypothesise that increased automation on both sides will accelerate the coevolutionary cycle, thus begging the question of whether there are any resultant fixed points, and how they are characterised. Working within the threat model of Locked Shields, Europe's largest cyberdefense exercise, we study blackbox adversarial attacks on network classifiers. Given already existing attack capabilities, we question the utility of optimal evasion attack frameworks based on minimal evasion distances. Instead, we suggest a novel reinforcement learning setting that can be used to efficiently generate arbitrary adversarial perturbations. We then argue that attacker-defender fixed points are themselves general-sum games with complex phase transitions, and introduce a temporally extended multi-agent reinforcement learning framework in which the resultant dynamics can be studied. We hypothesise that one plausible fixed point of AI-NIDS may be a scenario where the defense strategy relies heavily on whitelisted feature flow subspaces. Finally, we demonstrate that a continual learning approach is required to study attacker-defender dynamics in temporally extended general-sum games.



## **7. Watermarking Graph Neural Networks based on Backdoor Attacks**

cs.LG

**SubmitDate**: 2021-11-23    [paper-pdf](http://arxiv.org/pdf/2110.11024v2)

**Authors**: Jing Xu, Stjepan Picek

**Abstracts**: Graph Neural Networks (GNNs) have achieved promising performance in various real-world applications. Building a powerful GNN model is not a trivial task, as it requires a large amount of training data, powerful computing resources, and human expertise on fine-tuning the model. What is more, with the development of adversarial attacks, e.g., model stealing attacks, GNNs raise challenges to model authentication. To avoid copyright infringement on GNNs, it is necessary to verify the ownership of the GNN models.   In this paper, we present a watermarking framework for GNNs for both graph and node classification tasks. We 1) design two strategies to generate watermarked data for the graph classification and one for the node classification task, 2) embed the watermark into the host model through training to obtain the watermarked GNN model, and 3) verify the ownership of the suspicious model in a black-box setting. The experiments show that our framework can verify the ownership of GNN models with a very high probability (around $100\%$) for both tasks. In addition, we experimentally show that our watermarking approach is still effective even when considering suspicious models obtained from different architectures than the owner's.



## **8. Adversarial machine learning for protecting against online manipulation**

cs.LG

To appear on IEEE Internet Computing. `Accepted manuscript' version

**SubmitDate**: 2021-11-23    [paper-pdf](http://arxiv.org/pdf/2111.12034v1)

**Authors**: Stefano Cresci, Marinella Petrocchi, Angelo Spognardi, Stefano Tognazzi

**Abstracts**: Adversarial examples are inputs to a machine learning system that result in an incorrect output from that system. Attacks launched through this type of input can cause severe consequences: for example, in the field of image recognition, a stop signal can be misclassified as a speed limit indication.However, adversarial examples also represent the fuel for a flurry of research directions in different domains and applications. Here, we give an overview of how they can be profitably exploited as powerful tools to build stronger learning models, capable of better-withstanding attacks, for two crucial tasks: fake news and social bot detection.



## **9. An Improved Genetic Algorithm and Its Application in Neural Network Adversarial Attack**

cs.NE

14 pages, 7 figures, 4 tables and 20 References

**SubmitDate**: 2021-11-23    [paper-pdf](http://arxiv.org/pdf/2110.01818v3)

**Authors**: Dingming Yang, Zeyu Yu, Hongqiang Yuan, Yanrong Cui

**Abstracts**: The choice of crossover and mutation strategies plays a crucial role in the search ability, convergence efficiency and precision of genetic algorithms. In this paper, a new improved genetic algorithm is proposed by improving the crossover and mutation operation of the simple genetic algorithm, and it is verified by four test functions. Simulation results show that, comparing with three other mainstream swarm intelligence optimization algorithms, the algorithm can not only improve the global search ability, convergence efficiency and precision, but also increase the success rate of convergence to the optimal value under the same experimental conditions. Finally, the algorithm is applied to neural networks adversarial attacks. The applied results show that the method does not need the structure and parameter information inside the neural network model, and it can obtain the adversarial samples with high confidence in a brief time just by the classification and confidence information output from the neural network.



## **10. Relevance Attack on Detectors**

cs.CV

accepted by Pattern Recognition

**SubmitDate**: 2021-11-23    [paper-pdf](http://arxiv.org/pdf/2008.06822v4)

**Authors**: Sizhe Chen, Fan He, Xiaolin Huang, Kun Zhang

**Abstracts**: This paper focuses on high-transferable adversarial attacks on detectors, which are hard to attack in a black-box manner, because of their multiple-output characteristics and the diversity across architectures. To pursue a high attack transferability, one plausible way is to find a common property across detectors, which facilitates the discovery of common weaknesses. We are the first to suggest that the relevance map from interpreters for detectors is such a property. Based on it, we design a Relevance Attack on Detectors (RAD), which achieves a state-of-the-art transferability, exceeding existing results by above 20%. On MS COCO, the detection mAPs for all 8 black-box architectures are more than halved and the segmentation mAPs are also significantly influenced. Given the great transferability of RAD, we generate the first adversarial dataset for object detection and instance segmentation, i.e., Adversarial Objects in COntext (AOCO), which helps to quickly evaluate and improve the robustness of detectors.



## **11. A Comparison of State-of-the-Art Techniques for Generating Adversarial Malware Binaries**

cs.CR

18 pages, 7 figures; summer project report from NREIP internship at  Naval Research Laboratory

**SubmitDate**: 2021-11-22    [paper-pdf](http://arxiv.org/pdf/2111.11487v1)

**Authors**: Prithviraj Dasgupta, Zachariah Osman

**Abstracts**: We consider the problem of generating adversarial malware by a cyber-attacker where the attacker's task is to strategically modify certain bytes within existing binary malware files, so that the modified files are able to evade a malware detector such as machine learning-based malware classifier. We have evaluated three recent adversarial malware generation techniques using binary malware samples drawn from a single, publicly available malware data set and compared their performances for evading a machine-learning based malware classifier called MalConv. Our results show that among the compared techniques, the most effective technique is the one that strategically modifies bytes in a binary's header. We conclude by discussing the lessons learned and future research directions on the topic of adversarial malware generation.



## **12. Adversarial Examples on Segmentation Models Can be Easy to Transfer**

cs.CV

**SubmitDate**: 2021-11-22    [paper-pdf](http://arxiv.org/pdf/2111.11368v1)

**Authors**: Jindong Gu, Hengshuang Zhao, Volker Tresp, Philip Torr

**Abstracts**: Deep neural network-based image classification can be misled by adversarial examples with small and quasi-imperceptible perturbations. Furthermore, the adversarial examples created on one classification model can also fool another different model. The transferability of the adversarial examples has recently attracted a growing interest since it makes black-box attacks on classification models feasible. As an extension of classification, semantic segmentation has also received much attention towards its adversarial robustness. However, the transferability of adversarial examples on segmentation models has not been systematically studied. In this work, we intensively study this topic. First, we explore the overfitting phenomenon of adversarial examples on classification and segmentation models. In contrast to the observation made on classification models that the transferability is limited by overfitting to the source model, we find that the adversarial examples on segmentations do not always overfit the source models. Even when no overfitting is presented, the transferability of adversarial examples is limited. We attribute the limitation to the architectural traits of segmentation models, i.e., multi-scale object recognition. Then, we propose a simple and effective method, dubbed dynamic scaling, to overcome the limitation. The high transferability achieved by our method shows that, in contrast to the observations in previous work, adversarial examples on a segmentation model can be easy to transfer to other segmentation models. Our analysis and proposals are supported by extensive experiments.



## **13. Shift Invariance Can Reduce Adversarial Robustness**

cs.LG

Published as a conference paper at NeurIPS 2021

**SubmitDate**: 2021-11-22    [paper-pdf](http://arxiv.org/pdf/2103.02695v3)

**Authors**: Songwei Ge, Vasu Singla, Ronen Basri, David Jacobs

**Abstracts**: Shift invariance is a critical property of CNNs that improves performance on classification. However, we show that invariance to circular shifts can also lead to greater sensitivity to adversarial attacks. We first characterize the margin between classes when a shift-invariant linear classifier is used. We show that the margin can only depend on the DC component of the signals. Then, using results about infinitely wide networks, we show that in some simple cases, fully connected and shift-invariant neural networks produce linear decision boundaries. Using this, we prove that shift invariance in neural networks produces adversarial examples for the simple case of two classes, each consisting of a single image with a black or white dot on a gray background. This is more than a curiosity; we show empirically that with real datasets and realistic architectures, shift invariance reduces adversarial robustness. Finally, we describe initial experiments using synthetic data to probe the source of this connection.



## **14. NTD: Non-Transferability Enabled Backdoor Detection**

cs.CR

**SubmitDate**: 2021-11-22    [paper-pdf](http://arxiv.org/pdf/2111.11157v1)

**Authors**: Yinshan Li, Hua Ma, Zhi Zhang, Yansong Gao, Alsharif Abuadbba, Anmin Fu, Yifeng Zheng, Said F. Al-Sarawi, Derek Abbott

**Abstracts**: A backdoor deep learning (DL) model behaves normally upon clean inputs but misbehaves upon trigger inputs as the backdoor attacker desires, posing severe consequences to DL model deployments. State-of-the-art defenses are either limited to specific backdoor attacks (source-agnostic attacks) or non-user-friendly in that machine learning (ML) expertise or expensive computing resources are required. This work observes that all existing backdoor attacks have an inevitable intrinsic weakness, non-transferability, that is, a trigger input hijacks a backdoored model but cannot be effective to another model that has not been implanted with the same backdoor. With this key observation, we propose non-transferability enabled backdoor detection (NTD) to identify trigger inputs for a model-under-test (MUT) during run-time.Specifically, NTD allows a potentially backdoored MUT to predict a class for an input. In the meantime, NTD leverages a feature extractor (FE) to extract feature vectors for the input and a group of samples randomly picked from its predicted class, and then compares similarity between the input and the samples in the FE's latent space. If the similarity is low, the input is an adversarial trigger input; otherwise, benign. The FE is a free pre-trained model privately reserved from open platforms. As the FE and MUT are from different sources, the attacker is very unlikely to insert the same backdoor into both of them. Because of non-transferability, a trigger effect that does work on the MUT cannot be transferred to the FE, making NTD effective against different types of backdoor attacks. We evaluate NTD on three popular customized tasks such as face recognition, traffic sign recognition and general animal classification, results of which affirm that NDT has high effectiveness (low false acceptance rate) and usability (low false rejection rate) with low detection latency.



## **15. Efficient Combinatorial Optimization for Word-level Adversarial Textual Attack**

cs.CL

**SubmitDate**: 2021-11-22    [paper-pdf](http://arxiv.org/pdf/2109.02229v3)

**Authors**: Shengcai Liu, Ning Lu, Cheng Chen, Ke Tang

**Abstracts**: Over the past few years, various word-level textual attack approaches have been proposed to reveal the vulnerability of deep neural networks used in natural language processing. Typically, these approaches involve an important optimization step to determine which substitute to be used for each word in the original input. However, current research on this step is still rather limited, from the perspectives of both problem-understanding and problem-solving. In this paper, we address these issues by uncovering the theoretical properties of the problem and proposing an efficient local search algorithm (LS) to solve it. We establish the first provable approximation guarantee on solving the problem in general cases.Extensive experiments involving 5 NLP tasks, 8 datasets and 26 NLP models show that LS can largely reduce the number of queries usually by an order of magnitude to achieve high attack success rates. Further experiments show that the adversarial examples crafted by LS usually have higher quality, exhibit better transferability, and can bring more robustness improvement to victim models by adversarial training.



## **16. Myope Models -- Are face presentation attack detection models short-sighted?**

cs.CV

Accepted at the 2ND WORKSHOP ON EXPLAINABLE & INTERPRETABLE  ARTIFICIAL INTELLIGENCE FOR BIOMETRICS AT WACV 2022

**SubmitDate**: 2021-11-22    [paper-pdf](http://arxiv.org/pdf/2111.11127v1)

**Authors**: Pedro C. Neto, Ana F. Sequeira, Jaime S. Cardoso

**Abstracts**: Presentation attacks are recurrent threats to biometric systems, where impostors attempt to bypass these systems. Humans often use background information as contextual cues for their visual system. Yet, regarding face-based systems, the background is often discarded, since face presentation attack detection (PAD) models are mostly trained with face crops. This work presents a comparative study of face PAD models (including multi-task learning, adversarial training and dynamic frame selection) in two settings: with and without crops. The results show that the performance is consistently better when the background is present in the images. The proposed multi-task methodology beats the state-of-the-art results on the ROSE-Youtu dataset by a large margin with an equal error rate of 0.2%. Furthermore, we analyze the models' predictions with Grad-CAM++ with the aim to investigate to what extent the models focus on background elements that are known to be useful for human inspection. From this analysis we can conclude that the background cues are not relevant across all the attacks. Thus, showing the capability of the model to leverage the background information only when necessary.



## **17. Meta-Learning the Search Distribution of Black-Box Random Search Based Adversarial Attacks**

cs.LG

accepted at NeurIPS 2021; updated the numbers in Table 5 and added  references; added acknowledgements

**SubmitDate**: 2021-11-22    [paper-pdf](http://arxiv.org/pdf/2111.01714v3)

**Authors**: Maksym Yatsura, Jan Hendrik Metzen, Matthias Hein

**Abstracts**: Adversarial attacks based on randomized search schemes have obtained state-of-the-art results in black-box robustness evaluation recently. However, as we demonstrate in this work, their efficiency in different query budget regimes depends on manual design and heuristic tuning of the underlying proposal distributions. We study how this issue can be addressed by adapting the proposal distribution online based on the information obtained during the attack. We consider Square Attack, which is a state-of-the-art score-based black-box attack, and demonstrate how its performance can be improved by a learned controller that adjusts the parameters of the proposal distribution online during the attack. We train the controller using gradient-based end-to-end training on a CIFAR10 model with white box access. We demonstrate that plugging the learned controller into the attack consistently improves its black-box robustness estimate in different query regimes by up to 20% for a wide range of different models with black-box access. We further show that the learned adaptation principle transfers well to the other data distributions such as CIFAR100 or ImageNet and to the targeted attack setting.



## **18. Evaluating Adversarial Attacks on ImageNet: A Reality Check on Misclassification Classes**

cs.CV

Accepted for publication in 35th Conference on Neural Information  Processing Systems (NeurIPS 2021), Workshop on ImageNet: Past,Present, and  Future

**SubmitDate**: 2021-11-22    [paper-pdf](http://arxiv.org/pdf/2111.11056v1)

**Authors**: Utku Ozbulak, Maura Pintor, Arnout Van Messem, Wesley De Neve

**Abstracts**: Although ImageNet was initially proposed as a dataset for performance benchmarking in the domain of computer vision, it also enabled a variety of other research efforts. Adversarial machine learning is one such research effort, employing deceptive inputs to fool models in making wrong predictions. To evaluate attacks and defenses in the field of adversarial machine learning, ImageNet remains one of the most frequently used datasets. However, a topic that is yet to be investigated is the nature of the classes into which adversarial examples are misclassified. In this paper, we perform a detailed analysis of these misclassification classes, leveraging the ImageNet class hierarchy and measuring the relative positions of the aforementioned type of classes in the unperturbed origins of the adversarial examples. We find that $71\%$ of the adversarial examples that achieve model-to-model adversarial transferability are misclassified into one of the top-5 classes predicted for the underlying source images. We also find that a large subset of untargeted misclassifications are, in fact, misclassifications into semantically similar classes. Based on these findings, we discuss the need to take into account the ImageNet class hierarchy when evaluating untargeted adversarial successes. Furthermore, we advocate for future research efforts to incorporate categorical information.



## **19. Selection of Source Images Heavily Influences the Effectiveness of Adversarial Attacks**

cs.CV

Accepted for publication in the 32nd British Machine Vision  Conference (BMVC)

**SubmitDate**: 2021-11-22    [paper-pdf](http://arxiv.org/pdf/2106.07141v3)

**Authors**: Utku Ozbulak, Esla Timothy Anzaku, Wesley De Neve, Arnout Van Messem

**Abstracts**: Although the adoption rate of deep neural networks (DNNs) has tremendously increased in recent years, a solution for their vulnerability against adversarial examples has not yet been found. As a result, substantial research efforts are dedicated to fix this weakness, with many studies typically using a subset of source images to generate adversarial examples, treating every image in this subset as equal. We demonstrate that, in fact, not every source image is equally suited for this kind of assessment. To do so, we devise a large-scale model-to-model transferability scenario for which we meticulously analyze the properties of adversarial examples, generated from every suitable source image in ImageNet by making use of three of the most frequently deployed attacks. In this transferability scenario, which involves seven distinct DNN models, including the recently proposed vision transformers, we reveal that it is possible to have a difference of up to $12.5\%$ in model-to-model transferability success, $1.01$ in average $L_2$ perturbation, and $0.03$ ($8/225$) in average $L_{\infty}$ perturbation when $1,000$ source images are sampled randomly among all suitable candidates. We then take one of the first steps in evaluating the robustness of images used to create adversarial examples, proposing a number of simple but effective methods to identify unsuitable source images, thus making it possible to mitigate extreme cases in experimentation and support high-quality benchmarking.



## **20. Imperceptible Transfer Attack and Defense on 3D Point Cloud Classification**

cs.CV

**SubmitDate**: 2021-11-22    [paper-pdf](http://arxiv.org/pdf/2111.10990v1)

**Authors**: Daizong Liu, Wei Hu

**Abstracts**: Although many efforts have been made into attack and defense on the 2D image domain in recent years, few methods explore the vulnerability of 3D models. Existing 3D attackers generally perform point-wise perturbation over point clouds, resulting in deformed structures or outliers, which is easily perceivable by humans. Moreover, their adversarial examples are generated under the white-box setting, which frequently suffers from low success rates when transferred to attack remote black-box models. In this paper, we study 3D point cloud attacks from two new and challenging perspectives by proposing a novel Imperceptible Transfer Attack (ITA): 1) Imperceptibility: we constrain the perturbation direction of each point along its normal vector of the neighborhood surface, leading to generated examples with similar geometric properties and thus enhancing the imperceptibility. 2) Transferability: we develop an adversarial transformation model to generate the most harmful distortions and enforce the adversarial examples to resist it, improving their transferability to unknown black-box models. Further, we propose to train more robust black-box 3D models to defend against such ITA attacks by learning more discriminative point cloud representations. Extensive evaluations demonstrate that our ITA attack is more imperceptible and transferable than state-of-the-arts and validate the superiority of our defense strategy.



## **21. Medical Aegis: Robust adversarial protectors for medical images**

cs.CV

**SubmitDate**: 2021-11-22    [paper-pdf](http://arxiv.org/pdf/2111.10969v1)

**Authors**: Qingsong Yao, Zecheng He, S. Kevin Zhou

**Abstracts**: Deep neural network based medical image systems are vulnerable to adversarial examples. Many defense mechanisms have been proposed in the literature, however, the existing defenses assume a passive attacker who knows little about the defense system and does not change the attack strategy according to the defense. Recent works have shown that a strong adaptive attack, where an attacker is assumed to have full knowledge about the defense system, can easily bypass the existing defenses. In this paper, we propose a novel adversarial example defense system called Medical Aegis. To the best of our knowledge, Medical Aegis is the first defense in the literature that successfully addresses the strong adaptive adversarial example attacks to medical images. Medical Aegis boasts two-tier protectors: The first tier of Cushion weakens the adversarial manipulation capability of an attack by removing its high-frequency components, yet posing a minimal effect on classification performance of the original image; the second tier of Shield learns a set of per-class DNN models to predict the logits of the protected model. Deviation from the Shield's prediction indicates adversarial examples. Shield is inspired by the observations in our stress tests that there exist robust trails in the shallow layers of a DNN model, which the adaptive attacks can hardly destruct. Experimental results show that the proposed defense accurately detects adaptive attacks, with negligible overhead for model inference.



## **22. Local Model Poisoning Attacks to Byzantine-Robust Federated Learning**

cs.CR

Appeared in Usenix Security Symposium 2020. Fixed an error in Theorem  1. For demo code, see https://people.duke.edu/~zg70/code/fltrust.zip . For  slides, see https://people.duke.edu/~zg70/code/Secure_Federated_Learning.pdf  . For the talk, see https://www.youtube.com/watch?v=LP4uqW18yA0

**SubmitDate**: 2021-11-21    [paper-pdf](http://arxiv.org/pdf/1911.11815v4)

**Authors**: Minghong Fang, Xiaoyu Cao, Jinyuan Jia, Neil Zhenqiang Gong

**Abstracts**: In federated learning, multiple client devices jointly learn a machine learning model: each client device maintains a local model for its local training dataset, while a master device maintains a global model via aggregating the local models from the client devices. The machine learning community recently proposed several federated learning methods that were claimed to be robust against Byzantine failures (e.g., system failures, adversarial manipulations) of certain client devices. In this work, we perform the first systematic study on local model poisoning attacks to federated learning. We assume an attacker has compromised some client devices, and the attacker manipulates the local model parameters on the compromised client devices during the learning process such that the global model has a large testing error rate. We formulate our attacks as optimization problems and apply our attacks to four recent Byzantine-robust federated learning methods. Our empirical results on four real-world datasets show that our attacks can substantially increase the error rates of the models learnt by the federated learning methods that were claimed to be robust against Byzantine failures of some client devices. We generalize two defenses for data poisoning attacks to defend against our local model poisoning attacks. Our evaluation results show that one defense can effectively defend against our attacks in some cases, but the defenses are not effective enough in other cases, highlighting the need for new defenses against our local model poisoning attacks to federated learning.



## **23. Denoised Internal Models: a Brain-Inspired Autoencoder against Adversarial Attacks**

cs.CV

16 pages, 3 figures

**SubmitDate**: 2021-11-21    [paper-pdf](http://arxiv.org/pdf/2111.10844v1)

**Authors**: Kaiyuan Liu, Xingyu Li, Yi Zhou, Jisong Guan, Yurui Lai, Ge Zhang, Hang Su, Jiachen Wang, Chunxu Guo

**Abstracts**: Despite its great success, deep learning severely suffers from robustness; that is, deep neural networks are very vulnerable to adversarial attacks, even the simplest ones. Inspired by recent advances in brain science, we propose the Denoised Internal Models (DIM), a novel generative autoencoder-based model to tackle this challenge. Simulating the pipeline in the human brain for visual signal processing, DIM adopts a two-stage approach. In the first stage, DIM uses a denoiser to reduce the noise and the dimensions of inputs, reflecting the information pre-processing in the thalamus. Inspired from the sparse coding of memory-related traces in the primary visual cortex, the second stage produces a set of internal models, one for each category. We evaluate DIM over 42 adversarial attacks, showing that DIM effectively defenses against all the attacks and outperforms the SOTA on the overall robustness.



## **24. Modelling Direct Messaging Networks with Multiple Recipients for Cyber Deception**

cs.CR

**SubmitDate**: 2021-11-21    [paper-pdf](http://arxiv.org/pdf/2111.11932v1)

**Authors**: Kristen Moore, Cody J. Christopher, David Liebowitz, Surya Nepal, Renee Selvey

**Abstracts**: Cyber deception is emerging as a promising approach to defending networks and systems against attackers and data thieves. However, despite being relatively cheap to deploy, the generation of realistic content at scale is very costly, due to the fact that rich, interactive deceptive technologies are largely hand-crafted. With recent improvements in Machine Learning, we now have the opportunity to bring scale and automation to the creation of realistic and enticing simulated content. In this work, we propose a framework to automate the generation of email and instant messaging-style group communications at scale. Such messaging platforms within organisations contain a lot of valuable information inside private communications and document attachments, making them an enticing target for an adversary. We address two key aspects of simulating this type of system: modelling when and with whom participants communicate, and generating topical, multi-party text to populate simulated conversation threads. We present the LogNormMix-Net Temporal Point Process as an approach to the first of these, building upon the intensity-free modeling approach of Shchur et al.~\cite{shchur2019intensity} to create a generative model for unicast and multi-cast communications. We demonstrate the use of fine-tuned, pre-trained language models to generate convincing multi-party conversation threads. A live email server is simulated by uniting our LogNormMix-Net TPP (to generate the communication timestamp, sender and recipients) with the language model, which generates the contents of the multi-party email threads. We evaluate the generated content with respect to a number of realism-based properties, that encourage a model to learn to generate content that will engage the attention of an adversary to achieve a deception outcome.



## **25. Inconspicuous Adversarial Patches for Fooling Image Recognition Systems on Mobile Devices**

cs.CV

accpeted by iotj. arXiv admin note: substantial text overlap with  arXiv:2009.09774

**SubmitDate**: 2021-11-21    [paper-pdf](http://arxiv.org/pdf/2106.15202v2)

**Authors**: Tao Bai, Jinqi Luo, Jun Zhao

**Abstracts**: Deep learning based image recognition systems have been widely deployed on mobile devices in today's world. In recent studies, however, deep learning models are shown vulnerable to adversarial examples. One variant of adversarial examples, called adversarial patch, draws researchers' attention due to its strong attack abilities. Though adversarial patches achieve high attack success rates, they are easily being detected because of the visual inconsistency between the patches and the original images. Besides, it usually requires a large amount of data for adversarial patch generation in the literature, which is computationally expensive and time-consuming. To tackle these challenges, we propose an approach to generate inconspicuous adversarial patches with one single image. In our approach, we first decide the patch locations basing on the perceptual sensitivity of victim models, then produce adversarial patches in a coarse-to-fine way by utilizing multiple-scale generators and discriminators. The patches are encouraged to be consistent with the background images with adversarial training while preserving strong attack abilities. Our approach shows the strong attack abilities in white-box settings and the excellent transferability in black-box settings through extensive experiments on various models with different architectures and training methods. Compared to other adversarial patches, our adversarial patches hold the most negligible risks to be detected and can evade human observations, which is supported by the illustrations of saliency maps and results of user evaluations. Lastly, we show that our adversarial patches can be applied in the physical world.



## **26. Adversarial Mask: Real-World Adversarial Attack Against Face Recognition Models**

cs.CV

**SubmitDate**: 2021-11-21    [paper-pdf](http://arxiv.org/pdf/2111.10759v1)

**Authors**: Alon Zolfi, Shai Avidan, Yuval Elovici, Asaf Shabtai

**Abstracts**: Deep learning-based facial recognition (FR) models have demonstrated state-of-the-art performance in the past few years, even when wearing protective medical face masks became commonplace during the COVID-19 pandemic. Given the outstanding performance of these models, the machine learning research community has shown increasing interest in challenging their robustness. Initially, researchers presented adversarial attacks in the digital domain, and later the attacks were transferred to the physical domain. However, in many cases, attacks in the physical domain are conspicuous, requiring, for example, the placement of a sticker on the face, and thus may raise suspicion in real-world environments (e.g., airports). In this paper, we propose Adversarial Mask, a physical adversarial universal perturbation (UAP) against state-of-the-art FR models that is applied on face masks in the form of a carefully crafted pattern. In our experiments, we examined the transferability of our adversarial mask to a wide range of FR model architectures and datasets. In addition, we validated our adversarial mask effectiveness in real-world experiments by printing the adversarial pattern on a fabric medical face mask, causing the FR system to identify only 3.34% of the participants wearing the mask (compared to a minimum of 83.34% with other evaluated masks).



## **27. Stochastic Variance Reduced Ensemble Adversarial Attack for Boosting the Adversarial Transferability**

cs.LG

10 pages, 5 figures, submitted to a conference for review

**SubmitDate**: 2021-11-21    [paper-pdf](http://arxiv.org/pdf/2111.10752v1)

**Authors**: Yifeng Xiong, Jiadong Lin, Min Zhang, John E. Hopcroft, Kun He

**Abstracts**: The black-box adversarial attack has attracted impressive attention for its practical use in the field of deep learning security, meanwhile, it is very challenging as there is no access to the network architecture or internal weights of the target model. Based on the hypothesis that if an example remains adversarial for multiple models, then it is more likely to transfer the attack capability to other models, the ensemble-based adversarial attack methods are efficient and widely used for black-box attacks. However, ways of ensemble attack are rather less investigated, and existing ensemble attacks simply fuse the outputs of all the models evenly. In this work, we treat the iterative ensemble attack as a stochastic gradient descent optimization process, in which the variance of the gradients on different models may lead to poor local optima. To this end, we propose a novel attack method called the stochastic variance reduced ensemble (SVRE) attack, which could reduce the gradient variance of the ensemble models and take full advantage of the ensemble attack. Empirical results on the standard ImageNet dataset demonstrate that the proposed method could boost the adversarial transferability and outperforms existing ensemble attacks significantly.



## **28. AEVA: Black-box Backdoor Detection Using Adversarial Extreme Value Analysis**

cs.LG

**SubmitDate**: 2021-11-21    [paper-pdf](http://arxiv.org/pdf/2110.14880v3)

**Authors**: Junfeng Guo, Ang Li, Cong Liu

**Abstracts**: Deep neural networks (DNNs) are proved to be vulnerable against backdoor attacks. A backdoor is often embedded in the target DNNs through injecting a backdoor trigger into training examples, which can cause the target DNNs misclassify an input attached with the backdoor trigger. Existing backdoor detection methods often require the access to the original poisoned training data, the parameters of the target DNNs, or the predictive confidence for each given input, which are impractical in many real-world applications, e.g., on-device deployed DNNs. We address the black-box hard-label backdoor detection problem where the DNN is fully black-box and only its final output label is accessible. We approach this problem from the optimization perspective and show that the objective of backdoor detection is bounded by an adversarial objective. Further theoretical and empirical studies reveal that this adversarial objective leads to a solution with highly skewed distribution; a singularity is often observed in the adversarial map of a backdoor-infected example, which we call the adversarial singularity phenomenon. Based on this observation, we propose the adversarial extreme value analysis(AEVA) to detect backdoors in black-box neural networks. AEVA is based on an extreme value analysis of the adversarial map, computed from the monte-carlo gradient estimation. Evidenced by extensive experiments across multiple popular tasks and backdoor attacks, our approach is shown effective in detecting backdoor attacks under the black-box hard-label scenarios.



## **29. Are Vision Transformers Robust to Patch Perturbations?**

cs.CV

**SubmitDate**: 2021-11-20    [paper-pdf](http://arxiv.org/pdf/2111.10659v1)

**Authors**: Jindong Gu, Volker Tresp, Yao Qin

**Abstracts**: The recent advances in Vision Transformer (ViT) have demonstrated its impressive performance in image classification, which makes it a promising alternative to Convolutional Neural Network (CNN). Unlike CNNs, ViT represents an input image as a sequence of image patches. The patch-wise input image representation makes the following question interesting: How does ViT perform when individual input image patches are perturbed with natural corruptions or adversarial perturbations, compared to CNNs? In this work, we study the robustness of vision transformers to patch-wise perturbations. Surprisingly, we find that vision transformers are more robust to naturally corrupted patches than CNNs, whereas they are more vulnerable to adversarial patches. Furthermore, we conduct extensive qualitative and quantitative experiments to understand the robustness to patch perturbations. We have revealed that ViT's stronger robustness to natural corrupted patches and higher vulnerability against adversarial patches are both caused by the attention mechanism. Specifically, the attention model can help improve the robustness of vision transformers by effectively ignoring natural corrupted patches. However, when vision transformers are attacked by an adversary, the attention mechanism can be easily fooled to focus more on the adversarially perturbed patches and cause a mistake.



## **30. Modeling Design and Control Problems Involving Neural Network Surrogates**

math.OC

24 Pages, 11 Figures

**SubmitDate**: 2021-11-20    [paper-pdf](http://arxiv.org/pdf/2111.10489v1)

**Authors**: Dominic Yang, Prasanna Balaprakash, Sven Leyffer

**Abstracts**: We consider nonlinear optimization problems that involve surrogate models represented by neural networks. We demonstrate first how to directly embed neural network evaluation into optimization models, highlight a difficulty with this approach that can prevent convergence, and then characterize stationarity of such models. We then present two alternative formulations of these problems in the specific case of feedforward neural networks with ReLU activation: as a mixed-integer optimization problem and as a mathematical program with complementarity constraints. For the latter formulation we prove that stationarity at a point for this problem corresponds to stationarity of the embedded formulation. Each of these formulations may be solved with state-of-the-art optimization methods, and we show how to obtain good initial feasible solutions for these methods. We compare our formulations on three practical applications arising in the design and control of combustion engines, in the generation of adversarial attacks on classifier networks, and in the determination of optimal flows in an oil well network.



## **31. Zero-Shot Certified Defense against Adversarial Patches with Vision Transformers**

cs.CV

12 pages, 5 figures

**SubmitDate**: 2021-11-19    [paper-pdf](http://arxiv.org/pdf/2111.10481v1)

**Authors**: Yuheng Huang, Yuanchun Li

**Abstracts**: Adversarial patch attack aims to fool a machine learning model by arbitrarily modifying pixels within a restricted region of an input image. Such attacks are a major threat to models deployed in the physical world, as they can be easily realized by presenting a customized object in the camera view. Defending against such attacks is challenging due to the arbitrariness of patches, and existing provable defenses suffer from poor certified accuracy. In this paper, we propose PatchVeto, a zero-shot certified defense against adversarial patches based on Vision Transformer (ViT) models. Rather than training a robust model to resist adversarial patches which may inevitably sacrifice accuracy, PatchVeto reuses a pretrained ViT model without any additional training, which can achieve high accuracy on clean inputs while detecting adversarial patched inputs by simply manipulating the attention map of ViT. Specifically, each input is tested by voting over multiple inferences with different attention masks, where at least one inference is guaranteed to exclude the adversarial patch. The prediction is certifiably robust if all masked inferences reach consensus, which ensures that any adversarial patch would be detected with no false negative. Extensive experiments have shown that PatchVeto is able to achieve high certified accuracy (e.g. 67.1% on ImageNet for 2%-pixel adversarial patches), significantly outperforming state-of-the-art methods. The clean accuracy is the same as vanilla ViT models (81.8% on ImageNet) since the model parameters are directly reused. Meanwhile, our method can flexibly handle different adversarial patch sizes by simply changing the masking strategy.



## **32. Rethinking Clustering for Robustness**

cs.LG

Accepted to the 32nd British Machine Vision Conference (BMVC'21)

**SubmitDate**: 2021-11-19    [paper-pdf](http://arxiv.org/pdf/2006.07682v3)

**Authors**: Motasem Alfarra, Juan C. Pérez, Adel Bibi, Ali Thabet, Pablo Arbeláez, Bernard Ghanem

**Abstracts**: This paper studies how encouraging semantically-aligned features during deep neural network training can increase network robustness. Recent works observed that Adversarial Training leads to robust models, whose learnt features appear to correlate with human perception. Inspired by this connection from robustness to semantics, we study the complementary connection: from semantics to robustness. To do so, we provide a robustness certificate for distance-based classification models (clustering-based classifiers). Moreover, we show that this certificate is tight, and we leverage it to propose ClusTR (Clustering Training for Robustness), a clustering-based and adversary-free training framework to learn robust models. Interestingly, \textit{ClusTR} outperforms adversarially-trained networks by up to $4\%$ under strong PGD attacks.



## **33. Meta Adversarial Perturbations**

cs.LG

**SubmitDate**: 2021-11-19    [paper-pdf](http://arxiv.org/pdf/2111.10291v1)

**Authors**: Chia-Hung Yuan, Pin-Yu Chen, Chia-Mu Yu

**Abstracts**: A plethora of attack methods have been proposed to generate adversarial examples, among which the iterative methods have been demonstrated the ability to find a strong attack. However, the computation of an adversarial perturbation for a new data point requires solving a time-consuming optimization problem from scratch. To generate a stronger attack, it normally requires updating a data point with more iterations. In this paper, we show the existence of a meta adversarial perturbation (MAP), a better initialization that causes natural images to be misclassified with high probability after being updated through only a one-step gradient ascent update, and propose an algorithm for computing such perturbations. We conduct extensive experiments, and the empirical results demonstrate that state-of-the-art deep neural networks are vulnerable to meta perturbations. We further show that these perturbations are not only image-agnostic, but also model-agnostic, as a single perturbation generalizes well across unseen data points and different neural network architectures.



## **34. Resilience from Diversity: Population-based approach to harden models against adversarial attacks**

cs.LG

10 pages, 6 figures, 5 tables

**SubmitDate**: 2021-11-19    [paper-pdf](http://arxiv.org/pdf/2111.10272v1)

**Authors**: Jasser Jasser, Ivan Garibay

**Abstracts**: Traditional deep learning models exhibit intriguing vulnerabilities that allow an attacker to force them to fail at their task. Notorious attacks such as the Fast Gradient Sign Method (FGSM) and the more powerful Projected Gradient Descent (PGD) generate adversarial examples by adding a magnitude of perturbation $\epsilon$ to the input's computed gradient, resulting in a deterioration of the effectiveness of the model's classification. This work introduces a model that is resilient to adversarial attacks. Our model leverages a well established principle from biological sciences: population diversity produces resilience against environmental changes. More precisely, our model consists of a population of $n$ diverse submodels, each one of them trained to individually obtain a high accuracy for the task at hand, while forced to maintain meaningful differences in their weight tensors. Each time our model receives a classification query, it selects a submodel from its population at random to answer the query. To introduce and maintain diversity in population of submodels, we introduce the concept of counter linking weights. A Counter-Linked Model (CLM) consists of submodels of the same architecture where a periodic random similarity examination is conducted during the simultaneous training to guarantee diversity while maintaining accuracy. In our testing, CLM robustness got enhanced by around 20% when tested on the MNIST dataset and at least 15% when tested on the CIFAR-10 dataset. When implemented with adversarially trained submodels, this methodology achieves state-of-the-art robustness. On the MNIST dataset with $\epsilon=0.3$, it achieved 94.34% against FGSM and 91% against PGD. On the CIFAR-10 dataset with $\epsilon=8/255$, it achieved 62.97% against FGSM and 59.16% against PGD.



## **35. Fast Minimum-norm Adversarial Attacks through Adaptive Norm Constraints**

cs.LG

Accepted at NeurIPS'21

**SubmitDate**: 2021-11-19    [paper-pdf](http://arxiv.org/pdf/2102.12827v3)

**Authors**: Maura Pintor, Fabio Roli, Wieland Brendel, Battista Biggio

**Abstracts**: Evaluating adversarial robustness amounts to finding the minimum perturbation needed to have an input sample misclassified. The inherent complexity of the underlying optimization requires current gradient-based attacks to be carefully tuned, initialized, and possibly executed for many computationally-demanding iterations, even if specialized to a given perturbation model. In this work, we overcome these limitations by proposing a fast minimum-norm (FMN) attack that works with different $\ell_p$-norm perturbation models ($p=0, 1, 2, \infty$), is robust to hyperparameter choices, does not require adversarial starting points, and converges within few lightweight steps. It works by iteratively finding the sample misclassified with maximum confidence within an $\ell_p$-norm constraint of size $\epsilon$, while adapting $\epsilon$ to minimize the distance of the current sample to the decision boundary. Extensive experiments show that FMN significantly outperforms existing attacks in terms of convergence speed and computation time, while reporting comparable or even smaller perturbation sizes.



## **36. Federated Learning for Malware Detection in IoT Devices**

cs.CR

**SubmitDate**: 2021-11-19    [paper-pdf](http://arxiv.org/pdf/2104.09994v3)

**Authors**: Valerian Rey, Pedro Miguel Sánchez Sánchez, Alberto Huertas Celdrán, Gérôme Bovet, Martin Jaggi

**Abstracts**: This work investigates the possibilities enabled by federated learning concerning IoT malware detection and studies security issues inherent to this new learning paradigm. In this context, a framework that uses federated learning to detect malware affecting IoT devices is presented. N-BaIoT, a dataset modeling network traffic of several real IoT devices while affected by malware, has been used to evaluate the proposed framework. Both supervised and unsupervised federated models (multi-layer perceptron and autoencoder) able to detect malware affecting seen and unseen IoT devices of N-BaIoT have been trained and evaluated. Furthermore, their performance has been compared to two traditional approaches. The first one lets each participant locally train a model using only its own data, while the second consists of making the participants share their data with a central entity in charge of training a global model. This comparison has shown that the use of more diverse and large data, as done in the federated and centralized methods, has a considerable positive impact on the model performance. Besides, the federated models, while preserving the participant's privacy, show similar results as the centralized ones. As an additional contribution and to measure the robustness of the federated approach, an adversarial setup with several malicious participants poisoning the federated model has been considered. The baseline model aggregation averaging step used in most federated learning algorithms appears highly vulnerable to different attacks, even with a single adversary. The performance of other model aggregation functions acting as countermeasures is thus evaluated under the same attack scenarios. These functions provide a significant improvement against malicious participants, but more efforts are still needed to make federated approaches robust.



## **37. Fooling Adversarial Training with Inducing Noise**

cs.LG

**SubmitDate**: 2021-11-19    [paper-pdf](http://arxiv.org/pdf/2111.10130v1)

**Authors**: Zhirui Wang, Yifei Wang, Yisen Wang

**Abstracts**: Adversarial training is widely believed to be a reliable approach to improve model robustness against adversarial attack. However, in this paper, we show that when trained on one type of poisoned data, adversarial training can also be fooled to have catastrophic behavior, e.g., $<1\%$ robust test accuracy with $>90\%$ robust training accuracy on CIFAR-10 dataset. Previously, there are other types of noise poisoned in the training data that have successfully fooled standard training ($15.8\%$ standard test accuracy with $99.9\%$ standard training accuracy on CIFAR-10 dataset), but their poisonings can be easily removed when adopting adversarial training. Therefore, we aim to design a new type of inducing noise, named ADVIN, which is an irremovable poisoning of training data. ADVIN can not only degrade the robustness of adversarial training by a large margin, for example, from $51.7\%$ to $0.57\%$ on CIFAR-10 dataset, but also be effective for fooling standard training ($13.1\%$ standard test accuracy with $100\%$ standard training accuracy). Additionally, ADVIN can be applied to preventing personal data (like selfies) from being exploited without authorization under whether standard or adversarial training.



## **38. Exposing Weaknesses of Malware Detectors with Explainability-Guided Evasion Attacks**

cs.CR

**SubmitDate**: 2021-11-19    [paper-pdf](http://arxiv.org/pdf/2111.10085v1)

**Authors**: Wei Wang, Ruoxi Sun, Tian Dong, Shaofeng Li, Minhui Xue, Gareth Tyson, Haojin Zhu

**Abstracts**: Numerous open-source and commercial malware detectors are available. However, the efficacy of these tools has been threatened by new adversarial attacks, whereby malware attempts to evade detection using, for example, machine learning techniques. In this work, we design an adversarial evasion attack that relies on both feature-space and problem-space manipulation. It uses explainability-guided feature selection to maximize evasion by identifying the most critical features that impact detection. We then use this attack as a benchmark to evaluate several state-of-the-art malware detectors. We find that (i) state-of-the-art malware detectors are vulnerable to even simple evasion strategies, and they can easily be tricked using off-the-shelf techniques; (ii) feature-space manipulation and problem-space obfuscation can be combined to enable evasion without needing white-box understanding of the detector; (iii) we can use explainability approaches (e.g., SHAP) to guide the feature manipulation and explain how attacks can transfer across multiple detectors. Our findings shed light on the weaknesses of current malware detectors, as well as how they can be improved.



## **39. Enhanced countering adversarial attacks via input denoising and feature restoring**

cs.CV

**SubmitDate**: 2021-11-19    [paper-pdf](http://arxiv.org/pdf/2111.10075v1)

**Authors**: Yanni Li, Wenhui Zhang, Jiawei Liu, Xiaoli Kou, Hui Li, Jiangtao Cui

**Abstracts**: Despite the fact that deep neural networks (DNNs) have achieved prominent performance in various applications, it is well known that DNNs are vulnerable to adversarial examples/samples (AEs) with imperceptible perturbations in clean/original samples. To overcome the weakness of the existing defense methods against adversarial attacks, which damages the information on the original samples, leading to the decrease of the target classifier accuracy, this paper presents an enhanced countering adversarial attack method IDFR (via Input Denoising and Feature Restoring). The proposed IDFR is made up of an enhanced input denoiser (ID) and a hidden lossy feature restorer (FR) based on the convex hull optimization. Extensive experiments conducted on benchmark datasets show that the proposed IDFR outperforms the various state-of-the-art defense methods, and is highly effective for protecting target models against various adversarial black-box or white-box attacks. \footnote{Souce code is released at: \href{https://github.com/ID-FR/IDFR}{https://github.com/ID-FR/IDFR}}



## **40. Towards Efficiently Evaluating the Robustness of Deep Neural Networks in IoT Systems: A GAN-based Method**

cs.LG

arXiv admin note: text overlap with arXiv:2002.02196

**SubmitDate**: 2021-11-19    [paper-pdf](http://arxiv.org/pdf/2111.10055v1)

**Authors**: Tao Bai, Jun Zhao, Jinlin Zhu, Shoudong Han, Jiefeng Chen, Bo Li, Alex Kot

**Abstracts**: Intelligent Internet of Things (IoT) systems based on deep neural networks (DNNs) have been widely deployed in the real world. However, DNNs are found to be vulnerable to adversarial examples, which raises people's concerns about intelligent IoT systems' reliability and security. Testing and evaluating the robustness of IoT systems becomes necessary and essential. Recently various attacks and strategies have been proposed, but the efficiency problem remains unsolved properly. Existing methods are either computationally extensive or time-consuming, which is not applicable in practice. In this paper, we propose a novel framework called Attack-Inspired GAN (AI-GAN) to generate adversarial examples conditionally. Once trained, it can generate adversarial perturbations efficiently given input images and target classes. We apply AI-GAN on different datasets in white-box settings, black-box settings and targeted models protected by state-of-the-art defenses. Through extensive experiments, AI-GAN achieves high attack success rates, outperforming existing methods, and reduces generation time significantly. Moreover, for the first time, AI-GAN successfully scales to complex datasets e.g. CIFAR-100 and ImageNet, with about $90\%$ success rates among all classes.



## **41. Generating Unrestricted 3D Adversarial Point Clouds**

cs.CV

**SubmitDate**: 2021-11-19    [paper-pdf](http://arxiv.org/pdf/2111.08973v2)

**Authors**: Xuelong Dai, Yanjie Li, Hua Dai, Bin Xiao

**Abstracts**: Utilizing 3D point cloud data has become an urgent need for the deployment of artificial intelligence in many areas like facial recognition and self-driving. However, deep learning for 3D point clouds is still vulnerable to adversarial attacks, e.g., iterative attacks, point transformation attacks, and generative attacks. These attacks need to restrict perturbations of adversarial examples within a strict bound, leading to the unrealistic adversarial 3D point clouds. In this paper, we propose an Adversarial Graph-Convolutional Generative Adversarial Network (AdvGCGAN) to generate visually realistic adversarial 3D point clouds from scratch. Specifically, we use a graph convolutional generator and a discriminator with an auxiliary classifier to generate realistic point clouds, which learn the latent distribution from the real 3D data. The unrestricted adversarial attack loss is incorporated in the special adversarial training of GAN, which enables the generator to generate the adversarial examples to spoof the target network. Compared with the existing state-of-art attack methods, the experiment results demonstrate the effectiveness of our unrestricted adversarial attack methods with a higher attack success rate and visual quality. Additionally, the proposed AdvGCGAN can achieve better performance against defense models and better transferability than existing attack methods with strong camouflage.



## **42. Arbitrarily Fast Switched Distributed Stabilization of Partially Unknown Interconnected Multiagent Systems: A Proactive Cyber Defense Perspective**

cs.SY

**SubmitDate**: 2021-11-19    [paper-pdf](http://arxiv.org/pdf/2110.14199v2)

**Authors**: Vahid Rezaei, Jafar Haadi Jafarian, Douglas C. Sicker

**Abstracts**: A design framework recently has been developed to stabilize interconnected multiagent systems in a distributed manner, and systematically capture the architectural aspect of cyber-physical systems. Such a control theoretic framework, however, results in a stabilization protocol which is passive with respect to the cyber attacks and conservative regarding the guaranteed level of resiliency. We treat the control layer topology and stabilization gains as the degrees of freedom, and develop a mixed control and cybersecurity design framework to address the above concerns. From a control perspective, despite the agent layer modeling uncertainties and perturbations, we propose a new step-by-step procedure to design a set of control sublayers for an arbitrarily fast switching of the control layer topology. From a proactive cyber defense perspective, we propose a satisfiability modulo theory formulation to obtain a set of control sublayer structures with security considerations, and offer a frequent and fast mutation of these sublayers such that the control layer topology will remain unpredictable for the adversaries. We prove the robust input-to-state stability of the two-layer interconnected multiagent system, and validate the proposed ideas in simulation.



## **43. TnT Attacks! Universal Naturalistic Adversarial Patches Against Deep Neural Network Systems**

cs.CV

We demonstrate physical deployments in multiple videos at  https://tntattacks.github.io/

**SubmitDate**: 2021-11-19    [paper-pdf](http://arxiv.org/pdf/2111.09999v1)

**Authors**: Bao Gia Doan, Minhui Xue, Shiqing Ma, Ehsan Abbasnejad, Damith C. Ranasinghe

**Abstracts**: Deep neural networks are vulnerable to attacks from adversarial inputs and, more recently, Trojans to misguide or hijack the decision of the model. We expose the existence of an intriguing class of bounded adversarial examples -- Universal NaTuralistic adversarial paTches -- we call TnTs, by exploring the superset of the bounded adversarial example space and the natural input space within generative adversarial networks. Now, an adversary can arm themselves with a patch that is naturalistic, less malicious-looking, physically realizable, highly effective -- achieving high attack success rates, and universal. A TnT is universal because any input image captured with a TnT in the scene will: i) misguide a network (untargeted attack); or ii) force the network to make a malicious decision (targeted attack). Interestingly, now, an adversarial patch attacker has the potential to exert a greater level of control -- the ability to choose a location independent, natural-looking patch as a trigger in contrast to being constrained to noisy perturbations -- an ability is thus far shown to be only possible with Trojan attack methods needing to interfere with the model building processes to embed a backdoor at the risk discovery; but, still realize a patch deployable in the physical world. Through extensive experiments on the large-scale visual classification task, ImageNet with evaluations across its entire validation set of 50,000 images, we demonstrate the realistic threat from TnTs and the robustness of the attack. We show a generalization of the attack to create patches achieving higher attack success rates than existing state-of-the-art methods. Our results show the generalizability of the attack to different visual classification tasks (CIFAR-10, GTSRB, PubFig) and multiple state-of-the-art deep neural networks such as WideResnet50, Inception-V3 and VGG-16.



## **44. Combinatorial Bandits under Strategic Manipulations**

cs.LG

**SubmitDate**: 2021-11-19    [paper-pdf](http://arxiv.org/pdf/2102.12722v4)

**Authors**: Jing Dong, Ke Li, Shuai Li, Baoxiang Wang

**Abstracts**: Strategic behavior against sequential learning methods, such as "click framing" in real recommendation systems, have been widely observed. Motivated by such behavior we study the problem of combinatorial multi-armed bandits (CMAB) under strategic manipulations of rewards, where each arm can modify the emitted reward signals for its own interest. This characterization of the adversarial behavior is a relaxation of previously well-studied settings such as adversarial attacks and adversarial corruption. We propose a strategic variant of the combinatorial UCB algorithm, which has a regret of at most $O(m\log T + m B_{max})$ under strategic manipulations, where $T$ is the time horizon, $m$ is the number of arms, and $B_{max}$ is the maximum budget of an arm. We provide lower bounds on the budget for arms to incur certain regret of the bandit algorithm. Extensive experiments on online worker selection for crowdsourcing systems, online influence maximization and online recommendations with both synthetic and real datasets corroborate our theoretical findings on robustness and regret bounds, in a variety of regimes of manipulation budgets.



## **45. A Review of Adversarial Attack and Defense for Classification Methods**

cs.CR

**SubmitDate**: 2021-11-18    [paper-pdf](http://arxiv.org/pdf/2111.09961v1)

**Authors**: Yao Li, Minhao Cheng, Cho-Jui Hsieh, Thomas C. M. Lee

**Abstracts**: Despite the efficiency and scalability of machine learning systems, recent studies have demonstrated that many classification methods, especially deep neural networks (DNNs), are vulnerable to adversarial examples; i.e., examples that are carefully crafted to fool a well-trained classification model while being indistinguishable from natural data to human. This makes it potentially unsafe to apply DNNs or related methods in security-critical areas. Since this issue was first identified by Biggio et al. (2013) and Szegedy et al.(2014), much work has been done in this field, including the development of attack methods to generate adversarial examples and the construction of defense techniques to guard against such examples. This paper aims to introduce this topic and its latest developments to the statistical community, primarily focusing on the generation and guarding of adversarial examples. Computing codes (in python and R) used in the numerical experiments are publicly available for readers to explore the surveyed methods. It is the hope of the authors that this paper will encourage more statisticians to work on this important and exciting field of generating and defending against adversarial examples.



## **46. Resilient Consensus-based Multi-agent Reinforcement Learning with Function Approximation**

cs.LG

**SubmitDate**: 2021-11-18    [paper-pdf](http://arxiv.org/pdf/2111.06776v2)

**Authors**: Martin Figura, Yixuan Lin, Ji Liu, Vijay Gupta

**Abstracts**: Adversarial attacks during training can strongly influence the performance of multi-agent reinforcement learning algorithms. It is, thus, highly desirable to augment existing algorithms such that the impact of adversarial attacks on cooperative networks is eliminated, or at least bounded. In this work, we consider a fully decentralized network, where each agent receives a local reward and observes the global state and action. We propose a resilient consensus-based actor-critic algorithm, whereby each agent estimates the team-average reward and value function, and communicates the associated parameter vectors to its immediate neighbors. We show that in the presence of Byzantine agents, whose estimation and communication strategies are completely arbitrary, the estimates of the cooperative agents converge to a bounded consensus value with probability one, provided that there are at most $H$ Byzantine agents in the neighborhood of each cooperative agent and the network is $(2H+1)$-robust. Furthermore, we prove that the policy of the cooperative agents converges with probability one to a bounded neighborhood around a local maximizer of their team-average objective function under the assumption that the policies of the adversarial agents asymptotically become stationary.



## **47. Robust Person Re-identification with Multi-Modal Joint Defence**

cs.CV

**SubmitDate**: 2021-11-18    [paper-pdf](http://arxiv.org/pdf/2111.09571v1)

**Authors**: Yunpeng Gong, Lifei Chen

**Abstracts**: The Person Re-identification (ReID) system based on metric learning has been proved to inherit the vulnerability of deep neural networks (DNNs), which are easy to be fooled by adversarail metric attacks. Existing work mainly relies on adversarial training for metric defense, and more methods have not been fully studied. By exploring the impact of attacks on the underlying features, we propose targeted methods for metric attacks and defence methods. In terms of metric attack, we use the local color deviation to construct the intra-class variation of the input to attack color features. In terms of metric defenses, we propose a joint defense method which includes two parts of proactive defense and passive defense. Proactive defense helps to enhance the robustness of the model to color variations and the learning of structure relations across multiple modalities by constructing different inputs from multimodal images, and passive defense exploits the invariance of structural features in a changing pixel space by circuitous scaling to preserve structural features while eliminating some of the adversarial noise. Extensive experiments demonstrate that the proposed joint defense compared with the existing adversarial metric defense methods which not only against multiple attacks at the same time but also has not significantly reduced the generalization capacity of the model. The code is available at https://github.com/finger-monkey/multi-modal_joint_defence.



## **48. DPA: Learning Robust Physical Adversarial Camouflages for Object Detectors**

cs.CV

**SubmitDate**: 2021-11-18    [paper-pdf](http://arxiv.org/pdf/2109.00124v2)

**Authors**: Yexin Duan, Jialin Chen, Xingyu Zhou, Junhua Zou, Zhengyun He, Wu Zhang, Jin Zhang, Zhisong Pan

**Abstracts**: Adversarial attacks are feasible in the real world for object detection. However, most of the previous works have tried to learn local "patches" applied to an object to fool detectors, which become less effective in squint view angles. To address this issue, we propose the Dense Proposals Attack (DPA) to learn one-piece, physical, and targeted adversarial camouflages for detectors. The camouflages are one-piece because they are generated as a whole for an object, physical because they remain adversarial when filmed under arbitrary viewpoints and different illumination conditions, and targeted because they can cause detectors to misidentify an object as a specific target class. In order to make the generated camouflages robust in the physical world, we introduce a combination of transformations to model the physical phenomena. In addition, to improve the attacks, DPA simultaneously attacks all the classifications in the fixed proposals. Moreover, we build a virtual 3D scene using the Unity simulation engine to fairly and reproducibly evaluate different physical attacks. Extensive experiments demonstrate that DPA outperforms the state-of-the-art methods, and it is generic for any object and generalized well to the real world, posing a potential threat to the security-critical computer vision systems.



## **49. Adversarial attacks on voter model dynamics in complex networks**

physics.soc-ph

6 pages, 4 figures

**SubmitDate**: 2021-11-18    [paper-pdf](http://arxiv.org/pdf/2111.09561v1)

**Authors**: Katsumi Chiyomaru, Kazuhiro Takemoto

**Abstracts**: This study investigates adversarial attacks conducted to distort the voter model dynamics in complex networks. Specifically, a simple adversarial attack method is proposed for holding the state of an individual's opinions closer to the target state in the voter model dynamics; the method shows that even when one opinion is the majority, the vote outcome can be inverted (i.e., the outcome can lean toward the other opinion) by adding extremely small (hard-to-detect) perturbations strategically generated in social networks. Adversarial attacks are relatively more effective for complex (large and dense) networks. The results indicate that opinion dynamics can be unknowingly distorted.



## **50. ZeBRA: Precisely Destroying Neural Networks with Zero-Data Based Repeated Bit Flip Attack**

cs.LG

14 pages, 3 figures, 5 tables, Accepted at British Machine Vision  Conference (BMVC) 2021

**SubmitDate**: 2021-11-18    [paper-pdf](http://arxiv.org/pdf/2111.01080v2)

**Authors**: Dahoon Park, Kon-Woo Kwon, Sunghoon Im, Jaeha Kung

**Abstracts**: In this paper, we present Zero-data Based Repeated bit flip Attack (ZeBRA) that precisely destroys deep neural networks (DNNs) by synthesizing its own attack datasets. Many prior works on adversarial weight attack require not only the weight parameters, but also the training or test dataset in searching vulnerable bits to be attacked. We propose to synthesize the attack dataset, named distilled target data, by utilizing the statistics of batch normalization layers in the victim DNN model. Equipped with the distilled target data, our ZeBRA algorithm can search vulnerable bits in the model without accessing training or test dataset. Thus, our approach makes the adversarial weight attack more fatal to the security of DNNs. Our experimental results show that 2.0x (CIFAR-10) and 1.6x (ImageNet) less number of bit flips are required on average to destroy DNNs compared to the previous attack method. Our code is available at https://github. com/pdh930105/ZeBRA.



