# Latest Adversarial Attack Papers
**update at 2022-08-14 18:37:17**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. A Survey of MulVAL Extensions and Their Attack Scenarios Coverage**

cs.CR

**SubmitDate**: 2022-08-11    [paper-pdf](http://arxiv.org/pdf/2208.05750v1)

**Authors**: David Tayouri, Nick Baum, Asaf Shabtai, Rami Puzis

**Abstracts**: Organizations employ various adversary models in order to assess the risk and potential impact of attacks on their networks. Attack graphs represent vulnerabilities and actions an attacker can take to identify and compromise an organization's assets. Attack graphs facilitate both visual presentation and algorithmic analysis of attack scenarios in the form of attack paths. MulVAL is a generic open-source framework for constructing logical attack graphs, which has been widely used by researchers and practitioners and extended by them with additional attack scenarios. This paper surveys all of the existing MulVAL extensions, and maps all MulVAL interaction rules to MITRE ATT&CK Techniques to estimate their attack scenarios coverage. This survey aligns current MulVAL extensions along unified ontological concepts and highlights the existing gaps. It paves the way for methodical improvement of MulVAL and the comprehensive modeling of the entire landscape of adversarial behaviors captured in MITRE ATT&CK.



## **2. Diverse Generative Adversarial Perturbations on Attention Space for Transferable Adversarial Attacks**

cs.CV

ICIP 2022

**SubmitDate**: 2022-08-11    [paper-pdf](http://arxiv.org/pdf/2208.05650v1)

**Authors**: Woo Jae Kim, Seunghoon Hong, Sung-Eui Yoon

**Abstracts**: Adversarial attacks with improved transferability - the ability of an adversarial example crafted on a known model to also fool unknown models - have recently received much attention due to their practicality. Nevertheless, existing transferable attacks craft perturbations in a deterministic manner and often fail to fully explore the loss surface, thus falling into a poor local optimum and suffering from low transferability. To solve this problem, we propose Attentive-Diversity Attack (ADA), which disrupts diverse salient features in a stochastic manner to improve transferability. Primarily, we perturb the image attention to disrupt universal features shared by different models. Then, to effectively avoid poor local optima, we disrupt these features in a stochastic manner and explore the search space of transferable perturbations more exhaustively. More specifically, we use a generator to produce adversarial perturbations that each disturbs features in different ways depending on an input latent code. Extensive experimental evaluations demonstrate the effectiveness of our method, outperforming the transferability of state-of-the-art methods. Codes are available at https://github.com/wkim97/ADA.



## **3. Controlled Quantum Teleportation in the Presence of an Adversary**

quant-ph

**SubmitDate**: 2022-08-10    [paper-pdf](http://arxiv.org/pdf/2208.05554v1)

**Authors**: Sayan Gangopadhyay, Tiejun Wang, Atefeh Mashatan, Shohini Ghose

**Abstracts**: We present a device independent analysis of controlled quantum teleportation where the receiver is not trusted. We show that the notion of genuine tripartite nonlocality allows us to certify control power in such a scenario. By considering a specific adversarial attack strategy on a device characterized by depolarizing noise, we find that control power is a monotonically increasing function of genuine tripartite nonlocality. These results are relevant for building practical quantum communication networks and also shed light on the role of nonlocality in multipartite quantum information processing.



## **4. Pikachu: Securing PoS Blockchains from Long-Range Attacks by Checkpointing into Bitcoin PoW using Taproot**

cs.CR

To appear at ConsensusDay 22 (ACM CCS 2022 Workshop)

**SubmitDate**: 2022-08-10    [paper-pdf](http://arxiv.org/pdf/2208.05408v1)

**Authors**: Sarah Azouvi, Marko Vukolić

**Abstracts**: Blockchain systems based on a reusable resource, such as proof-of-stake (PoS), provide weaker security guarantees than those based on proof-of-work. Specifically, they are vulnerable to long-range attacks, where an adversary can corrupt prior participants in order to rewrite the full history of the chain. To prevent this attack on a PoS chain, we propose a protocol that checkpoints the state of the PoS chain to a proof-of-work blockchain such as Bitcoin. Our checkpointing protocol hence does not rely on any central authority. Our work uses Schnorr signatures and leverages Bitcoin recent Taproot upgrade, allowing us to create a checkpointing transaction of constant size. We argue for the security of our protocol and present an open-source implementation that was tested on the Bitcoin testnet.



## **5. StratDef: a strategic defense against adversarial attacks in malware detection**

cs.LG

**SubmitDate**: 2022-08-10    [paper-pdf](http://arxiv.org/pdf/2202.07568v2)

**Authors**: Aqib Rashid, Jose Such

**Abstracts**: Over the years, most research towards defenses against adversarial attacks on machine learning models has been in the image recognition domain. The malware detection domain has received less attention despite its importance. Moreover, most work exploring these defenses has focused on several methods but with no strategy when applying them. In this paper, we introduce StratDef, which is a strategic defense system tailored for the malware detection domain based on a moving target defense approach. We overcome challenges related to the systematic construction, selection and strategic use of models to maximize adversarial robustness. StratDef dynamically and strategically chooses the best models to increase the uncertainty for the attacker, whilst minimizing critical aspects in the adversarial ML domain like attack transferability. We provide the first comprehensive evaluation of defenses against adversarial attacks on machine learning for malware detection, where our threat model explores different levels of threat, attacker knowledge, capabilities, and attack intensities. We show that StratDef performs better than other defenses even when facing the peak adversarial threat. We also show that, from the existing defenses, only a few adversarially-trained models provide substantially better protection than just using vanilla models but are still outperformed by StratDef.



## **6. Reducing Exploitability with Population Based Training**

cs.LG

Presented at New Frontiers in Adversarial Machine Learning Workshop,  ICML 2022

**SubmitDate**: 2022-08-10    [paper-pdf](http://arxiv.org/pdf/2208.05083v1)

**Authors**: Pavel Czempin, Adam Gleave

**Abstracts**: Self-play reinforcement learning has achieved state-of-the-art, and often superhuman, performance in a variety of zero-sum games. Yet prior work has found that policies that are highly capable against regular opponents can fail catastrophically against adversarial policies: an opponent trained explicitly against the victim. Prior defenses using adversarial training were able to make the victim robust to a specific adversary, but the victim remained vulnerable to new ones. We conjecture this limitation was due to insufficient diversity of adversaries seen during training. We propose a defense using population based training to pit the victim against a diverse set of opponents. We evaluate this defense's robustness against new adversaries in two low-dimensional environments. Our defense increases robustness against adversaries, as measured by number of attacker training timesteps to exploit the victim. Furthermore, we show that robustness is correlated with the size of the opponent population.



## **7. Adversarial Machine Learning-Based Anticipation of Threats Against Vehicle-to-Microgrid Services**

cs.CR

IEEE Global Communications Conference (Globecom), 2022, 6 pages, 2  Figures, 4 Tables

**SubmitDate**: 2022-08-09    [paper-pdf](http://arxiv.org/pdf/2208.05073v1)

**Authors**: Ahmed Omara, Burak Kantarci

**Abstracts**: In this paper, we study the expanding attack surface of Adversarial Machine Learning (AML) and the potential attacks against Vehicle-to-Microgrid (V2M) services. We present an anticipatory study of a multi-stage gray-box attack that can achieve a comparable result to a white-box attack. Adversaries aim to deceive the targeted Machine Learning (ML) classifier at the network edge to misclassify the incoming energy requests from microgrids. With an inference attack, an adversary can collect real-time data from the communication between smart microgrids and a 5G gNodeB to train a surrogate (i.e., shadow) model of the targeted classifier at the edge. To anticipate the associated impact of an adversary's capability to collect real-time data instances, we study five different cases, each representing different amounts of real-time data instances collected by an adversary. Out of six ML models trained on the complete dataset, K-Nearest Neighbour (K-NN) is selected as the surrogate model, and through simulations, we demonstrate that the multi-stage gray-box attack is able to mislead the ML classifier and cause an Evasion Increase Rate (EIR) up to 73.2% using 40% less data than what a white-box attack needs to achieve a similar EIR.



## **8. Get your Foes Fooled: Proximal Gradient Split Learning for Defense against Model Inversion Attacks on IoMT data**

cs.CR

10 pages, 5 figures, 2 tables

**SubmitDate**: 2022-08-09    [paper-pdf](http://arxiv.org/pdf/2201.04569v3)

**Authors**: Sunder Ali Khowaja, Ik Hyun Lee, Kapal Dev, Muhammad Aslam Jarwar, Nawab Muhammad Faseeh Qureshi

**Abstracts**: The past decade has seen a rapid adoption of Artificial Intelligence (AI), specifically the deep learning networks, in Internet of Medical Things (IoMT) ecosystem. However, it has been shown recently that the deep learning networks can be exploited by adversarial attacks that not only make IoMT vulnerable to the data theft but also to the manipulation of medical diagnosis. The existing studies consider adding noise to the raw IoMT data or model parameters which not only reduces the overall performance concerning medical inferences but also is ineffective to the likes of deep leakage from gradients method. In this work, we propose proximal gradient split learning (PSGL) method for defense against the model inversion attacks. The proposed method intentionally attacks the IoMT data when undergoing the deep neural network training process at client side. We propose the use of proximal gradient method to recover gradient maps and a decision-level fusion strategy to improve the recognition performance. Extensive analysis show that the PGSL not only provides effective defense mechanism against the model inversion attacks but also helps in improving the recognition performance on publicly available datasets. We report 14.0$\%$, 17.9$\%$, and 36.9$\%$ gains in accuracy over reconstructed and adversarial attacked images, respectively.



## **9. Bayesian Pseudo Labels: Expectation Maximization for Robust and Efficient Semi-Supervised Segmentation**

cs.CV

MICCAI 2022 (Early accept, Student Travel Award)

**SubmitDate**: 2022-08-08    [paper-pdf](http://arxiv.org/pdf/2208.04435v1)

**Authors**: Mou-Cheng Xu, Yukun Zhou, Chen Jin, Marius de Groot, Daniel C. Alexander, Neil P. Oxtoby, Yipeng Hu, Joseph Jacob

**Abstracts**: This paper concerns pseudo labelling in segmentation. Our contribution is fourfold. Firstly, we present a new formulation of pseudo-labelling as an Expectation-Maximization (EM) algorithm for clear statistical interpretation. Secondly, we propose a semi-supervised medical image segmentation method purely based on the original pseudo labelling, namely SegPL. We demonstrate SegPL is a competitive approach against state-of-the-art consistency regularisation based methods on semi-supervised segmentation on a 2D multi-class MRI brain tumour segmentation task and a 3D binary CT lung vessel segmentation task. The simplicity of SegPL allows less computational cost comparing to prior methods. Thirdly, we demonstrate that the effectiveness of SegPL may originate from its robustness against out-of-distribution noises and adversarial attacks. Lastly, under the EM framework, we introduce a probabilistic generalisation of SegPL via variational inference, which learns a dynamic threshold for pseudo labelling during the training. We show that SegPL with variational inference can perform uncertainty estimation on par with the gold-standard method Deep Ensemble.



## **10. Can collaborative learning be private, robust and scalable?**

cs.LG

Accepted at MICCAI DeCaF 2022

**SubmitDate**: 2022-08-08    [paper-pdf](http://arxiv.org/pdf/2205.02652v2)

**Authors**: Dmitrii Usynin, Helena Klause, Johannes C. Paetzold, Daniel Rueckert, Georgios Kaissis

**Abstracts**: In federated learning for medical image analysis, the safety of the learning protocol is paramount. Such settings can often be compromised by adversaries that target either the private data used by the federation or the integrity of the model itself. This requires the medical imaging community to develop mechanisms to train collaborative models that are private and robust against adversarial data. In response to these challenges, we propose a practical open-source framework to study the effectiveness of combining differential privacy, model compression and adversarial training to improve the robustness of models against adversarial samples under train- and inference-time attacks. Using our framework, we achieve competitive model performance, a significant reduction in model's size and an improved empirical adversarial robustness without a severe performance degradation, critical in medical image analysis.



## **11. Sparse Adversarial Attack in Multi-agent Reinforcement Learning**

cs.AI

**SubmitDate**: 2022-08-08    [paper-pdf](http://arxiv.org/pdf/2205.09362v2)

**Authors**: Yizheng Hu, Zhihua Zhang

**Abstracts**: Cooperative multi-agent reinforcement learning (cMARL) has many real applications, but the policy trained by existing cMARL algorithms is not robust enough when deployed. There exist also many methods about adversarial attacks on the RL system, which implies that the RL system can suffer from adversarial attacks, but most of them focused on single agent RL. In this paper, we propose a \textit{sparse adversarial attack} on cMARL systems. We use (MA)RL with regularization to train the attack policy. Our experiments show that the policy trained by the current cMARL algorithm can obtain poor performance when only one or a few agents in the team (e.g., 1 of 8 or 5 of 25) were attacked at a few timesteps (e.g., attack 3 of total 40 timesteps).



## **12. Adversarial Pixel Restoration as a Pretext Task for Transferable Perturbations**

cs.CV

**SubmitDate**: 2022-08-08    [paper-pdf](http://arxiv.org/pdf/2207.08803v2)

**Authors**: Hashmat Shadab Malik, Shahina K Kunhimon, Muzammal Naseer, Salman Khan, Fahad Shahbaz Khan

**Abstracts**: Transferable adversarial attacks optimize adversaries from a pretrained surrogate model and known label space to fool the unknown black-box models. Therefore, these attacks are restricted by the availability of an effective surrogate model. In this work, we relax this assumption and propose Adversarial Pixel Restoration as a self-supervised alternative to train an effective surrogate model from scratch under the condition of no labels and few data samples. Our training approach is based on a min-max scheme which reduces overfitting via an adversarial objective and thus optimizes for a more generalizable surrogate model. Our proposed attack is complimentary to the adversarial pixel restoration and is independent of any task specific objective as it can be launched in a self-supervised manner. We successfully demonstrate the adversarial transferability of our approach to Vision Transformers as well as Convolutional Neural Networks for the tasks of classification, object detection, and video segmentation. Our training approach improves the transferability of the baseline unsupervised training method by 16.4% on ImageNet val. set. Our codes & pre-trained surrogate models are available at: https://github.com/HashmatShadab/APR



## **13. Adversarial robustness of $β-$VAE through the lens of local geometry**

cs.LG

The 2022 ICML Workshop on New Frontiers in Adversarial Machine  Learning

**SubmitDate**: 2022-08-08    [paper-pdf](http://arxiv.org/pdf/2208.03923v1)

**Authors**: Asif Khan, Amos Storkey

**Abstracts**: Variational autoencoders (VAEs) are susceptible to adversarial attacks. An adversary can find a small perturbation in the input sample to change its latent encoding non-smoothly, thereby compromising the reconstruction. A known reason for such vulnerability is the latent space distortions arising from a mismatch between approximated latent posterior and a prior distribution. Consequently, a slight change in the inputs leads to a significant change in the latent space encodings. This paper demonstrates that the sensitivity around a data point is due to a directional bias of a stochastic pullback metric tensor induced by the encoder network. The pullback metric tensor measures the infinitesimal volume change from input to latent space. Thus, it can be viewed as a lens to analyse the effect of small changes in the input leading to distortions in the latent space. We propose robustness evaluation scores using the eigenspectrum of a pullback metric. Moreover, we empirically show that the scores correlate with the robustness parameter $\beta$ of the $\beta-$VAE.



## **14. Adversarial Fine-tuning for Backdoor Defense: Connecting Backdoor Attacks to Adversarial Attacks**

cs.CV

**SubmitDate**: 2022-08-08    [paper-pdf](http://arxiv.org/pdf/2202.06312v3)

**Authors**: Bingxu Mu, Zhenxing Niu, Le Wang, Xue Wang, Rong Jin, Gang Hua

**Abstracts**: Deep neural networks (DNNs) are known to be vulnerable to both backdoor attacks as well as adversarial attacks. In the literature, these two types of attacks are commonly treated as distinct problems and solved separately, since they belong to training-time and inference-time attacks respectively. However, in this paper we find an intriguing connection between them: for a model planted with backdoors, we observe that its adversarial examples have similar behaviors as its triggered samples, i.e., both activate the same subset of DNN neurons. It indicates that planting a backdoor into a model will significantly affect the model's adversarial examples. Based on this observations, we design a new Adversarial Fine-Tuning (AFT) algorithm to defend against backdoor attacks. We empirically show that, against 5 state-of-the-art backdoor attacks, our AFT can effectively erase the backdoor triggers without obvious performance degradation on clean samples and significantly outperforms existing defense methods.



## **15. Privacy Against Inference Attacks in Vertical Federated Learning**

cs.LG

**SubmitDate**: 2022-08-07    [paper-pdf](http://arxiv.org/pdf/2207.11788v2)

**Authors**: Borzoo Rassouli, Morteza Varasteh, Deniz Gunduz

**Abstracts**: Vertical federated learning is considered, where an active party, having access to true class labels, wishes to build a classification model by utilizing more features from a passive party, which has no access to the labels, to improve the model accuracy. In the prediction phase, with logistic regression as the classification model, several inference attack techniques are proposed that the adversary, i.e., the active party, can employ to reconstruct the passive party's features, regarded as sensitive information. These attacks, which are mainly based on a classical notion of the center of a set, i.e., the Chebyshev center, are shown to be superior to those proposed in the literature. Moreover, several theoretical performance guarantees are provided for the aforementioned attacks. Subsequently, we consider the minimum amount of information that the adversary needs to fully reconstruct the passive party's features. In particular, it is shown that when the passive party holds one feature, and the adversary is only aware of the signs of the parameters involved, it can perfectly reconstruct that feature when the number of predictions is large enough. Next, as a defense mechanism, a privacy-preserving scheme is proposed that worsen the adversary's reconstruction attacks, while preserving the full benefits that VFL brings to the active party. Finally, experimental results demonstrate the effectiveness of the proposed attacks and the privacy-preserving scheme.



## **16. Garbled EDA: Privacy Preserving Electronic Design Automation**

cs.CR

**SubmitDate**: 2022-08-07    [paper-pdf](http://arxiv.org/pdf/2208.03822v1)

**Authors**: Mohammad Hashemi, Steffi Roy, Fatemeh Ganji, Domenic Forte

**Abstracts**: The complexity of modern integrated circuits (ICs) necessitates collaboration between multiple distrusting parties, including thirdparty intellectual property (3PIP) vendors, design houses, CAD/EDA tool vendors, and foundries, which jeopardizes confidentiality and integrity of each party's IP. IP protection standards and the existing techniques proposed by researchers are ad hoc and vulnerable to numerous structural, functional, and/or side-channel attacks. Our framework, Garbled EDA, proposes an alternative direction through formulating the problem in a secure multi-party computation setting, where the privacy of IPs, CAD tools, and process design kits (PDKs) is maintained. As a proof-of-concept, Garbled EDA is evaluated in the context of simulation, where multiple IP description formats (Verilog, C, S) are supported. Our results demonstrate a reasonable logical-resource cost and negligible memory overhead. To further reduce the overhead, we present another efficient implementation methodology, feasible when the resource utilization is a bottleneck, but the communication between two parties is not restricted. Interestingly, this implementation is private and secure even in the presence of malicious adversaries attempting to, e.g., gain access to PDKs or in-house IPs of the CAD tool providers.



## **17. Federated Adversarial Learning: A Framework with Convergence Analysis**

cs.LG

**SubmitDate**: 2022-08-07    [paper-pdf](http://arxiv.org/pdf/2208.03635v1)

**Authors**: Xiaoxiao Li, Zhao Song, Jiaming Yang

**Abstracts**: Federated learning (FL) is a trending training paradigm to utilize decentralized training data. FL allows clients to update model parameters locally for several epochs, then share them to a global model for aggregation. This training paradigm with multi-local step updating before aggregation exposes unique vulnerabilities to adversarial attacks. Adversarial training is a popular and effective method to improve the robustness of networks against adversaries. In this work, we formulate a general form of federated adversarial learning (FAL) that is adapted from adversarial learning in the centralized setting. On the client side of FL training, FAL has an inner loop to generate adversarial samples for adversarial training and an outer loop to update local model parameters. On the server side, FAL aggregates local model updates and broadcast the aggregated model. We design a global robust training loss and formulate FAL training as a min-max optimization problem. Unlike the convergence analysis in classical centralized training that relies on the gradient direction, it is significantly harder to analyze the convergence in FAL for three reasons: 1) the complexity of min-max optimization, 2) model not updating in the gradient direction due to the multi-local updates on the client-side before aggregation and 3) inter-client heterogeneity. We address these challenges by using appropriate gradient approximation and coupling techniques and present the convergence analysis in the over-parameterized regime. Our main result theoretically shows that the minimum loss under our algorithm can converge to $\epsilon$ small with chosen learning rate and communication rounds. It is noteworthy that our analysis is feasible for non-IID clients.



## **18. Blackbox Attacks via Surrogate Ensemble Search**

cs.LG

**SubmitDate**: 2022-08-07    [paper-pdf](http://arxiv.org/pdf/2208.03610v1)

**Authors**: Zikui Cai, Chengyu Song, Srikanth Krishnamurthy, Amit Roy-Chowdhury, M. Salman Asif

**Abstracts**: Blackbox adversarial attacks can be categorized into transfer- and query-based attacks. Transfer methods do not require any feedback from the victim model, but provide lower success rates compared to query-based methods. Query attacks often require a large number of queries for success. To achieve the best of both approaches, recent efforts have tried to combine them, but still require hundreds of queries to achieve high success rates (especially for targeted attacks). In this paper, we propose a novel method for blackbox attacks via surrogate ensemble search (BASES) that can generate highly successful blackbox attacks using an extremely small number of queries. We first define a perturbation machine that generates a perturbed image by minimizing a weighted loss function over a fixed set of surrogate models. To generate an attack for a given victim model, we search over the weights in the loss function using queries generated by the perturbation machine. Since the dimension of the search space is small (same as the number of surrogate models), the search requires a small number of queries. We demonstrate that our proposed method achieves better success rate with at least 30x fewer queries compared to state-of-the-art methods on different image classifiers trained with ImageNet (including VGG-19, DenseNet-121, and ResNext-50). In particular, our method requires as few as 3 queries per image (on average) to achieve more than a 90% success rate for targeted attacks and 1-2 queries per image for over a 99% success rate for non-targeted attacks. Our method is also effective on Google Cloud Vision API and achieved a 91% non-targeted attack success rate with 2.9 queries per image. We also show that the perturbations generated by our proposed method are highly transferable and can be adopted for hard-label blackbox attacks.



## **19. Revisiting Gaussian Neurons for Online Clustering with Unknown Number of Clusters**

cs.LG

Reviewed at  https://openreview.net/forum?id=h05RLBNweX&referrer=%5BTMLR%5D(%2Fgroup%3Fid%3DTMLR)

**SubmitDate**: 2022-08-06    [paper-pdf](http://arxiv.org/pdf/2205.00920v2)

**Authors**: Ole Christian Eidheim

**Abstracts**: Despite the recent success of artificial neural networks, more biologically plausible learning methods may be needed to resolve the weaknesses of backpropagation trained models such as catastrophic forgetting and adversarial attacks. Although these weaknesses are not specifically addressed, a novel local learning rule is presented that performs online clustering with an upper limit on the number of clusters to be found rather than a fixed cluster count. Instead of using orthogonal weight or output activation constraints, activation sparsity is achieved by mutual repulsion of lateral Gaussian neurons ensuring that multiple neuron centers cannot occupy the same location in the input domain. An update method is also presented for adjusting the widths of the Gaussian neurons in cases where the data samples can be represented by means and variances. The algorithms were applied on the MNIST and CIFAR-10 datasets to create filters capturing the input patterns of pixel patches of various sizes. The experimental results demonstrate stability in the learned parameters across a large number of training samples.



## **20. On the Fundamental Limits of Formally (Dis)Proving Robustness in Proof-of-Learning**

cs.LG

**SubmitDate**: 2022-08-06    [paper-pdf](http://arxiv.org/pdf/2208.03567v1)

**Authors**: Congyu Fang, Hengrui Jia, Anvith Thudi, Mohammad Yaghini, Christopher A. Choquette-Choo, Natalie Dullerud, Varun Chandrasekaran, Nicolas Papernot

**Abstracts**: Proof-of-learning (PoL) proposes a model owner use machine learning training checkpoints to establish a proof of having expended the necessary compute for training. The authors of PoL forego cryptographic approaches and trade rigorous security guarantees for scalability to deep learning by being applicable to stochastic gradient descent and adaptive variants. This lack of formal analysis leaves the possibility that an attacker may be able to spoof a proof for a model they did not train.   We contribute a formal analysis of why the PoL protocol cannot be formally (dis)proven to be robust against spoofing adversaries. To do so, we disentangle the two roles of proof verification in PoL: (a) efficiently determining if a proof is a valid gradient descent trajectory, and (b) establishing precedence by making it more expensive to craft a proof after training completes (i.e., spoofing). We show that efficient verification results in a tradeoff between accepting legitimate proofs and rejecting invalid proofs because deep learning necessarily involves noise. Without a precise analytical model for how this noise affects training, we cannot formally guarantee if a PoL verification algorithm is robust. Then, we demonstrate that establishing precedence robustly also reduces to an open problem in learning theory: spoofing a PoL post hoc training is akin to finding different trajectories with the same endpoint in non-convex learning. Yet, we do not rigorously know if priori knowledge of the final model weights helps discover such trajectories.   We conclude that, until the aforementioned open problems are addressed, relying more heavily on cryptography is likely needed to formulate a new class of PoL protocols with formal robustness guarantees. In particular, this will help with establishing precedence. As a by-product of insights from our analysis, we also demonstrate two novel attacks against PoL.



## **21. Preventing or Mitigating Adversarial Supply Chain Attacks; a legal analysis**

cs.CY

23 pages

**SubmitDate**: 2022-08-06    [paper-pdf](http://arxiv.org/pdf/2208.03466v1)

**Authors**: Kaspar Rosager Ludvigsen, Shishir Nagaraja, Angela Daly

**Abstracts**: The world is currently strongly connected through both the internet at large, but also the very supply chains which provide everything from food to infrastructure and technology. The supply chains are themselves vulnerable to adversarial attacks, both in a digital and physical sense, which can disrupt or at worst destroy them. In this paper, we take a look at two examples of such successful attacks and consider what their consequences may be going forward, and analyse how EU and national law can prevent these attacks or otherwise punish companies which do not try to mitigate them at all possible costs. We find that the current types of national regulation are not technology specific enough, and cannot force or otherwise mandate the correct parties who could play the biggest role in preventing supply chain attacks to do everything in their power to mitigate them. But, current EU law is on the right path, and further vigilance may be what is necessary to consider these large threats, as national law tends to fail at properly regulating companies when it comes to cybersecurity.



## **22. Searching for the Essence of Adversarial Perturbations**

cs.LG

**SubmitDate**: 2022-08-06    [paper-pdf](http://arxiv.org/pdf/2205.15357v2)

**Authors**: Dennis Y. Menn, Hung-yi Lee

**Abstracts**: Neural networks have achieved the state-of-the-art performance in various machine learning fields, yet the incorporation of malicious perturbations with input data (adversarial example) is shown to fool neural networks' predictions. This would lead to potential risks for real-world applications such as endangering autonomous driving and messing up text identification. To mitigate such risks, an understanding of how adversarial examples operate is critical, which however remains unresolved. Here we demonstrate that adversarial perturbations contain human-recognizable information, which is the key conspirator responsible for a neural network's erroneous prediction, in contrast to a widely discussed argument that human-imperceptible information plays the critical role in fooling a network. This concept of human-recognizable information allows us to explain key features related to adversarial perturbations, including the existence of adversarial examples, the transferability among different neural networks, and the increased neural network interpretability for adversarial training. Two unique properties in adversarial perturbations that fool neural networks are uncovered: masking and generation. A special class, the complementary class, is identified when neural networks classify input images. The human-recognizable information contained in adversarial perturbations allows researchers to gain insight on the working principles of neural networks and may lead to develop techniques that detect/defense adversarial attacks.



## **23. Success of Uncertainty-Aware Deep Models Depends on Data Manifold Geometry**

cs.LG

**SubmitDate**: 2022-08-05    [paper-pdf](http://arxiv.org/pdf/2208.01705v2)

**Authors**: Mark Penrod, Harrison Termotto, Varshini Reddy, Jiayu Yao, Finale Doshi-Velez, Weiwei Pan

**Abstracts**: For responsible decision making in safety-critical settings, machine learning models must effectively detect and process edge-case data. Although existing works show that predictive uncertainty is useful for these tasks, it is not evident from literature which uncertainty-aware models are best suited for a given dataset. Thus, we compare six uncertainty-aware deep learning models on a set of edge-case tasks: robustness to adversarial attacks as well as out-of-distribution and adversarial detection. We find that the geometry of the data sub-manifold is an important factor in determining the success of various models. Our finding suggests an interesting direction in the study of uncertainty-aware deep learning models.



## **24. Attacking Adversarial Defences by Smoothing the Loss Landscape**

cs.LG

**SubmitDate**: 2022-08-05    [paper-pdf](http://arxiv.org/pdf/2208.00862v2)

**Authors**: Panagiotis Eustratiadis, Henry Gouk, Da Li, Timothy Hospedales

**Abstracts**: This paper investigates a family of methods for defending against adversarial attacks that owe part of their success to creating a noisy, discontinuous, or otherwise rugged loss landscape that adversaries find difficult to navigate. A common, but not universal, way to achieve this effect is via the use of stochastic neural networks. We show that this is a form of gradient obfuscation, and propose a general extension to gradient-based adversaries based on the Weierstrass transform, which smooths the surface of the loss function and provides more reliable gradient estimates. We further show that the same principle can strengthen gradient-free adversaries. We demonstrate the efficacy of our loss-smoothing method against both stochastic and non-stochastic adversarial defences that exhibit robustness due to this type of obfuscation. Furthermore, we provide analysis of how it interacts with Expectation over Transformation; a popular gradient-sampling method currently used to attack stochastic defences.



## **25. Adversarial Robustness of MR Image Reconstruction under Realistic Perturbations**

eess.IV

Accepted at the MICCAI-2022 workshop: Machine Learning for Medical  Image Reconstruction

**SubmitDate**: 2022-08-05    [paper-pdf](http://arxiv.org/pdf/2208.03161v1)

**Authors**: Jan Nikolas Morshuis, Sergios Gatidis, Matthias Hein, Christian F. Baumgartner

**Abstracts**: Deep Learning (DL) methods have shown promising results for solving ill-posed inverse problems such as MR image reconstruction from undersampled $k$-space data. However, these approaches currently have no guarantees for reconstruction quality and the reliability of such algorithms is only poorly understood. Adversarial attacks offer a valuable tool to understand possible failure modes and worst case performance of DL-based reconstruction algorithms. In this paper we describe adversarial attacks on multi-coil $k$-space measurements and evaluate them on the recently proposed E2E-VarNet and a simpler UNet-based model. In contrast to prior work, the attacks are targeted to specifically alter diagnostically relevant regions. Using two realistic attack models (adversarial $k$-space noise and adversarial rotations) we are able to show that current state-of-the-art DL-based reconstruction algorithms are indeed sensitive to such perturbations to a degree where relevant diagnostic information may be lost. Surprisingly, in our experiments the UNet and the more sophisticated E2E-VarNet were similarly sensitive to such attacks. Our findings add further to the evidence that caution must be exercised as DL-based methods move closer to clinical practice.



## **26. A Systematic Survey of Attack Detection and Prevention in Connected and Autonomous Vehicles**

cs.CR

This article is published in the Vehicular Communications journal

**SubmitDate**: 2022-08-05    [paper-pdf](http://arxiv.org/pdf/2203.14965v2)

**Authors**: Trupil Limbasiya, Ko Zheng Teng, Sudipta Chattopadhyay, Jianying Zhou

**Abstracts**: The number of Connected and Autonomous Vehicles (CAVs) is increasing rapidly in various smart transportation services and applications, considering many benefits to society, people, and the environment. Several research surveys for CAVs were conducted by primarily focusing on various security threats and vulnerabilities in the domain of CAVs to classify different types of attacks, impacts of attacks, attack features, cyber-risk, defense methodologies against attacks, and safety standards. However, the importance of attack detection and prevention approaches for CAVs has not been discussed extensively in the state-of-the-art surveys, and there is a clear gap in the existing literature on such methodologies to detect new and conventional threats and protect the CAV systems from unexpected hazards on the road. Some surveys have a limited discussion on Attacks Detection and Prevention Systems (ADPS), but such surveys provide only partial coverage of different types of ADPS for CAVs. Furthermore, there is a scope for discussing security, privacy, and efficiency challenges in ADPS that can give an overview of important security and performance attributes.   This survey paper, therefore, presents the significance of CAVs in the market, potential challenges in CAVs, key requirements of essential security and privacy properties, various capabilities of adversaries, possible attacks in CAVs, and performance evaluation parameters for ADPS. An extensive analysis is discussed of different ADPS categories for CAVs and state-of-the-art research works based on each ADPS category that gives the latest findings in this research domain. This survey also discusses crucial and open security research problems that are required to be focused on the secure deployment of CAVs in the market.



## **27. Differentially Private Counterfactuals via Functional Mechanism**

cs.LG

**SubmitDate**: 2022-08-04    [paper-pdf](http://arxiv.org/pdf/2208.02878v1)

**Authors**: Fan Yang, Qizhang Feng, Kaixiong Zhou, Jiahao Chen, Xia Hu

**Abstracts**: Counterfactual, serving as one emerging type of model explanation, has attracted tons of attentions recently from both industry and academia. Different from the conventional feature-based explanations (e.g., attributions), counterfactuals are a series of hypothetical samples which can flip model decisions with minimal perturbations on queries. Given valid counterfactuals, humans are capable of reasoning under ``what-if'' circumstances, so as to better understand the model decision boundaries. However, releasing counterfactuals could be detrimental, since it may unintentionally leak sensitive information to adversaries, which brings about higher risks on both model security and data privacy. To bridge the gap, in this paper, we propose a novel framework to generate differentially private counterfactual (DPC) without touching the deployed model or explanation set, where noises are injected for protection while maintaining the explanation roles of counterfactual. In particular, we train an autoencoder with the functional mechanism to construct noisy class prototypes, and then derive the DPC from the latent prototypes based on the post-processing immunity of differential privacy. Further evaluations demonstrate the effectiveness of the proposed framework, showing that DPC can successfully relieve the risks on both extraction and inference attacks.



## **28. Self-Ensembling Vision Transformer (SEViT) for Robust Medical Image Classification**

cs.CV

**SubmitDate**: 2022-08-04    [paper-pdf](http://arxiv.org/pdf/2208.02851v1)

**Authors**: Faris Almalik, Mohammad Yaqub, Karthik Nandakumar

**Abstracts**: Vision Transformers (ViT) are competing to replace Convolutional Neural Networks (CNN) for various computer vision tasks in medical imaging such as classification and segmentation. While the vulnerability of CNNs to adversarial attacks is a well-known problem, recent works have shown that ViTs are also susceptible to such attacks and suffer significant performance degradation under attack. The vulnerability of ViTs to carefully engineered adversarial samples raises serious concerns about their safety in clinical settings. In this paper, we propose a novel self-ensembling method to enhance the robustness of ViT in the presence of adversarial attacks. The proposed Self-Ensembling Vision Transformer (SEViT) leverages the fact that feature representations learned by initial blocks of a ViT are relatively unaffected by adversarial perturbations. Learning multiple classifiers based on these intermediate feature representations and combining these predictions with that of the final ViT classifier can provide robustness against adversarial attacks. Measuring the consistency between the various predictions can also help detect adversarial samples. Experiments on two modalities (chest X-ray and fundoscopy) demonstrate the efficacy of SEViT architecture to defend against various adversarial attacks in the gray-box (attacker has full knowledge of the target model, but not the defense mechanism) setting. Code: https://github.com/faresmalik/SEViT



## **29. Adversarial Attacks on Image Generation With Made-Up Words**

cs.CV

**SubmitDate**: 2022-08-04    [paper-pdf](http://arxiv.org/pdf/2208.04135v1)

**Authors**: Raphaël Millière

**Abstracts**: Text-guided image generation models can be prompted to generate images using nonce words adversarially designed to robustly evoke specific visual concepts. Two approaches for such generation are introduced: macaronic prompting, which involves designing cryptic hybrid words by concatenating subword units from different languages; and evocative prompting, which involves designing nonce words whose broad morphological features are similar enough to that of existing words to trigger robust visual associations. The two methods can also be combined to generate images associated with more specific visual concepts. The implications of these techniques for the circumvention of existing approaches to content moderation, and particularly the generation of offensive or harmful images, are discussed.



## **30. Mass Exit Attacks on the Lightning Network**

cs.CR

**SubmitDate**: 2022-08-04    [paper-pdf](http://arxiv.org/pdf/2208.01908v2)

**Authors**: Cosimo Sguanci, Anastasios Sidiropoulos

**Abstracts**: The Lightning Network (LN) has enjoyed rapid growth over recent years, and has become the most popular scaling solution for the Bitcoin blockchain. The security of the LN hinges on the ability of the nodes to close a channel by settling their balances, which requires confirming a transaction on the Bitcoin blockchain within a pre-agreed time period. This inherent timing restriction that the LN must satisfy, make it susceptible to attacks that seek to increase the congestion on the Bitcoin blockchain, thus preventing correct protocol execution. We study the susceptibility of the LN to \emph{mass exit} attacks, in the presence of a small coalition of adversarial nodes. This is a scenario where an adversary forces a large set of honest protocol participants to interact with the blockchain. We focus on two types of attacks: (i) The first is a \emph{zombie} attack, where a set of $k$ nodes become unresponsive with the goal to lock the funds of many channels for a period of time longer than what the LN protocol dictates. (ii) The second is a \emph{mass double-spend} attack, where a set of $k$ nodes attempt to steal funds by submitting many closing transactions that settle channels using expired protocol states; this causes many honest nodes to have to quickly respond by submitting invalidating transactions. We show via simulations that, under historically-plausible congestion conditions, with mild statistical assumptions on channel balances, both of the attacks can be performed by a very small coalition. To perform our simulations, we formulate the problem of finding a worst-case coalition of $k$ adversarial nodes as a graph cut problem. Our experimental findings are supported by a theoretical justification based on the scale-free topology of the LN.



## **31. Design Considerations and Architecture for a Resilient Risk based Adaptive Authentication and Authorization (RAD-AA) Framework**

cs.CR

**SubmitDate**: 2022-08-04    [paper-pdf](http://arxiv.org/pdf/2208.02592v1)

**Authors**: Jaimandeep Singh, Chintan Patel, Naveen Kumar Chaudhary

**Abstracts**: A strong cyber attack is capable of degrading the performance of any Information Technology (IT) or Operational Technology (OT) system. In recent cyber attacks, credential theft emerged as one of the primary vectors of gaining entry into the system. Once, an attacker has a foothold in the system, they use token manipulation techniques to elevate the privileges and access protected resources. This makes authentication and authorization a critical component for a secure and resilient cyber system. In this paper we consider the design considerations for such a secure and resilient authentication and authorization framework capable of self-adapting based on the risk scores and trust profiles. We compare this design with the existing standards such as OAuth 2.0, OpenID Connect and SAML 2.0. We then study popular threat models such as STRIDE and PASTA and summarize the resilience of the proposed architecture against common and relevant threat vectors. We call this framework Resilient Risk-based Adaptive Authentication and Authorization (RAD-AA). The proposed framework excessively increases the cost for an adversary to launch any cyber attack and provides much-needed strength to critical infrastructure.



## **32. Prompt Tuning for Generative Multimodal Pretrained Models**

cs.CL

Work in progress

**SubmitDate**: 2022-08-04    [paper-pdf](http://arxiv.org/pdf/2208.02532v1)

**Authors**: Hao Yang, Junyang Lin, An Yang, Peng Wang, Chang Zhou, Hongxia Yang

**Abstracts**: Prompt tuning has become a new paradigm for model tuning and it has demonstrated success in natural language pretraining and even vision pretraining. In this work, we explore the transfer of prompt tuning to multimodal pretraining, with a focus on generative multimodal pretrained models, instead of contrastive ones. Specifically, we implement prompt tuning on the unified sequence-to-sequence pretrained model adaptive to both understanding and generation tasks. Experimental results demonstrate that the light-weight prompt tuning can achieve comparable performance with finetuning and surpass other light-weight tuning methods. Besides, in comparison with finetuned models, the prompt-tuned models demonstrate improved robustness against adversarial attacks. We further figure out that experimental factors, including the prompt length, prompt depth, and reparameteratization, have great impacts on the model performance, and thus we empirically provide a recommendation for the setups of prompt tuning. Despite the observed advantages, we still find some limitations in prompt tuning, and we correspondingly point out the directions for future studies. Codes are available at \url{https://github.com/OFA-Sys/OFA}



## **33. NoiLIn: Improving Adversarial Training and Correcting Stereotype of Noisy Labels**

cs.LG

Accepted at Transactions on Machine Learning Research (TMLR) at June  2022

**SubmitDate**: 2022-08-04    [paper-pdf](http://arxiv.org/pdf/2105.14676v2)

**Authors**: Jingfeng Zhang, Xilie Xu, Bo Han, Tongliang Liu, Gang Niu, Lizhen Cui, Masashi Sugiyama

**Abstracts**: Adversarial training (AT) formulated as the minimax optimization problem can effectively enhance the model's robustness against adversarial attacks. The existing AT methods mainly focused on manipulating the inner maximization for generating quality adversarial variants or manipulating the outer minimization for designing effective learning objectives. However, empirical results of AT always exhibit the robustness at odds with accuracy and the existence of the cross-over mixture problem, which motivates us to study some label randomness for benefiting the AT. First, we thoroughly investigate noisy labels (NLs) injection into AT's inner maximization and outer minimization, respectively and obtain the observations on when NL injection benefits AT. Second, based on the observations, we propose a simple but effective method -- NoiLIn that randomly injects NLs into training data at each training epoch and dynamically increases the NL injection rate once robust overfitting occurs. Empirically, NoiLIn can significantly mitigate the AT's undesirable issue of robust overfitting and even further improve the generalization of the state-of-the-art AT methods. Philosophically, NoiLIn sheds light on a new perspective of learning with NLs: NLs should not always be deemed detrimental, and even in the absence of NLs in the training set, we may consider injecting them deliberately. Codes are available in https://github.com/zjfheart/NoiLIn.



## **34. A Robust graph attention network with dynamic adjusted Graph**

cs.LG

21 pages,13 figures

**SubmitDate**: 2022-08-04    [paper-pdf](http://arxiv.org/pdf/2009.13038v3)

**Authors**: Xianchen Zhou, Yaoyun Zeng, Hongxia Wang

**Abstracts**: Graph Attention Networks(GATs) are useful deep learning models to deal with the graph data. However, recent works show that the classical GAT is vulnerable to adversarial attacks. It degrades dramatically with slight perturbations. Therefore, how to enhance the robustness of GAT is a critical problem. Robust GAT(RoGAT) is proposed in this paper to improve the robustness of GAT based on the revision of the attention mechanism. Different from the original GAT, which uses the attention mechanism for different edges but is still sensitive to the perturbation, RoGAT adds an extra dynamic attention score progressively and improves the robustness. Firstly, RoGAT revises the edges weight based on the smoothness assumption which is quite common for ordinary graphs. Secondly, RoGAT further revises the features to suppress features' noise. Then, an extra attention score is generated by the dynamic edge's weight and can be used to reduce the impact of adversarial attacks. Different experiments against targeted and untargeted attacks on citation data on citation data demonstrate that RoGAT outperforms most of the recent defensive methods.



## **35. Privacy Safe Representation Learning via Frequency Filtering Encoder**

cs.CV

The IJCAI-ECAI-22 Workshop on Artificial Intelligence Safety  (AISafety 2022)

**SubmitDate**: 2022-08-04    [paper-pdf](http://arxiv.org/pdf/2208.02482v1)

**Authors**: Jonghu Jeong, Minyong Cho, Philipp Benz, Jinwoo Hwang, Jeewook Kim, Seungkwan Lee, Tae-hoon Kim

**Abstracts**: Deep learning models are increasingly deployed in real-world applications. These models are often deployed on the server-side and receive user data in an information-rich representation to solve a specific task, such as image classification. Since images can contain sensitive information, which users might not be willing to share, privacy protection becomes increasingly important. Adversarial Representation Learning (ARL) is a common approach to train an encoder that runs on the client-side and obfuscates an image. It is assumed, that the obfuscated image can safely be transmitted and used for the task on the server without privacy concerns. However, in this work, we find that training a reconstruction attacker can successfully recover the original image of existing ARL methods. To this end, we introduce a novel ARL method enhanced through low-pass filtering, limiting the available information amount to be encoded in the frequency domain. Our experimental results reveal that our approach withstands reconstruction attacks while outperforming previous state-of-the-art methods regarding the privacy-utility trade-off. We further conduct a user study to qualitatively assess our defense of the reconstruction attack.



## **36. Node Copying: A Random Graph Model for Effective Graph Sampling**

stat.ML

**SubmitDate**: 2022-08-04    [paper-pdf](http://arxiv.org/pdf/2208.02435v1)

**Authors**: Florence Regol, Soumyasundar Pal, Jianing Sun, Yingxue Zhang, Yanhui Geng, Mark Coates

**Abstracts**: There has been an increased interest in applying machine learning techniques on relational structured-data based on an observed graph. Often, this graph is not fully representative of the true relationship amongst nodes. In these settings, building a generative model conditioned on the observed graph allows to take the graph uncertainty into account. Various existing techniques either rely on restrictive assumptions, fail to preserve topological properties within the samples or are prohibitively expensive for larger graphs. In this work, we introduce the node copying model for constructing a distribution over graphs. Sampling of a random graph is carried out by replacing each node's neighbors by those of a randomly sampled similar node. The sampled graphs preserve key characteristics of the graph structure without explicitly targeting them. Additionally, sampling from this model is extremely simple and scales linearly with the nodes. We show the usefulness of the copying model in three tasks. First, in node classification, a Bayesian formulation based on node copying achieves higher accuracy in sparse data settings. Second, we employ our proposed model to mitigate the effect of adversarial attacks on the graph topology. Last, incorporation of the model in a recommendation system setting improves recall over state-of-the-art methods.



## **37. Is current research on adversarial robustness addressing the right problem?**

cs.CV

**SubmitDate**: 2022-08-04    [paper-pdf](http://arxiv.org/pdf/2208.00539v2)

**Authors**: Ali Borji

**Abstracts**: Short answer: Yes, Long answer: No! Indeed, research on adversarial robustness has led to invaluable insights helping us understand and explore different aspects of the problem. Many attacks and defenses have been proposed over the last couple of years. The problem, however, remains largely unsolved and poorly understood. Here, I argue that the current formulation of the problem serves short term goals, and needs to be revised for us to achieve bigger gains. Specifically, the bound on perturbation has created a somewhat contrived setting and needs to be relaxed. This has misled us to focus on model classes that are not expressive enough to begin with. Instead, inspired by human vision and the fact that we rely more on robust features such as shape, vertices, and foreground objects than non-robust features such as texture, efforts should be steered towards looking for significantly different classes of models. Maybe instead of narrowing down on imperceptible adversarial perturbations, we should attack a more general problem which is finding architectures that are simultaneously robust to perceptible perturbations, geometric transformations (e.g. rotation, scaling), image distortions (lighting, blur), and more (e.g. occlusion, shadow). Only then we may be able to solve the problem of adversarial vulnerability.



## **38. A New Kind of Adversarial Example**

cs.CV

**SubmitDate**: 2022-08-04    [paper-pdf](http://arxiv.org/pdf/2208.02430v1)

**Authors**: Ali Borji

**Abstracts**: Almost all adversarial attacks are formulated to add an imperceptible perturbation to an image in order to fool a model. Here, we consider the opposite which is adversarial examples that can fool a human but not a model. A large enough and perceptible perturbation is added to an image such that a model maintains its original decision, whereas a human will most likely make a mistake if forced to decide (or opt not to decide at all). Existing targeted attacks can be reformulated to synthesize such adversarial examples. Our proposed attack, dubbed NKE, is similar in essence to the fooling images, but is more efficient since it uses gradient descent instead of evolutionary algorithms. It also offers a new and unified perspective into the problem of adversarial vulnerability. Experimental results over MNIST and CIFAR-10 datasets show that our attack is quite efficient in fooling deep neural networks. Code is available at https://github.com/aliborji/NKE.



## **39. MOVE: Effective and Harmless Ownership Verification via Embedded External Features**

cs.CR

15 pages. The journal extension of our conference paper in AAAI 2022  (https://ojs.aaai.org/index.php/AAAI/article/view/20036). arXiv admin note:  substantial text overlap with arXiv:2112.03476

**SubmitDate**: 2022-08-04    [paper-pdf](http://arxiv.org/pdf/2208.02820v1)

**Authors**: Yiming Li, Linghui Zhu, Xiaojun Jia, Yang Bai, Yong Jiang, Shu-Tao Xia, Xiaochun Cao

**Abstracts**: Currently, deep neural networks (DNNs) are widely adopted in different applications. Despite its commercial values, training a well-performed DNN is resource-consuming. Accordingly, the well-trained model is valuable intellectual property for its owner. However, recent studies revealed the threats of model stealing, where the adversaries can obtain a function-similar copy of the victim model, even when they can only query the model. In this paper, we propose an effective and harmless model ownership verification (MOVE) to defend against different types of model stealing simultaneously, without introducing new security risks. In general, we conduct the ownership verification by verifying whether a suspicious model contains the knowledge of defender-specified external features. Specifically, we embed the external features by tempering a few training samples with style transfer. We then train a meta-classifier to determine whether a model is stolen from the victim. This approach is inspired by the understanding that the stolen models should contain the knowledge of features learned by the victim model. In particular, we develop our MOVE method under both white-box and black-box settings to provide comprehensive model protection. Extensive experiments on benchmark datasets verify the effectiveness of our method and its resistance to potential adaptive attacks. The codes for reproducing the main experiments of our method are available at \url{https://github.com/THUYimingLi/MOVE}.



## **40. Deep VULMAN: A Deep Reinforcement Learning-Enabled Cyber Vulnerability Management Framework**

cs.AI

12 pages, 3 figures

**SubmitDate**: 2022-08-03    [paper-pdf](http://arxiv.org/pdf/2208.02369v1)

**Authors**: Soumyadeep Hore, Ankit Shah, Nathaniel D. Bastian

**Abstracts**: Cyber vulnerability management is a critical function of a cybersecurity operations center (CSOC) that helps protect organizations against cyber-attacks on their computer and network systems. Adversaries hold an asymmetric advantage over the CSOC, as the number of deficiencies in these systems is increasing at a significantly higher rate compared to the expansion rate of the security teams to mitigate them in a resource-constrained environment. The current approaches are deterministic and one-time decision-making methods, which do not consider future uncertainties when prioritizing and selecting vulnerabilities for mitigation. These approaches are also constrained by the sub-optimal distribution of resources, providing no flexibility to adjust their response to fluctuations in vulnerability arrivals. We propose a novel framework, Deep VULMAN, consisting of a deep reinforcement learning agent and an integer programming method to fill this gap in the cyber vulnerability management process. Our sequential decision-making framework, first, determines the near-optimal amount of resources to be allocated for mitigation under uncertainty for a given system state and then determines the optimal set of prioritized vulnerability instances for mitigation. Our proposed framework outperforms the current methods in prioritizing the selection of important organization-specific vulnerabilities, on both simulated and real-world vulnerability data, observed over a one-year period.



## **41. Membership Inference Attacks and Defenses in Neural Network Pruning**

cs.CR

This paper has been accepted to USENIX Security Symposium 2022. This  is an extended version with more experimental results

**SubmitDate**: 2022-08-03    [paper-pdf](http://arxiv.org/pdf/2202.03335v2)

**Authors**: Xiaoyong Yuan, Lan Zhang

**Abstracts**: Neural network pruning has been an essential technique to reduce the computation and memory requirements for using deep neural networks for resource-constrained devices. Most existing research focuses primarily on balancing the sparsity and accuracy of a pruned neural network by strategically removing insignificant parameters and retraining the pruned model. Such efforts on reusing training samples pose serious privacy risks due to increased memorization, which, however, has not been investigated yet.   In this paper, we conduct the first analysis of privacy risks in neural network pruning. Specifically, we investigate the impacts of neural network pruning on training data privacy, i.e., membership inference attacks. We first explore the impact of neural network pruning on prediction divergence, where the pruning process disproportionately affects the pruned model's behavior for members and non-members. Meanwhile, the influence of divergence even varies among different classes in a fine-grained manner. Enlighten by such divergence, we proposed a self-attention membership inference attack against the pruned neural networks. Extensive experiments are conducted to rigorously evaluate the privacy impacts of different pruning approaches, sparsity levels, and adversary knowledge. The proposed attack shows the higher attack performance on the pruned models when compared with eight existing membership inference attacks. In addition, we propose a new defense mechanism to protect the pruning process by mitigating the prediction divergence based on KL-divergence distance, whose effectiveness has been experimentally demonstrated to effectively mitigate the privacy risks while maintaining the sparsity and accuracy of the pruned models.



## **42. Design of secure and robust cognitive system for malware detection**

cs.CR

arXiv admin note: substantial text overlap with arXiv:2104.06652

**SubmitDate**: 2022-08-03    [paper-pdf](http://arxiv.org/pdf/2208.02310v1)

**Authors**: Sanket Shukla

**Abstracts**: Machine learning based malware detection techniques rely on grayscale images of malware and tends to classify malware based on the distribution of textures in graycale images. Albeit the advancement and promising results shown by machine learning techniques, attackers can exploit the vulnerabilities by generating adversarial samples. Adversarial samples are generated by intelligently crafting and adding perturbations to the input samples. There exists majority of the software based adversarial attacks and defenses. To defend against the adversaries, the existing malware detection based on machine learning and grayscale images needs a preprocessing for the adversarial data. This can cause an additional overhead and can prolong the real-time malware detection. So, as an alternative to this, we explore RRAM (Resistive Random Access Memory) based defense against adversaries. Therefore, the aim of this thesis is to address the above mentioned critical system security issues. The above mentioned challenges are addressed by demonstrating proposed techniques to design a secure and robust cognitive system. First, a novel technique to detect stealthy malware is proposed. The technique uses malware binary images and then extract different features from the same and then employ different ML-classifiers on the dataset thus obtained. Results demonstrate that this technique is successful in differentiating classes of malware based on the features extracted. Secondly, I demonstrate the effects of adversarial attacks on a reconfigurable RRAM-neuromorphic architecture with different learning algorithms and device characteristics. I also propose an integrated solution for mitigating the effects of the adversarial attack using the reconfigurable RRAM architecture.



## **43. Generating Image Adversarial Examples by Embedding Digital Watermarks**

cs.CV

10 pages, 4 figures

**SubmitDate**: 2022-08-03    [paper-pdf](http://arxiv.org/pdf/2009.05107v2)

**Authors**: Yuexin Xiang, Tiantian Li, Wei Ren, Tianqing Zhu, Kim-Kwang Raymond Choo

**Abstracts**: With the increasing attention to deep neural network (DNN) models, attacks are also upcoming for such models. For example, an attacker may carefully construct images in specific ways (also referred to as adversarial examples) aiming to mislead the DNN models to output incorrect classification results. Similarly, many efforts are proposed to detect and mitigate adversarial examples, usually for certain dedicated attacks. In this paper, we propose a novel digital watermark-based method to generate image adversarial examples to fool DNN models. Specifically, partial main features of the watermark image are embedded into the host image almost invisibly, aiming to tamper with and damage the recognition capabilities of the DNN models. We devise an efficient mechanism to select host images and watermark images and utilize the improved discrete wavelet transform (DWT) based Patchwork watermarking algorithm with a set of valid hyperparameters to embed digital watermarks from the watermark image dataset into original images for generating image adversarial examples. The experimental results illustrate that the attack success rate on common DNN models can reach an average of 95.47% on the CIFAR-10 dataset and the highest at 98.71%. Besides, our scheme is able to generate a large number of adversarial examples efficiently, concretely, an average of 1.17 seconds for completing the attacks on each image on the CIFAR-10 dataset. In addition, we design a baseline experiment using the watermark images generated by Gaussian noise as the watermark image dataset that also displays the effectiveness of our scheme. Similarly, we also propose the modified discrete cosine transform (DCT) based Patchwork watermarking algorithm. To ensure repeatability and reproducibility, the source code is available on GitHub.



## **44. Abusing Commodity DRAMs in IoT Devices to Remotely Spy on Temperature**

cs.CR

Submitted to IEEE TIFS and currently under review

**SubmitDate**: 2022-08-03    [paper-pdf](http://arxiv.org/pdf/2208.02125v1)

**Authors**: Florian Frank, Wenjie Xiong, Nikolaos Athanasios Anagnostopoulos, André Schaller, Tolga Arul, Farinaz Koushanfar, Stefan Katzenbeisser, Ulrich Ruhrmair, Jakub Szefer

**Abstracts**: The ubiquity and pervasiveness of modern Internet of Things (IoT) devices opens up vast possibilities for novel applications, but simultaneously also allows spying on, and collecting data from, unsuspecting users to a previously unseen extent. This paper details a new attack form in this vein, in which the decay properties of widespread, off-the-shelf DRAM modules are exploited to accurately sense the temperature in the vicinity of the DRAM-carrying device. Among others, this enables adversaries to remotely and purely digitally spy on personal behavior in users' private homes, or to collect security-critical data in server farms, cloud storage centers, or commercial production lines. We demonstrate that our attack can be performed by merely compromising the software of an IoT device and does not require hardware modifications or physical access at attack time. It can achieve temperature resolutions of up to 0.5{\deg}C over a range of 0{\deg}C to 70{\deg}C in practice. Perhaps most interestingly, it even works in devices that do not have a dedicated temperature sensor on board. To complete our work, we discuss practical attack scenarios as well as possible countermeasures against our temperature espionage attacks.



## **45. Local Differential Privacy for Federated Learning**

cs.CR

17 pages

**SubmitDate**: 2022-08-03    [paper-pdf](http://arxiv.org/pdf/2202.06053v2)

**Authors**: M. A. P. Chamikara, Dongxi Liu, Seyit Camtepe, Surya Nepal, Marthie Grobler, Peter Bertok, Ibrahim Khalil

**Abstracts**: Advanced adversarial attacks such as membership inference and model memorization can make federated learning (FL) vulnerable and potentially leak sensitive private data. Local differentially private (LDP) approaches are gaining more popularity due to stronger privacy notions and native support for data distribution compared to other differentially private (DP) solutions. However, DP approaches assume that the FL server (that aggregates the models) is honest (run the FL protocol honestly) or semi-honest (run the FL protocol honestly while also trying to learn as much information as possible). These assumptions make such approaches unrealistic and unreliable for real-world settings. Besides, in real-world industrial environments (e.g., healthcare), the distributed entities (e.g., hospitals) are already composed of locally running machine learning models (this setting is also referred to as the cross-silo setting). Existing approaches do not provide a scalable mechanism for privacy-preserving FL to be utilized under such settings, potentially with untrusted parties. This paper proposes a new local differentially private FL (named LDPFL) protocol for industrial settings. LDPFL can run in industrial settings with untrusted entities while enforcing stronger privacy guarantees than existing approaches. LDPFL shows high FL model performance (up to 98%) under small privacy budgets (e.g., epsilon = 0.5) in comparison to existing methods.



## **46. SAC-AP: Soft Actor Critic based Deep Reinforcement Learning for Alert Prioritization**

cs.CR

8 pages, 8 figures, IEEE WORLD CONGRESS ON COMPUTATIONAL INTELLIGENCE  2022

**SubmitDate**: 2022-08-03    [paper-pdf](http://arxiv.org/pdf/2207.13666v3)

**Authors**: Lalitha Chavali, Tanay Gupta, Paresh Saxena

**Abstracts**: Intrusion detection systems (IDS) generate a large number of false alerts which makes it difficult to inspect true positives. Hence, alert prioritization plays a crucial role in deciding which alerts to investigate from an enormous number of alerts that are generated by IDS. Recently, deep reinforcement learning (DRL) based deep deterministic policy gradient (DDPG) off-policy method has shown to achieve better results for alert prioritization as compared to other state-of-the-art methods. However, DDPG is prone to the problem of overfitting. Additionally, it also has a poor exploration capability and hence it is not suitable for problems with a stochastic environment. To address these limitations, we present a soft actor-critic based DRL algorithm for alert prioritization (SAC-AP), an off-policy method, based on the maximum entropy reinforcement learning framework that aims to maximize the expected reward while also maximizing the entropy. Further, the interaction between an adversary and a defender is modeled as a zero-sum game and a double oracle framework is utilized to obtain the approximate mixed strategy Nash equilibrium (MSNE). SAC-AP finds robust alert investigation policies and computes pure strategy best response against opponent's mixed strategy. We present the overall design of SAC-AP and evaluate its performance as compared to other state-of-the art alert prioritization methods. We consider defender's loss, i.e., the defender's inability to investigate the alerts that are triggered due to attacks, as the performance metric. Our results show that SAC-AP achieves up to 30% decrease in defender's loss as compared to the DDPG based alert prioritization method and hence provides better protection against intrusions. Moreover, the benefits are even higher when SAC-AP is compared to other traditional alert prioritization methods including Uniform, GAIN, RIO and Suricata.



## **47. Spectrum Focused Frequency Adversarial Attacks for Automatic Modulation Classification**

cs.CR

6 pages, 9 figures

**SubmitDate**: 2022-08-03    [paper-pdf](http://arxiv.org/pdf/2208.01919v1)

**Authors**: Sicheng Zhang, Jiarun Yu, Zhida Bao, Shiwen Mao, Yun Lin

**Abstracts**: Artificial intelligence (AI) technology has provided a potential solution for automatic modulation recognition (AMC). Unfortunately, AI-based AMC models are vulnerable to adversarial examples, which seriously threatens the efficient, secure and trusted application of AI in AMC. This issue has attracted the attention of researchers. Various studies on adversarial attacks and defenses evolve in a spiral. However, the existing adversarial attack methods are all designed in the time domain. They introduce more high-frequency components in the frequency domain, due to abrupt updates in the time domain. For this issue, from the perspective of frequency domain, we propose a spectrum focused frequency adversarial attacks (SFFAA) for AMC model, and further draw on the idea of meta-learning, propose a Meta-SFFAA algorithm to improve the transferability in the black-box attacks. Extensive experiments, qualitative and quantitative metrics demonstrate that the proposed algorithm can concentrate the adversarial energy on the spectrum where the signal is located, significantly improve the adversarial attack performance while maintaining the concealment in the frequency domain.



## **48. On the Evaluation of User Privacy in Deep Neural Networks using Timing Side Channel**

cs.CR

15 pages, 20 figures

**SubmitDate**: 2022-08-03    [paper-pdf](http://arxiv.org/pdf/2208.01113v2)

**Authors**: Shubhi Shukla, Manaar Alam, Sarani Bhattacharya, Debdeep Mukhopadhyay, Pabitra Mitra

**Abstracts**: Recent Deep Learning (DL) advancements in solving complex real-world tasks have led to its widespread adoption in practical applications. However, this opportunity comes with significant underlying risks, as many of these models rely on privacy-sensitive data for training in a variety of applications, making them an overly-exposed threat surface for privacy violations. Furthermore, the widespread use of cloud-based Machine-Learning-as-a-Service (MLaaS) for its robust infrastructure support has broadened the threat surface to include a variety of remote side-channel attacks. In this paper, we first identify and report a novel data-dependent timing side-channel leakage (termed Class Leakage) in DL implementations originating from non-constant time branching operation in a widely used DL framework PyTorch. We further demonstrate a practical inference-time attack where an adversary with user privilege and hard-label black-box access to an MLaaS can exploit Class Leakage to compromise the privacy of MLaaS users. DL models are vulnerable to Membership Inference Attack (MIA), where an adversary's objective is to deduce whether any particular data has been used while training the model. In this paper, as a separate case study, we demonstrate that a DL model secured with differential privacy (a popular countermeasure against MIA) is still vulnerable to MIA against an adversary exploiting Class Leakage. We develop an easy-to-implement countermeasure by making a constant-time branching operation that alleviates the Class Leakage and also aids in mitigating MIA. We have chosen two standard benchmarking image classification datasets, CIFAR-10 and CIFAR-100 to train five state-of-the-art pre-trained DL models, over two different computing environments having Intel Xeon and Intel i7 processors to validate our approach.



## **49. Adversarial Attacks on ASR Systems: An Overview**

cs.SD

**SubmitDate**: 2022-08-03    [paper-pdf](http://arxiv.org/pdf/2208.02250v1)

**Authors**: Xiao Zhang, Hao Tan, Xuan Huang, Denghui Zhang, Keke Tang, Zhaoquan Gu

**Abstracts**: With the development of hardware and algorithms, ASR(Automatic Speech Recognition) systems evolve a lot. As The models get simpler, the difficulty of development and deployment become easier, ASR systems are getting closer to our life. On the one hand, we often use APPs or APIs of ASR to generate subtitles and record meetings. On the other hand, smart speaker and self-driving car rely on ASR systems to control AIoT devices. In past few years, there are a lot of works on adversarial examples attacks against ASR systems. By adding a small perturbation to the waveforms, the recognition results make a big difference. In this paper, we describe the development of ASR system, different assumptions of attacks, and how to evaluate these attacks. Next, we introduce the current works on adversarial examples attacks from two attack assumptions: white-box attack and black-box attack. Different from other surveys, we pay more attention to which layer they perturb waveforms in ASR system, the relationship between these attacks, and their implementation methods. We focus on the effect of their works.



## **50. Robust Graph Neural Networks using Weighted Graph Laplacian**

cs.LG

Accepted at IEEE International Conference on Signal Processing and  Communications (SPCOM), 2022

**SubmitDate**: 2022-08-03    [paper-pdf](http://arxiv.org/pdf/2208.01853v1)

**Authors**: Bharat Runwal, Vivek, Sandeep Kumar

**Abstracts**: Graph neural network (GNN) is achieving remarkable performances in a variety of application domains. However, GNN is vulnerable to noise and adversarial attacks in input data. Making GNN robust against noises and adversarial attacks is an important problem. The existing defense methods for GNNs are computationally demanding and are not scalable. In this paper, we propose a generic framework for robustifying GNN known as Weighted Laplacian GNN (RWL-GNN). The method combines Weighted Graph Laplacian learning with the GNN implementation. The proposed method benefits from the positive semi-definiteness property of Laplacian matrix, feature smoothness, and latent features via formulating a unified optimization framework, which ensures the adversarial/noisy edges are discarded and connections in the graph are appropriately weighted. For demonstration, the experiments are conducted with Graph convolutional neural network(GCNN) architecture, however, the proposed framework is easily amenable to any existing GNN architecture. The simulation results with benchmark dataset establish the efficacy of the proposed method, both in accuracy and computational efficiency. Code can be accessed at https://github.com/Bharat-Runwal/RWL-GNN.



