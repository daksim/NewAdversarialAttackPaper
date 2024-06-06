# Latest Adversarial Attack Papers
**update at 2024-06-06 09:44:15**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. Can Implicit Bias Imply Adversarial Robustness?**

cs.LG

icml 2024 camera-ready

**SubmitDate**: 2024-06-05    [abs](http://arxiv.org/abs/2405.15942v2) [paper-pdf](http://arxiv.org/pdf/2405.15942v2)

**Authors**: Hancheng Min, René Vidal

**Abstract**: The implicit bias of gradient-based training algorithms has been considered mostly beneficial as it leads to trained networks that often generalize well. However, Frei et al. (2023) show that such implicit bias can harm adversarial robustness. Specifically, they show that if the data consists of clusters with small inter-cluster correlation, a shallow (two-layer) ReLU network trained by gradient flow generalizes well, but it is not robust to adversarial attacks of small radius. Moreover, this phenomenon occurs despite the existence of a much more robust classifier that can be explicitly constructed from a shallow network. In this paper, we extend recent analyses of neuron alignment to show that a shallow network with a polynomial ReLU activation (pReLU) trained by gradient flow not only generalizes well but is also robust to adversarial attacks. Our results highlight the importance of the interplay between data structure and architecture design in the implicit bias and robustness of trained networks.



## **2. Defending Large Language Models Against Attacks With Residual Stream Activation Analysis**

cs.CR

**SubmitDate**: 2024-06-05    [abs](http://arxiv.org/abs/2406.03230v1) [paper-pdf](http://arxiv.org/pdf/2406.03230v1)

**Authors**: Amelia Kawasaki, Andrew Davis, Houssam Abbas

**Abstract**: The widespread adoption of Large Language Models (LLMs), exemplified by OpenAI's ChatGPT, brings to the forefront the imperative to defend against adversarial threats on these models. These attacks, which manipulate an LLM's output by introducing malicious inputs, undermine the model's integrity and the trust users place in its outputs. In response to this challenge, our paper presents an innovative defensive strategy, given white box access to an LLM, that harnesses residual activation analysis between transformer layers of the LLM. We apply an established methodology for analyzing distinctive activation patterns in the residual streams for a novel result of attack prompt classification. We curate multiple datasets to demonstrate how this method of classification has high accuracy across multiple types of attack scenarios, including our newly-created attack dataset. Furthermore, we enhance the model's resilience by integrating safety fine-tuning techniques for LLMs in order to measure its effect on our capability to detect attacks. The results underscore the effectiveness of our approach in enhancing the detection and mitigation of adversarial inputs, advancing the security framework within which LLMs operate.



## **3. Graph Neural Network Explanations are Fragile**

cs.CR

17 pages, 64 figures

**SubmitDate**: 2024-06-05    [abs](http://arxiv.org/abs/2406.03193v1) [paper-pdf](http://arxiv.org/pdf/2406.03193v1)

**Authors**: Jiate Li, Meng Pang, Yun Dong, Jinyuan Jia, Binghui Wang

**Abstract**: Explainable Graph Neural Network (GNN) has emerged recently to foster the trust of using GNNs. Existing GNN explainers are developed from various perspectives to enhance the explanation performance. We take the first step to study GNN explainers under adversarial attack--We found that an adversary slightly perturbing graph structure can ensure GNN model makes correct predictions, but the GNN explainer yields a drastically different explanation on the perturbed graph. Specifically, we first formulate the attack problem under a practical threat model (i.e., the adversary has limited knowledge about the GNN explainer and a restricted perturbation budget). We then design two methods (i.e., one is loss-based and the other is deduction-based) to realize the attack. We evaluate our attacks on various GNN explainers and the results show these explainers are fragile.



## **4. Reconstructing training data from document understanding models**

cs.CR

**SubmitDate**: 2024-06-05    [abs](http://arxiv.org/abs/2406.03182v1) [paper-pdf](http://arxiv.org/pdf/2406.03182v1)

**Authors**: Jérémie Dentan, Arnaud Paran, Aymen Shabou

**Abstract**: Document understanding models are increasingly employed by companies to supplant humans in processing sensitive documents, such as invoices, tax notices, or even ID cards. However, the robustness of such models to privacy attacks remains vastly unexplored. This paper presents CDMI, the first reconstruction attack designed to extract sensitive fields from the training data of these models. We attack LayoutLM and BROS architectures, demonstrating that an adversary can perfectly reconstruct up to 4.1% of the fields of the documents used for fine-tuning, including some names, dates, and invoice amounts up to six-digit numbers. When our reconstruction attack is combined with a membership inference attack, our attack accuracy escalates to 22.5%. In addition, we introduce two new end-to-end metrics and evaluate our approach under various conditions: unimodal or bimodal data, LayoutLM or BROS backbones, four fine-tuning tasks, and two public datasets (FUNSD and SROIE). We also investigate the interplay between overfitting, predictive performance, and susceptibility to our attack. We conclude with a discussion on possible defenses against our attack and potential future research directions to construct robust document understanding models.



## **5. ZeroPur: Succinct Training-Free Adversarial Purification**

cs.CV

16 pages, 5 figures, under review

**SubmitDate**: 2024-06-05    [abs](http://arxiv.org/abs/2406.03143v1) [paper-pdf](http://arxiv.org/pdf/2406.03143v1)

**Authors**: Xiuli Bi, Zonglin Yang, Bo Liu, Xiaodong Cun, Chi-Man Pun, Pietro Lio, Bin Xiao

**Abstract**: Adversarial purification is a kind of defense technique that can defend various unseen adversarial attacks without modifying the victim classifier. Existing methods often depend on external generative models or cooperation between auxiliary functions and victim classifiers. However, retraining generative models, auxiliary functions, or victim classifiers relies on the domain of the fine-tuned dataset and is computation-consuming. In this work, we suppose that adversarial images are outliers of the natural image manifold and the purification process can be considered as returning them to this manifold. Following this assumption, we present a simple adversarial purification method without further training to purify adversarial images, called ZeroPur. ZeroPur contains two steps: given an adversarial example, Guided Shift obtains the shifted embedding of the adversarial example by the guidance of its blurred counterparts; after that, Adaptive Projection constructs a directional vector by this shifted embedding to provide momentum, projecting adversarial images onto the manifold adaptively. ZeroPur is independent of external models and requires no retraining of victim classifiers or auxiliary functions, relying solely on victim classifiers themselves to achieve purification. Extensive experiments on three datasets (CIFAR-10, CIFAR-100, and ImageNet-1K) using various classifier architectures (ResNet, WideResNet) demonstrate that our method achieves state-of-the-art robust performance. The code will be publicly available.



## **6. VQUNet: Vector Quantization U-Net for Defending Adversarial Atacks by Regularizing Unwanted Noise**

cs.CV

8 pages, 6 figures

**SubmitDate**: 2024-06-05    [abs](http://arxiv.org/abs/2406.03117v1) [paper-pdf](http://arxiv.org/pdf/2406.03117v1)

**Authors**: Zhixun He, Mukesh Singhal

**Abstract**: Deep Neural Networks (DNN) have become a promising paradigm when developing Artificial Intelligence (AI) and Machine Learning (ML) applications. However, DNN applications are vulnerable to fake data that are crafted with adversarial attack algorithms. Under adversarial attacks, the prediction accuracy of DNN applications suffers, making them unreliable. In order to defend against adversarial attacks, we introduce a novel noise-reduction procedure, Vector Quantization U-Net (VQUNet), to reduce adversarial noise and reconstruct data with high fidelity. VQUNet features a discrete latent representation learning through a multi-scale hierarchical structure for both noise reduction and data reconstruction. The empirical experiments show that the proposed VQUNet provides better robustness to the target DNN models, and it outperforms other state-of-the-art noise-reduction-based defense methods under various adversarial attacks for both Fashion-MNIST and CIFAR10 datasets. When there is no adversarial attack, the defense method has less than 1% accuracy degradation for both datasets.



## **7. Revisiting the Trade-off between Accuracy and Robustness via Weight Distribution of Filters**

cs.CV

Accepted by TPAMI2024

**SubmitDate**: 2024-06-05    [abs](http://arxiv.org/abs/2306.03430v4) [paper-pdf](http://arxiv.org/pdf/2306.03430v4)

**Authors**: Xingxing Wei, Shiji Zhao, Bo li

**Abstract**: Adversarial attacks have been proven to be potential threats to Deep Neural Networks (DNNs), and many methods are proposed to defend against adversarial attacks. However, while enhancing the robustness, the clean accuracy will decline to a certain extent, implying a trade-off existed between the accuracy and robustness. In this paper, to meet the trade-off problem, we theoretically explore the underlying reason for the difference of the filters' weight distribution between standard-trained and robust-trained models and then argue that this is an intrinsic property for static neural networks, thus they are difficult to fundamentally improve the accuracy and adversarial robustness at the same time. Based on this analysis, we propose a sample-wise dynamic network architecture named Adversarial Weight-Varied Network (AW-Net), which focuses on dealing with clean and adversarial examples with a "divide and rule" weight strategy. The AW-Net adaptively adjusts the network's weights based on regulation signals generated by an adversarial router, which is directly influenced by the input sample. Benefiting from the dynamic network architecture, clean and adversarial examples can be processed with different network weights, which provides the potential to enhance both accuracy and adversarial robustness. A series of experiments demonstrate that our AW-Net is architecture-friendly to handle both clean and adversarial examples and can achieve better trade-off performance than state-of-the-art robust models.



## **8. Enhancing the Resilience of Graph Neural Networks to Topological Perturbations in Sparse Graphs**

cs.LG

**SubmitDate**: 2024-06-05    [abs](http://arxiv.org/abs/2406.03097v1) [paper-pdf](http://arxiv.org/pdf/2406.03097v1)

**Authors**: Shuqi He, Jun Zhuang, Ding Wang, Luyao Peng, Jun Song

**Abstract**: Graph neural networks (GNNs) have been extensively employed in node classification. Nevertheless, recent studies indicate that GNNs are vulnerable to topological perturbations, such as adversarial attacks and edge disruptions. Considerable efforts have been devoted to mitigating these challenges. For example, pioneering Bayesian methodologies, including GraphSS and LlnDT, incorporate Bayesian label transitions and topology-based label sampling to strengthen the robustness of GNNs. However, GraphSS is hindered by slow convergence, while LlnDT faces challenges in sparse graphs. To overcome these limitations, we propose a novel label inference framework, TraTopo, which combines topology-driven label propagation, Bayesian label transitions, and link analysis via random walks. TraTopo significantly surpasses its predecessors on sparse graphs by utilizing random walk sampling, specifically targeting isolated nodes for link prediction, thus enhancing its effectiveness in topological sampling contexts. Additionally, TraTopo employs a shortest-path strategy to refine link prediction, thereby reducing predictive overhead and improving label inference accuracy. Empirical evaluations highlight TraTopo's superiority in node classification, significantly exceeding contemporary GCN models in accuracy.



## **9. On the Duality Between Sharpness-Aware Minimization and Adversarial Training**

cs.LG

ICML 2024

**SubmitDate**: 2024-06-05    [abs](http://arxiv.org/abs/2402.15152v2) [paper-pdf](http://arxiv.org/pdf/2402.15152v2)

**Authors**: Yihao Zhang, Hangzhou He, Jingyu Zhu, Huanran Chen, Yifei Wang, Zeming Wei

**Abstract**: Adversarial Training (AT), which adversarially perturb the input samples during training, has been acknowledged as one of the most effective defenses against adversarial attacks, yet suffers from inevitably decreased clean accuracy. Instead of perturbing the samples, Sharpness-Aware Minimization (SAM) perturbs the model weights during training to find a more flat loss landscape and improve generalization. However, as SAM is designed for better clean accuracy, its effectiveness in enhancing adversarial robustness remains unexplored. In this work, considering the duality between SAM and AT, we investigate the adversarial robustness derived from SAM. Intriguingly, we find that using SAM alone can improve adversarial robustness. To understand this unexpected property of SAM, we first provide empirical and theoretical insights into how SAM can implicitly learn more robust features, and conduct comprehensive experiments to show that SAM can improve adversarial robustness notably without sacrificing any clean accuracy, shedding light on the potential of SAM to be a substitute for AT when accuracy comes at a higher priority. Code is available at https://github.com/weizeming/SAM_AT.



## **10. Are Your Models Still Fair? Fairness Attacks on Graph Neural Networks via Node Injections**

cs.LG

21 pages

**SubmitDate**: 2024-06-05    [abs](http://arxiv.org/abs/2406.03052v1) [paper-pdf](http://arxiv.org/pdf/2406.03052v1)

**Authors**: Zihan Luo, Hong Huang, Yongkang Zhou, Jiping Zhang, Nuo Chen

**Abstract**: Despite the remarkable capabilities demonstrated by Graph Neural Networks (GNNs) in graph-related tasks, recent research has revealed the fairness vulnerabilities in GNNs when facing malicious adversarial attacks. However, all existing fairness attacks require manipulating the connectivity between existing nodes, which may be prohibited in reality. To this end, we introduce a Node Injection-based Fairness Attack (NIFA), exploring the vulnerabilities of GNN fairness in such a more realistic setting. In detail, NIFA first designs two insightful principles for node injection operations, namely the uncertainty-maximization principle and homophily-increase principle, and then optimizes injected nodes' feature matrix to further ensure the effectiveness of fairness attacks. Comprehensive experiments on three real-world datasets consistently demonstrate that NIFA can significantly undermine the fairness of mainstream GNNs, even including fairness-aware GNNs, by injecting merely 1% of nodes. We sincerely hope that our work can stimulate increasing attention from researchers on the vulnerability of GNN fairness, and encourage the development of corresponding defense mechanisms.



## **11. SLIFER: Investigating Performance and Robustness of Malware Detection Pipelines**

cs.CR

**SubmitDate**: 2024-06-05    [abs](http://arxiv.org/abs/2405.14478v2) [paper-pdf](http://arxiv.org/pdf/2405.14478v2)

**Authors**: Andrea Ponte, Dmitrijs Trizna, Luca Demetrio, Battista Biggio, Ivan Tesfai Ogbu, Fabio Roli

**Abstract**: As a result of decades of research, Windows malware detection is approached through a plethora of techniques. However, there is an ongoing mismatch between academia -- which pursues an optimal performances in terms of detection rate and low false alarms -- and the requirements of real-world scenarios. In particular, academia focuses on combining static and dynamic analysis within a single or ensemble of models, falling into several pitfalls like (i) firing dynamic analysis without considering the computational burden it requires; (ii) discarding impossible-to-analyse samples; and (iii) analysing robustness against adversarial attacks without considering that malware detectors are complemented with more non-machine-learning components. Thus, in this paper we propose SLIFER, a novel Windows malware detection pipeline sequentially leveraging both static and dynamic analysis, interrupting computations as soon as one module triggers an alarm, requiring dynamic analysis only when needed. Contrary to the state of the art, we investigate how to deal with samples resistance to analysis, showing how much they impact performances, concluding that it is better to flag them as legitimate to not drastically increase false alarms. Lastly, we perform a robustness evaluation of SLIFER leveraging content-injections attacks, and we show that, counter-intuitively, attacks are blocked more by YARA rules than dynamic analysis due to byte artifacts created while optimizing the adversarial strategy.



## **12. DifAttack++: Query-Efficient Black-Box Adversarial Attack via Hierarchical Disentangled Feature Space in Cross Domain**

cs.CV

arXiv admin note: substantial text overlap with arXiv:2309.14585

**SubmitDate**: 2024-06-05    [abs](http://arxiv.org/abs/2406.03017v1) [paper-pdf](http://arxiv.org/pdf/2406.03017v1)

**Authors**: Jun Liu, Jiantao Zhou, Jiandian Zeng, Jinyu Tian

**Abstract**: This work investigates efficient score-based black-box adversarial attacks with a high Attack Success Rate (ASR) and good generalizability. We design a novel attack method based on a \textit{Hierarchical} \textbf{Di}sentangled \textbf{F}eature space and \textit{cross domain}, called \textbf{DifAttack++}, which differs significantly from the existing ones operating over the entire feature space. Specifically, DifAttack++ firstly disentangles an image's latent feature into an \textit{adversarial feature} (AF) and a \textit{visual feature} (VF) via an autoencoder equipped with our specially designed \textbf{H}ierarchical \textbf{D}ecouple-\textbf{F}usion (HDF) module, where the AF dominates the adversarial capability of an image, while the VF largely determines its visual appearance. We train such autoencoders for the clean and adversarial image domains respectively, meanwhile realizing feature disentanglement, by using pairs of clean images and their Adversarial Examples (AEs) generated from available surrogate models via white-box attack methods. Eventually, in the black-box attack stage, DifAttack++ iteratively optimizes the AF according to the query feedback from the victim model until a successful AE is generated, while keeping the VF unaltered. Extensive experimental results demonstrate that our method achieves superior ASR and query efficiency than SOTA methods, meanwhile exhibiting much better visual quality of AEs. The code is available at https://github.com/csjunjun/DifAttack.git.



## **13. ACE: A Model Poisoning Attack on Contribution Evaluation Methods in Federated Learning**

cs.CR

To appear in the 33rd USENIX Security Symposium, 2024

**SubmitDate**: 2024-06-05    [abs](http://arxiv.org/abs/2405.20975v2) [paper-pdf](http://arxiv.org/pdf/2405.20975v2)

**Authors**: Zhangchen Xu, Fengqing Jiang, Luyao Niu, Jinyuan Jia, Bo Li, Radha Poovendran

**Abstract**: In Federated Learning (FL), a set of clients collaboratively train a machine learning model (called global model) without sharing their local training data. The local training data of clients is typically non-i.i.d. and heterogeneous, resulting in varying contributions from individual clients to the final performance of the global model. In response, many contribution evaluation methods were proposed, where the server could evaluate the contribution made by each client and incentivize the high-contributing clients to sustain their long-term participation in FL. Existing studies mainly focus on developing new metrics or algorithms to better measure the contribution of each client. However, the security of contribution evaluation methods of FL operating in adversarial environments is largely unexplored. In this paper, we propose the first model poisoning attack on contribution evaluation methods in FL, termed ACE. Specifically, we show that any malicious client utilizing ACE could manipulate the parameters of its local model such that it is evaluated to have a high contribution by the server, even when its local training data is indeed of low quality. We perform both theoretical analysis and empirical evaluations of ACE. Theoretically, we show our design of ACE can effectively boost the malicious client's perceived contribution when the server employs the widely-used cosine distance metric to measure contribution. Empirically, our results show ACE effectively and efficiently deceive five state-of-the-art contribution evaluation methods. In addition, ACE preserves the accuracy of the final global models on testing inputs. We also explore six countermeasures to defend ACE. Our results show they are inadequate to thwart ACE, highlighting the urgent need for new defenses to safeguard the contribution evaluation methods in FL.



## **14. Prepared for the Worst: A Learning-Based Adversarial Attack for Resilience Analysis of the ICP Algorithm**

cs.RO

9 pages (7 content, 1 reference, 1 appendix). 6 figures, submitted to  the IEEE Robotics and Automation Letters (RA-L)

**SubmitDate**: 2024-06-05    [abs](http://arxiv.org/abs/2403.05666v2) [paper-pdf](http://arxiv.org/pdf/2403.05666v2)

**Authors**: Ziyu Zhang, Johann Laconte, Daniil Lisus, Timothy D. Barfoot

**Abstract**: This paper presents a novel method to assess the resilience of the Iterative Closest Point (ICP) algorithm via deep-learning-based attacks on lidar point clouds. For safety-critical applications such as autonomous navigation, ensuring the resilience of algorithms prior to deployments is of utmost importance. The ICP algorithm has become the standard for lidar-based localization. However, the pose estimate it produces can be greatly affected by corruption in the measurements. Corruption can arise from a variety of scenarios such as occlusions, adverse weather, or mechanical issues in the sensor. Unfortunately, the complex and iterative nature of ICP makes assessing its resilience to corruption challenging. While there have been efforts to create challenging datasets and develop simulations to evaluate the resilience of ICP empirically, our method focuses on finding the maximum possible ICP pose error using perturbation-based adversarial attacks. The proposed attack induces significant pose errors on ICP and outperforms baselines more than 88% of the time across a wide range of scenarios. As an example application, we demonstrate that our attack can be used to identify areas on a map where ICP is particularly vulnerable to corruption in the measurements.



## **15. Nonlinear Transformations Against Unlearnable Datasets**

cs.LG

**SubmitDate**: 2024-06-05    [abs](http://arxiv.org/abs/2406.02883v1) [paper-pdf](http://arxiv.org/pdf/2406.02883v1)

**Authors**: Thushari Hapuarachchi, Jing Lin, Kaiqi Xiong, Mohamed Rahouti, Gitte Ost

**Abstract**: Automated scraping stands out as a common method for collecting data in deep learning models without the authorization of data owners. Recent studies have begun to tackle the privacy concerns associated with this data collection method. Notable approaches include Deepconfuse, error-minimizing, error-maximizing (also known as adversarial poisoning), Neural Tangent Generalization Attack, synthetic, autoregressive, One-Pixel Shortcut, Self-Ensemble Protection, Entangled Features, Robust Error-Minimizing, Hypocritical, and TensorClog. The data generated by those approaches, called "unlearnable" examples, are prevented "learning" by deep learning models. In this research, we investigate and devise an effective nonlinear transformation framework and conduct extensive experiments to demonstrate that a deep neural network can effectively learn from the data/examples traditionally considered unlearnable produced by the above twelve approaches. The resulting approach improves the ability to break unlearnable data compared to the linear separable technique recently proposed by researchers. Specifically, our extensive experiments show that the improvement ranges from 0.34% to 249.59% for the unlearnable CIFAR10 datasets generated by those twelve data protection approaches, except for One-Pixel Shortcut. Moreover, the proposed framework achieves over 100% improvement of test accuracy for Autoregressive and REM approaches compared to the linear separable technique. Our findings suggest that these approaches are inadequate in preventing unauthorized uses of data in machine learning models. There is an urgent need to develop more robust protection mechanisms that effectively thwart an attacker from accessing data without proper authorization from the owners.



## **16. Improving the Adversarial Robustness for Speaker Verification by Self-Supervised Learning**

cs.SD

Accepted by TASLP

**SubmitDate**: 2024-06-05    [abs](http://arxiv.org/abs/2106.00273v4) [paper-pdf](http://arxiv.org/pdf/2106.00273v4)

**Authors**: Haibin Wu, Xu Li, Andy T. Liu, Zhiyong Wu, Helen Meng, Hung-yi Lee

**Abstract**: Previous works have shown that automatic speaker verification (ASV) is seriously vulnerable to malicious spoofing attacks, such as replay, synthetic speech, and recently emerged adversarial attacks. Great efforts have been dedicated to defending ASV against replay and synthetic speech; however, only a few approaches have been explored to deal with adversarial attacks. All the existing approaches to tackle adversarial attacks for ASV require the knowledge for adversarial samples generation, but it is impractical for defenders to know the exact attack algorithms that are applied by the in-the-wild attackers. This work is among the first to perform adversarial defense for ASV without knowing the specific attack algorithms. Inspired by self-supervised learning models (SSLMs) that possess the merits of alleviating the superficial noise in the inputs and reconstructing clean samples from the interrupted ones, this work regards adversarial perturbations as one kind of noise and conducts adversarial defense for ASV by SSLMs. Specifically, we propose to perform adversarial defense from two perspectives: 1) adversarial perturbation purification and 2) adversarial perturbation detection. Experimental results show that our detection module effectively shields the ASV by detecting adversarial samples with an accuracy of around 80%. Moreover, since there is no common metric for evaluating the adversarial defense performance for ASV, this work also formalizes evaluation metrics for adversarial defense considering both purification and detection based approaches into account. We sincerely encourage future works to benchmark their approaches based on the proposed evaluation framework.



## **17. Efficient Model-Stealing Attacks Against Inductive Graph Neural Networks**

cs.LG

arXiv admin note: text overlap with arXiv:2112.08331 by other authors

**SubmitDate**: 2024-06-04    [abs](http://arxiv.org/abs/2405.12295v2) [paper-pdf](http://arxiv.org/pdf/2405.12295v2)

**Authors**: Marcin Podhajski, Jan Dubiński, Franziska Boenisch, Adam Dziedzic, Agnieszka Pregowska, Tomasz Michalak

**Abstract**: Graph Neural Networks (GNNs) are recognized as potent tools for processing real-world data organized in graph structures. Especially inductive GNNs, which enable the processing of graph-structured data without relying on predefined graph structures, are gaining importance in an increasingly wide variety of applications. As these networks demonstrate proficiency across a range of tasks, they become lucrative targets for model-stealing attacks where an adversary seeks to replicate the functionality of the targeted network. A large effort has been made to develop model-stealing attacks that focus on models trained with images and texts. However, little attention has been paid to GNNs trained on graph data. This paper introduces a novel method for unsupervised model-stealing attacks against inductive GNNs, based on graph contrasting learning and spectral graph augmentations to efficiently extract information from the target model. The proposed attack is thoroughly evaluated on six datasets. The results show that this approach demonstrates a higher level of efficiency compared to existing stealing attacks. More concretely, our attack outperforms the baseline on all benchmarks achieving higher fidelity and downstream accuracy of the stolen model while requiring fewer queries sent to the target model.



## **18. Auditing Privacy Mechanisms via Label Inference Attacks**

cs.LG

**SubmitDate**: 2024-06-04    [abs](http://arxiv.org/abs/2406.02797v1) [paper-pdf](http://arxiv.org/pdf/2406.02797v1)

**Authors**: Róbert István Busa-Fekete, Travis Dick, Claudio Gentile, Andrés Muñoz Medina, Adam Smith, Marika Swanberg

**Abstract**: We propose reconstruction advantage measures to audit label privatization mechanisms. A reconstruction advantage measure quantifies the increase in an attacker's ability to infer the true label of an unlabeled example when provided with a private version of the labels in a dataset (e.g., aggregate of labels from different users or noisy labels output by randomized response), compared to an attacker that only observes the feature vectors, but may have prior knowledge of the correlation between features and labels. We consider two such auditing measures: one additive, and one multiplicative. These incorporate previous approaches taken in the literature on empirical auditing and differential privacy. The measures allow us to place a variety of proposed privatization schemes -- some differentially private, some not -- on the same footing. We analyze these measures theoretically under a distributional model which encapsulates reasonable adversarial settings. We also quantify their behavior empirically on real and simulated prediction tasks. Across a range of experimental settings, we find that differentially private schemes dominate or match the privacy-utility tradeoff of more heuristic approaches.



## **19. Proof-of-Learning with Incentive Security**

cs.CR

16 pages

**SubmitDate**: 2024-06-04    [abs](http://arxiv.org/abs/2404.09005v5) [paper-pdf](http://arxiv.org/pdf/2404.09005v5)

**Authors**: Zishuo Zhao, Zhixuan Fang, Xuechao Wang, Xi Chen, Yuan Zhou

**Abstract**: Most concurrent blockchain systems rely heavily on the Proof-of-Work (PoW) or Proof-of-Stake (PoS) mechanisms for decentralized consensus and security assurance. However, the substantial energy expenditure stemming from computationally intensive yet meaningless tasks has raised considerable concerns surrounding traditional PoW approaches, The PoS mechanism, while free of energy consumption, is subject to security and economic issues. Addressing these issues, the paradigm of Proof-of-Useful-Work (PoUW) seeks to employ challenges of practical significance as PoW, thereby imbuing energy consumption with tangible value. While previous efforts in Proof of Learning (PoL) explored the utilization of deep learning model training SGD tasks as PoUW challenges, recent research has revealed its vulnerabilities to adversarial attacks and the theoretical hardness in crafting a byzantine-secure PoL mechanism. In this paper, we introduce the concept of incentive-security that incentivizes rational provers to behave honestly for their best interest, bypassing the existing hardness to design a PoL mechanism with computational efficiency, a provable incentive-security guarantee and controllable difficulty. Particularly, our work is secure against two attacks to the recent work of Jia et al. [2021], and also improves the computational overhead from $\Theta(1)$ to $O(\frac{\log E}{E})$. Furthermore, while most recent research assumes trusted problem providers and verifiers, our design also guarantees frontend incentive-security even when problem providers are untrusted, and verifier incentive-security that bypasses the Verifier's Dilemma. By incorporating ML training into blockchain consensus mechanisms with provable guarantees, our research not only proposes an eco-friendly solution to blockchain systems, but also provides a proposal for a completely decentralized computing power market in the new AI age.



## **20. Tree Proof-of-Position Algorithms**

cs.DS

**SubmitDate**: 2024-06-04    [abs](http://arxiv.org/abs/2405.06761v2) [paper-pdf](http://arxiv.org/pdf/2405.06761v2)

**Authors**: Aida Manzano Kharman, Pietro Ferraro, Homayoun Hamedmoghadam, Robert Shorten

**Abstract**: We present a novel class of proof-of-position algorithms: Tree-Proof-of-Position (T-PoP). This algorithm is decentralised, collaborative and can be computed in a privacy preserving manner, such that agents do not need to reveal their position publicly. We make no assumptions of honest behaviour in the system, and consider varying ways in which agents may misbehave. Our algorithm is therefore resilient to highly adversarial scenarios. This makes it suitable for a wide class of applications, namely those in which trust in a centralised infrastructure may not be assumed, or high security risk scenarios. Our algorithm has a worst case quadratic runtime, making it suitable for hardware constrained IoT applications. We also provide a mathematical model that summarises T-PoP's performance for varying operating conditions. We then simulate T-PoP's behaviour with a large number of agent-based simulations, which are in complete agreement with our mathematical model, thus demonstrating its validity. T-PoP can achieve high levels of reliability and security by tuning its operating conditions, both in high and low density environments. Finally, we also present a mathematical model to probabilistically detect platooning attacks.



## **21. Rethinking the Vulnerabilities of Face Recognition Systems:From a Practical Perspective**

cs.CR

19 pages,version 2

**SubmitDate**: 2024-06-04    [abs](http://arxiv.org/abs/2405.12786v2) [paper-pdf](http://arxiv.org/pdf/2405.12786v2)

**Authors**: Jiahao Chen, Zhiqiang Shen, Yuwen Pu, Chunyi Zhou, Changjiang Li, Ting Wang, Shouling Ji

**Abstract**: Face Recognition Systems (FRS) have increasingly integrated into critical applications, including surveillance and user authentication, highlighting their pivotal role in modern security systems. Recent studies have revealed vulnerabilities in FRS to adversarial (e.g., adversarial patch attacks) and backdoor attacks (e.g., training data poisoning), raising significant concerns about their reliability and trustworthiness. Previous studies primarily focus on traditional adversarial or backdoor attacks, overlooking the resource-intensive or privileged-manipulation nature of such threats, thus limiting their practical generalization, stealthiness, universality and robustness. Correspondingly, in this paper, we delve into the inherent vulnerabilities in FRS through user studies and preliminary explorations. By exploiting these vulnerabilities, we identify a novel attack, facial identity backdoor attack dubbed FIBA, which unveils a potentially more devastating threat against FRS:an enrollment-stage backdoor attack. FIBA circumvents the limitations of traditional attacks, enabling broad-scale disruption by allowing any attacker donning a specific trigger to bypass these systems. This implies that after a single, poisoned example is inserted into the database, the corresponding trigger becomes a universal key for any attackers to spoof the FRS. This strategy essentially challenges the conventional attacks by initiating at the enrollment stage, dramatically transforming the threat landscape by poisoning the feature database rather than the training data.



## **22. Advancing Generalized Transfer Attack with Initialization Derived Bilevel Optimization and Dynamic Sequence Truncation**

cs.LG

Accepted by IJCAI 2024. 10 pages

**SubmitDate**: 2024-06-04    [abs](http://arxiv.org/abs/2406.02064v1) [paper-pdf](http://arxiv.org/pdf/2406.02064v1)

**Authors**: Yaohua Liu, Jiaxin Gao, Xuan Liu, Xianghao Jiao, Xin Fan, Risheng Liu

**Abstract**: Transfer attacks generate significant interest for real-world black-box applications by crafting transferable adversarial examples through surrogate models. Whereas, existing works essentially directly optimize the single-level objective w.r.t. the surrogate model, which always leads to poor interpretability of attack mechanism and limited generalization performance over unknown victim models. In this work, we propose the \textbf{B}il\textbf{E}vel \textbf{T}ransfer \textbf{A}ttac\textbf{K} (BETAK) framework by establishing an initialization derived bilevel optimization paradigm, which explicitly reformulates the nested constraint relationship between the Upper-Level (UL) pseudo-victim attacker and the Lower-Level (LL) surrogate attacker. Algorithmically, we introduce the Hyper Gradient Response (HGR) estimation as an effective feedback for the transferability over pseudo-victim attackers, and propose the Dynamic Sequence Truncation (DST) technique to dynamically adjust the back-propagation path for HGR and reduce computational overhead simultaneously. Meanwhile, we conduct detailed algorithmic analysis and provide convergence guarantee to support non-convexity of the LL surrogate attacker. Extensive evaluations demonstrate substantial improvement of BETAK (e.g., $\mathbf{53.41}$\% increase of attack success rates against IncRes-v$2_{ens}$) against different victims and defense methods in targeted and untargeted attack scenarios. The source code is available at https://github.com/callous-youth/BETAK.



## **23. Graph Adversarial Diffusion Convolution**

cs.LG

Accepted by ICML 2024

**SubmitDate**: 2024-06-04    [abs](http://arxiv.org/abs/2406.02059v1) [paper-pdf](http://arxiv.org/pdf/2406.02059v1)

**Authors**: Songtao Liu, Jinghui Chen, Tianfan Fu, Lu Lin, Marinka Zitnik, Dinghao Wu

**Abstract**: This paper introduces a min-max optimization formulation for the Graph Signal Denoising (GSD) problem. In this formulation, we first maximize the second term of GSD by introducing perturbations to the graph structure based on Laplacian distance and then minimize the overall loss of the GSD. By solving the min-max optimization problem, we derive a new variant of the Graph Diffusion Convolution (GDC) architecture, called Graph Adversarial Diffusion Convolution (GADC). GADC differs from GDC by incorporating an additional term that enhances robustness against adversarial attacks on the graph structure and noise in node features. Moreover, GADC improves the performance of GDC on heterophilic graphs. Extensive experiments demonstrate the effectiveness of GADC across various datasets. Code is available at https://github.com/SongtaoLiu0823/GADC.



## **24. ASETF: A Novel Method for Jailbreak Attack on LLMs through Translate Suffix Embeddings**

cs.CL

**SubmitDate**: 2024-06-04    [abs](http://arxiv.org/abs/2402.16006v2) [paper-pdf](http://arxiv.org/pdf/2402.16006v2)

**Authors**: Hao Wang, Hao Li, Minlie Huang, Lei Sha

**Abstract**: The safety defense methods of Large language models(LLMs) stays limited because the dangerous prompts are manually curated to just few known attack types, which fails to keep pace with emerging varieties. Recent studies found that attaching suffixes to harmful instructions can hack the defense of LLMs and lead to dangerous outputs. However, similar to traditional text adversarial attacks, this approach, while effective, is limited by the challenge of the discrete tokens. This gradient based discrete optimization attack requires over 100,000 LLM calls, and due to the unreadable of adversarial suffixes, it can be relatively easily penetrated by common defense methods such as perplexity filters. To cope with this challenge, in this paper, we proposes an Adversarial Suffix Embedding Translation Framework (ASETF), aimed at transforming continuous adversarial suffix embeddings into coherent and understandable text. This method greatly reduces the computational overhead during the attack process and helps to automatically generate multiple adversarial samples, which can be used as data to strengthen LLMs security defense. Experimental evaluations were conducted on Llama2, Vicuna, and other prominent LLMs, employing harmful directives sourced from the Advbench dataset. The results indicate that our method significantly reduces the computation time of adversarial suffixes and achieves a much better attack success rate to existing techniques, while significantly enhancing the textual fluency of the prompts. In addition, our approach can be generalized into a broader method for generating transferable adversarial suffixes that can successfully attack multiple LLMs, even black-box LLMs, such as ChatGPT and Gemini.



## **25. SVASTIN: Sparse Video Adversarial Attack via Spatio-Temporal Invertible Neural Networks**

cs.CV

**SubmitDate**: 2024-06-04    [abs](http://arxiv.org/abs/2406.01894v1) [paper-pdf](http://arxiv.org/pdf/2406.01894v1)

**Authors**: Yi Pan, Jun-Jie Huang, Zihan Chen, Wentao Zhao, Ziyue Wang

**Abstract**: Robust and imperceptible adversarial video attack is challenging due to the spatial and temporal characteristics of videos. The existing video adversarial attack methods mainly take a gradient-based approach and generate adversarial videos with noticeable perturbations. In this paper, we propose a novel Sparse Adversarial Video Attack via Spatio-Temporal Invertible Neural Networks (SVASTIN) to generate adversarial videos through spatio-temporal feature space information exchanging. It consists of a Guided Target Video Learning (GTVL) module to balance the perturbation budget and optimization speed and a Spatio-Temporal Invertible Neural Network (STIN) module to perform spatio-temporal feature space information exchanging between a source video and the target feature tensor learned by GTVL module. Extensive experiments on UCF-101 and Kinetics-400 demonstrate that our proposed SVASTIN can generate adversarial examples with higher imperceptibility than the state-of-the-art methods with the higher fooling rate. Code is available at \href{https://github.com/Brittany-Chen/SVASTIN}{https://github.com/Brittany-Chen/SVASTIN}.



## **26. CR-UTP: Certified Robustness against Universal Text Perturbations on Large Language Models**

cs.CL

Accepted by ACL Findings 2024

**SubmitDate**: 2024-06-05    [abs](http://arxiv.org/abs/2406.01873v2) [paper-pdf](http://arxiv.org/pdf/2406.01873v2)

**Authors**: Qian Lou, Xin Liang, Jiaqi Xue, Yancheng Zhang, Rui Xie, Mengxin Zheng

**Abstract**: It is imperative to ensure the stability of every prediction made by a language model; that is, a language's prediction should remain consistent despite minor input variations, like word substitutions. In this paper, we investigate the problem of certifying a language model's robustness against Universal Text Perturbations (UTPs), which have been widely used in universal adversarial attacks and backdoor attacks. Existing certified robustness based on random smoothing has shown considerable promise in certifying the input-specific text perturbations (ISTPs), operating under the assumption that any random alteration of a sample's clean or adversarial words would negate the impact of sample-wise perturbations. However, with UTPs, masking only the adversarial words can eliminate the attack. A naive method is to simply increase the masking ratio and the likelihood of masking attack tokens, but it leads to a significant reduction in both certified accuracy and the certified radius due to input corruption by extensive masking. To solve this challenge, we introduce a novel approach, the superior prompt search method, designed to identify a superior prompt that maintains higher certified accuracy under extensive masking. Additionally, we theoretically motivate why ensembles are a particularly suitable choice as base prompts for random smoothing. The method is denoted by superior prompt ensembling technique. We also empirically confirm this technique, obtaining state-of-the-art results in multiple settings. These methodologies, for the first time, enable high certified accuracy against both UTPs and ISTPs. The source code of CR-UTP is available at \url {https://github.com/UCFML-Research/CR-UTP}.



## **27. Adversarial Attacks on Combinatorial Multi-Armed Bandits**

cs.LG

28 pages, Accepted to ICML 2024

**SubmitDate**: 2024-06-04    [abs](http://arxiv.org/abs/2310.05308v2) [paper-pdf](http://arxiv.org/pdf/2310.05308v2)

**Authors**: Rishab Balasubramanian, Jiawei Li, Prasad Tadepalli, Huazheng Wang, Qingyun Wu, Haoyu Zhao

**Abstract**: We study reward poisoning attacks on Combinatorial Multi-armed Bandits (CMAB). We first provide a sufficient and necessary condition for the attackability of CMAB, a notion to capture the vulnerability and robustness of CMAB. The attackability condition depends on the intrinsic properties of the corresponding CMAB instance such as the reward distributions of super arms and outcome distributions of base arms. Additionally, we devise an attack algorithm for attackable CMAB instances. Contrary to prior understanding of multi-armed bandits, our work reveals a surprising fact that the attackability of a specific CMAB instance also depends on whether the bandit instance is known or unknown to the adversary. This finding indicates that adversarial attacks on CMAB are difficult in practice and a general attack strategy for any CMAB instance does not exist since the environment is mostly unknown to the adversary. We validate our theoretical findings via extensive experiments on real-world CMAB applications including probabilistic maximum covering problem, online minimum spanning tree, cascading bandits for online ranking, and online shortest path.



## **28. Reproducibility Study on Adversarial Attacks Against Robust Transformer Trackers**

cs.CV

Published in Transactions on Machine Learning Research (05/2024):  https://openreview.net/forum?id=FEEKR0Vl9s

**SubmitDate**: 2024-06-03    [abs](http://arxiv.org/abs/2406.01765v1) [paper-pdf](http://arxiv.org/pdf/2406.01765v1)

**Authors**: Fatemeh Nourilenjan Nokabadi, Jean-François Lalonde, Christian Gagné

**Abstract**: New transformer networks have been integrated into object tracking pipelines and have demonstrated strong performance on the latest benchmarks. This paper focuses on understanding how transformer trackers behave under adversarial attacks and how different attacks perform on tracking datasets as their parameters change. We conducted a series of experiments to evaluate the effectiveness of existing adversarial attacks on object trackers with transformer and non-transformer backbones. We experimented on 7 different trackers, including 3 that are transformer-based, and 4 which leverage other architectures. These trackers are tested against 4 recent attack methods to assess their performance and robustness on VOT2022ST, UAV123 and GOT10k datasets. Our empirical study focuses on evaluating adversarial robustness of object trackers based on bounding box versus binary mask predictions, and attack methods at different levels of perturbations. Interestingly, our study found that altering the perturbation level may not significantly affect the overall object tracking results after the attack. Similarly, the sparsity and imperceptibility of the attack perturbations may remain stable against perturbation level shifts. By applying a specific attack on all transformer trackers, we show that new transformer trackers having a stronger cross-attention modeling achieve a greater adversarial robustness on tracking datasets, such as VOT2022ST and GOT10k. Our results also indicate the necessity for new attack methods to effectively tackle the latest types of transformer trackers. The codes necessary to reproduce this study are available at https://github.com/fatemehN/ReproducibilityStudy.



## **29. On the completeness of several fortification-interdiction games in the Polynomial Hierarchy**

cs.CC

**SubmitDate**: 2024-06-03    [abs](http://arxiv.org/abs/2406.01756v1) [paper-pdf](http://arxiv.org/pdf/2406.01756v1)

**Authors**: Alberto Boggio Tomasaz, Margarida Carvalho, Roberto Cordone, Pierre Hosteins

**Abstract**: Fortification-interdiction games are tri-level adversarial games where two opponents act in succession to protect, disrupt and simply use an infrastructure for a specific purpose. Many such games have been formulated and tackled in the literature through specific algorithmic methods, however very few investigations exist on the completeness of such fortification problems in order to locate them rigorously in the polynomial hierarchy. We clarify the completeness status of several well-known fortification problems, such as the Tri-level Interdiction Knapsack Problem with unit fortification and attack weights, the Max-flow Interdiction Problem and Shortest Path Interdiction Problem with Fortification, the Multi-level Critical Node Problem with unit weights, as well as a well-studied electric grid defence planning problem. For all of these problems, we prove their completeness either for the $\Sigma^p_2$ or the $\Sigma^p_3$ class of the polynomial hierarchy. We also prove that the Multi-level Fortification-Interdiction Knapsack Problem with an arbitrary number of protection and interdiction rounds and unit fortification and attack weights is complete for any level of the polynomial hierarchy, therefore providing a useful basis for further attempts at proving the completeness of protection-interdiction games at any level of said hierarchy.



## **30. MAWSEO: Adversarial Wiki Search Poisoning for Illicit Online Promotion**

cs.CR

Accepted at the 45th IEEE Symposium on Security and Privacy (IEEE S&P  2024)

**SubmitDate**: 2024-06-03    [abs](http://arxiv.org/abs/2304.11300v3) [paper-pdf](http://arxiv.org/pdf/2304.11300v3)

**Authors**: Zilong Lin, Zhengyi Li, Xiaojing Liao, XiaoFeng Wang, Xiaozhong Liu

**Abstract**: As a prominent instance of vandalism edits, Wiki search poisoning for illicit promotion is a cybercrime in which the adversary aims at editing Wiki articles to promote illicit businesses through Wiki search results of relevant queries. In this paper, we report a study that, for the first time, shows that such stealthy blackhat SEO on Wiki can be automated. Our technique, called MAWSEO, employs adversarial revisions to achieve real-world cybercriminal objectives, including rank boosting, vandalism detection evasion, topic relevancy, semantic consistency, user awareness (but not alarming) of promotional content, etc. Our evaluation and user study demonstrate that MAWSEO is capable of effectively and efficiently generating adversarial vandalism edits, which can bypass state-of-the-art built-in Wiki vandalism detectors, and also get promotional content through to Wiki users without triggering their alarms. In addition, we investigated potential defense, including coherence based detection and adversarial training of vandalism detection, against our attack in the Wiki ecosystem.



## **31. Mixing Classifiers to Alleviate the Accuracy-Robustness Trade-Off**

cs.LG

arXiv admin note: text overlap with arXiv:2301.12554

**SubmitDate**: 2024-06-03    [abs](http://arxiv.org/abs/2311.15165v2) [paper-pdf](http://arxiv.org/pdf/2311.15165v2)

**Authors**: Yatong Bai, Brendon G. Anderson, Somayeh Sojoudi

**Abstract**: Deep neural classifiers have recently found tremendous success in data-driven control systems. However, existing models suffer from a trade-off between accuracy and adversarial robustness. This limitation must be overcome in the control of safety-critical systems that require both high performance and rigorous robustness guarantees. In this work, we develop classifiers that simultaneously inherit high robustness from robust models and high accuracy from standard models. Specifically, we propose a theoretically motivated formulation that mixes the output probabilities of a standard neural network and a robust neural network. Both base classifiers are pre-trained, and thus our method does not require additional training. Our numerical experiments verify that the mixed classifier noticeably improves the accuracy-robustness trade-off and identify the confidence property of the robust base classifier as the key leverage of this more benign trade-off. Our theoretical results prove that under mild assumptions, when the robustness of the robust base model is certifiable, no alteration or attack within a closed-form $\ell_p$ radius on an input can result in the misclassification of the mixed classifier.



## **32. Model for Peanuts: Hijacking ML Models without Training Access is Possible**

cs.CR

17 pages, 14 figures, 7 tables

**SubmitDate**: 2024-06-03    [abs](http://arxiv.org/abs/2406.01708v1) [paper-pdf](http://arxiv.org/pdf/2406.01708v1)

**Authors**: Mahmoud Ghorbel, Halima Bouzidi, Ioan Marius Bilasco, Ihsen Alouani

**Abstract**: The massive deployment of Machine Learning (ML) models has been accompanied by the emergence of several attacks that threaten their trustworthiness and raise ethical and societal concerns such as invasion of privacy, discrimination risks, and lack of accountability. Model hijacking is one of these attacks, where the adversary aims to hijack a victim model to execute a different task than its original one. Model hijacking can cause accountability and security risks since a hijacked model owner can be framed for having their model offering illegal or unethical services. Prior state-of-the-art works consider model hijacking as a training time attack, whereby an adversary requires access to the ML model training to execute their attack. In this paper, we consider a stronger threat model where the attacker has no access to the training phase of the victim model. Our intuition is that ML models, typically over-parameterized, might (unintentionally) learn more than the intended task for they are trained. We propose a simple approach for model hijacking at inference time named SnatchML to classify unknown input samples using distance measures in the latent space of the victim model to previously known samples associated with the hijacking task classes. SnatchML empirically shows that benign pre-trained models can execute tasks that are semantically related to the initial task. Surprisingly, this can be true even for hijacking tasks unrelated to the original task. We also explore different methods to mitigate this risk. We first propose a novel approach we call meta-unlearning, designed to help the model unlearn a potentially malicious task while training on the original task dataset. We also provide insights on over-parameterization as one possible inherent factor that makes model hijacking easier, and we accordingly propose a compression-based countermeasure against this attack.



## **33. From Feature Visualization to Visual Circuits: Effect of Adversarial Model Manipulation**

cs.CV

Under review

**SubmitDate**: 2024-06-03    [abs](http://arxiv.org/abs/2406.01365v1) [paper-pdf](http://arxiv.org/pdf/2406.01365v1)

**Authors**: Geraldin Nanfack, Michael Eickenberg, Eugene Belilovsky

**Abstract**: Understanding the inner working functionality of large-scale deep neural networks is challenging yet crucial in several high-stakes applications. Mechanistic inter- pretability is an emergent field that tackles this challenge, often by identifying human-understandable subgraphs in deep neural networks known as circuits. In vision-pretrained models, these subgraphs are usually interpreted by visualizing their node features through a popular technique called feature visualization. Recent works have analyzed the stability of different feature visualization types under the adversarial model manipulation framework. This paper starts by addressing limitations in existing works by proposing a novel attack called ProxPulse that simultaneously manipulates the two types of feature visualizations. Surprisingly, when analyzing these attacks under the umbrella of visual circuits, we find that visual circuits show some robustness to ProxPulse. We, therefore, introduce a new attack based on ProxPulse that unveils the manipulability of visual circuits, shedding light on their lack of robustness. The effectiveness of these attacks is validated using pre-trained AlexNet and ResNet-50 models on ImageNet.



## **34. Exploring the Robustness of Decision-Level Through Adversarial Attacks on LLM-Based Embodied Models**

cs.MM

**SubmitDate**: 2024-06-03    [abs](http://arxiv.org/abs/2405.19802v2) [paper-pdf](http://arxiv.org/pdf/2405.19802v2)

**Authors**: Shuyuan Liu, Jiawei Chen, Shouwei Ruan, Hang Su, Zhaoxia Yin

**Abstract**: Embodied intelligence empowers agents with a profound sense of perception, enabling them to respond in a manner closely aligned with real-world situations. Large Language Models (LLMs) delve into language instructions with depth, serving a crucial role in generating plans for intricate tasks. Thus, LLM-based embodied models further enhance the agent's capacity to comprehend and process information. However, this amalgamation also ushers in new challenges in the pursuit of heightened intelligence. Specifically, attackers can manipulate LLMs to produce irrelevant or even malicious outputs by altering their prompts. Confronted with this challenge, we observe a notable absence of multi-modal datasets essential for comprehensively evaluating the robustness of LLM-based embodied models. Consequently, we construct the Embodied Intelligent Robot Attack Dataset (EIRAD), tailored specifically for robustness evaluation. Additionally, two attack strategies are devised, including untargeted attacks and targeted attacks, to effectively simulate a range of diverse attack scenarios. At the same time, during the attack process, to more accurately ascertain whether our method is successful in attacking the LLM-based embodied model, we devise a new attack success evaluation method utilizing the BLIP2 model. Recognizing the time and cost-intensive nature of the GCG algorithm in attacks, we devise a scheme for prompt suffix initialization based on various target tasks, thus expediting the convergence process. Experimental results demonstrate that our method exhibits a superior attack success rate when targeting LLM-based embodied models, indicating a lower level of decision-level robustness in these models.



## **35. Fundamental Limitations of Alignment in Large Language Models**

cs.CL

**SubmitDate**: 2024-06-03    [abs](http://arxiv.org/abs/2304.11082v6) [paper-pdf](http://arxiv.org/pdf/2304.11082v6)

**Authors**: Yotam Wolf, Noam Wies, Oshri Avnery, Yoav Levine, Amnon Shashua

**Abstract**: An important aspect in developing language models that interact with humans is aligning their behavior to be useful and unharmful for their human users. This is usually achieved by tuning the model in a way that enhances desired behaviors and inhibits undesired ones, a process referred to as alignment. In this paper, we propose a theoretical approach called Behavior Expectation Bounds (BEB) which allows us to formally investigate several inherent characteristics and limitations of alignment in large language models. Importantly, we prove that within the limits of this framework, for any behavior that has a finite probability of being exhibited by the model, there exist prompts that can trigger the model into outputting this behavior, with probability that increases with the length of the prompt. This implies that any alignment process that attenuates an undesired behavior but does not remove it altogether, is not safe against adversarial prompting attacks. Furthermore, our framework hints at the mechanism by which leading alignment approaches such as reinforcement learning from human feedback make the LLM prone to being prompted into the undesired behaviors. This theoretical result is being experimentally demonstrated in large scale by the so called contemporary "chatGPT jailbreaks", where adversarial users trick the LLM into breaking its alignment guardrails by triggering it into acting as a malicious persona. Our results expose fundamental limitations in alignment of LLMs and bring to the forefront the need to devise reliable mechanisms for ensuring AI safety.



## **36. Constraint-based Adversarial Example Synthesis**

cs.CR

**SubmitDate**: 2024-06-03    [abs](http://arxiv.org/abs/2406.01219v1) [paper-pdf](http://arxiv.org/pdf/2406.01219v1)

**Authors**: Fang Yu, Ya-Yu Chi, Yu-Fang Chen

**Abstract**: In the era of rapid advancements in artificial intelligence (AI), neural network models have achieved notable breakthroughs. However, concerns arise regarding their vulnerability to adversarial attacks. This study focuses on enhancing Concolic Testing, a specialized technique for testing Python programs implementing neural networks. The extended tool, PyCT, now accommodates a broader range of neural network operations, including floating-point and activation function computations. By systematically generating prediction path constraints, the research facilitates the identification of potential adversarial examples. Demonstrating effectiveness across various neural network architectures, the study highlights the vulnerability of Python-based neural network models to adversarial attacks. This research contributes to securing AI-powered applications by emphasizing the need for robust testing methodologies to detect and mitigate potential adversarial threats. It underscores the importance of rigorous testing techniques in fortifying neural network models for reliable applications in Python.



## **37. Are AI-Generated Text Detectors Robust to Adversarial Perturbations?**

cs.CL

Accepted to ACL 2024 main conference

**SubmitDate**: 2024-06-03    [abs](http://arxiv.org/abs/2406.01179v1) [paper-pdf](http://arxiv.org/pdf/2406.01179v1)

**Authors**: Guanhua Huang, Yuchen Zhang, Zhe Li, Yongjian You, Mingze Wang, Zhouwang Yang

**Abstract**: The widespread use of large language models (LLMs) has sparked concerns about the potential misuse of AI-generated text, as these models can produce content that closely resembles human-generated text. Current detectors for AI-generated text (AIGT) lack robustness against adversarial perturbations, with even minor changes in characters or words causing a reversal in distinguishing between human-created and AI-generated text. This paper investigates the robustness of existing AIGT detection methods and introduces a novel detector, the Siamese Calibrated Reconstruction Network (SCRN). The SCRN employs a reconstruction network to add and remove noise from text, extracting a semantic representation that is robust to local perturbations. We also propose a siamese calibration technique to train the model to make equally confidence predictions under different noise, which improves the model's robustness against adversarial perturbations. Experiments on four publicly available datasets show that the SCRN outperforms all baseline methods, achieving 6.5\%-18.25\% absolute accuracy improvement over the best baseline method under adversarial attacks. Moreover, it exhibits superior generalizability in cross-domain, cross-genre, and mixed-source scenarios. The code is available at \url{https://github.com/CarlanLark/Robust-AIGC-Detector}.



## **38. Genshin: General Shield for Natural Language Processing with Large Language Models**

cs.CL

**SubmitDate**: 2024-06-03    [abs](http://arxiv.org/abs/2405.18741v2) [paper-pdf](http://arxiv.org/pdf/2405.18741v2)

**Authors**: Xiao Peng, Tao Liu, Ying Wang

**Abstract**: Large language models (LLMs) like ChatGPT, Gemini, or LLaMA have been trending recently, demonstrating considerable advancement and generalizability power in countless domains. However, LLMs create an even bigger black box exacerbating opacity, with interpretability limited to few approaches. The uncertainty and opacity embedded in LLMs' nature restrict their application in high-stakes domains like financial fraud, phishing, etc. Current approaches mainly rely on traditional textual classification with posterior interpretable algorithms, suffering from attackers who may create versatile adversarial samples to break the system's defense, forcing users to make trade-offs between efficiency and robustness. To address this issue, we propose a novel cascading framework called Genshin (General Shield for Natural Language Processing with Large Language Models), utilizing LLMs as defensive one-time plug-ins. Unlike most applications of LLMs that try to transform text into something new or structural, Genshin uses LLMs to recover text to its original state. Genshin aims to combine the generalizability of the LLM, the discrimination of the median model, and the interpretability of the simple model. Our experiments on the task of sentimental analysis and spam detection have shown fatal flaws of the current median models and exhilarating results on LLMs' recovery ability, demonstrating that Genshin is both effective and efficient. In our ablation study, we unearth several intriguing observations. Utilizing the LLM defender, a tool derived from the 4th paradigm, we have reproduced BERT's 15% optimal mask rate results in the 3rd paradigm of NLP. Additionally, when employing the LLM as a potential adversarial tool, attackers are capable of executing effective attacks that are nearly semantically lossless.



## **39. BadRAG: Identifying Vulnerabilities in Retrieval Augmented Generation of Large Language Models**

cs.CR

**SubmitDate**: 2024-06-03    [abs](http://arxiv.org/abs/2406.00083v1) [paper-pdf](http://arxiv.org/pdf/2406.00083v1)

**Authors**: Jiaqi Xue, Mengxin Zheng, Yebowen Hu, Fei Liu, Xun Chen, Qian Lou

**Abstract**: Large Language Models (LLMs) are constrained by outdated information and a tendency to generate incorrect data, commonly referred to as "hallucinations." Retrieval-Augmented Generation (RAG) addresses these limitations by combining the strengths of retrieval-based methods and generative models. This approach involves retrieving relevant information from a large, up-to-date dataset and using it to enhance the generation process, leading to more accurate and contextually appropriate responses. Despite its benefits, RAG introduces a new attack surface for LLMs, particularly because RAG databases are often sourced from public data, such as the web. In this paper, we propose \TrojRAG{} to identify the vulnerabilities and attacks on retrieval parts (RAG database) and their indirect attacks on generative parts (LLMs). Specifically, we identify that poisoning several customized content passages could achieve a retrieval backdoor, where the retrieval works well for clean queries but always returns customized poisoned adversarial queries. Triggers and poisoned passages can be highly customized to implement various attacks. For example, a trigger could be a semantic group like "The Republican Party, Donald Trump, etc." Adversarial passages can be tailored to different contents, not only linked to the triggers but also used to indirectly attack generative LLMs without modifying them. These attacks can include denial-of-service attacks on RAG and semantic steering attacks on LLM generations conditioned by the triggers. Our experiments demonstrate that by just poisoning 10 adversarial passages can induce 98.2\% success rate to retrieve the adversarial passages. Then, these passages can increase the reject ratio of RAG-based GPT-4 from 0.01\% to 74.6\% or increase the rate of negative responses from 0.22\% to 72\% for targeted queries.



## **40. Assessing the Adversarial Security of Perceptual Hashing Algorithms**

cs.CR

**SubmitDate**: 2024-06-03    [abs](http://arxiv.org/abs/2406.00918v1) [paper-pdf](http://arxiv.org/pdf/2406.00918v1)

**Authors**: Jordan Madden, Moxanki Bhavsar, Lhamo Dorje, Xiaohua Li

**Abstract**: Perceptual hashing algorithms (PHAs) are utilized extensively for identifying illegal online content. Given their crucial role in sensitive applications, understanding their security strengths and weaknesses is critical. This paper compares three major PHAs deployed widely in practice: PhotoDNA, PDQ, and NeuralHash, and assesses their robustness against three typical attacks: normal image editing attacks, malicious adversarial attacks, and hash inversion attacks. Contrary to prevailing studies, this paper reveals that these PHAs exhibit resilience to black-box adversarial attacks when realistic constraints regarding the distortion and query budget are applied, attributed to the unique property of random hash variations. Moreover, this paper illustrates that original images can be reconstructed from the hash bits, raising significant privacy concerns. By comprehensively exposing their security vulnerabilities, this paper contributes to the ongoing efforts aimed at enhancing the security of PHAs for effective deployment.



## **41. Cross-lingual Cross-temporal Summarization: Dataset, Models, Evaluation**

cs.CL

Computational Linguistics. Submitted manuscript.  https://direct.mit.edu/coli/article/doi/10.1162/coli_a_00519/121095/Cross-lingual-Cross-temporal-Summarization-Dataset

**SubmitDate**: 2024-06-02    [abs](http://arxiv.org/abs/2306.12916v3) [paper-pdf](http://arxiv.org/pdf/2306.12916v3)

**Authors**: Ran Zhang, Jihed Ouni, Steffen Eger

**Abstract**: While summarization has been extensively researched in natural language processing (NLP), cross-lingual cross-temporal summarization (CLCTS) is a largely unexplored area that has the potential to improve cross-cultural accessibility and understanding. This paper comprehensively addresses the CLCTS task, including dataset creation, modeling, and evaluation. We (1) build the first CLCTS corpus with 328 instances for hDe-En (extended version with 455 instances) and 289 for hEn-De (extended version with 501 instances), leveraging historical fiction texts and Wikipedia summaries in English and German; (2) examine the effectiveness of popular transformer end-to-end models with different intermediate finetuning tasks; (3) explore the potential of GPT-3.5 as a summarizer; (4) report evaluations from humans, GPT-4, and several recent automatic evaluation metrics. Our results indicate that intermediate task finetuned end-to-end models generate bad to moderate quality summaries while GPT-3.5, as a zero-shot summarizer, provides moderate to good quality outputs. GPT-3.5 also seems very adept at normalizing historical text. To assess data contamination in GPT-3.5, we design an adversarial attack scheme in which we find that GPT-3.5 performs slightly worse for unseen source documents compared to seen documents. Moreover, it sometimes hallucinates when the source sentences are inverted against its prior knowledge with a summarization accuracy of 0.67 for plot omission, 0.71 for entity swap, and 0.53 for plot negation. Overall, our regression results of model performances suggest that longer, older, and more complex source texts (all of which are more characteristic for historical language variants) are harder to summarize for all models, indicating the difficulty of the CLCTS task.



## **42. PureEBM: Universal Poison Purification via Mid-Run Dynamics of Energy-Based Models**

cs.LG

arXiv admin note: substantial text overlap with arXiv:2405.18627

**SubmitDate**: 2024-06-02    [abs](http://arxiv.org/abs/2405.19376v2) [paper-pdf](http://arxiv.org/pdf/2405.19376v2)

**Authors**: Omead Pooladzandi, Jeffrey Jiang, Sunay Bhat, Gregory Pottie

**Abstract**: Data poisoning attacks pose a significant threat to the integrity of machine learning models by leading to misclassification of target distribution data by injecting adversarial examples during training. Existing state-of-the-art (SoTA) defense methods suffer from limitations, such as significantly reduced generalization performance and significant overhead during training, making them impractical or limited for real-world applications. In response to this challenge, we introduce a universal data purification method that defends naturally trained classifiers from malicious white-, gray-, and black-box image poisons by applying a universal stochastic preprocessing step $\Psi_{T}(x)$, realized by iterative Langevin sampling of a convergent Energy Based Model (EBM) initialized with an image $x.$ Mid-run dynamics of $\Psi_{T}(x)$ purify poison information with minimal impact on features important to the generalization of a classifier network. We show that EBMs remain universal purifiers, even in the presence of poisoned EBM training data, and achieve SoTA defense on leading triggered and triggerless poisons. This work is a subset of a larger framework introduced in \pgen with a more detailed focus on EBM purification and poison defense.



## **43. PureGen: Universal Data Purification for Train-Time Poison Defense via Generative Model Dynamics**

cs.LG

**SubmitDate**: 2024-06-02    [abs](http://arxiv.org/abs/2405.18627v2) [paper-pdf](http://arxiv.org/pdf/2405.18627v2)

**Authors**: Sunay Bhat, Jeffrey Jiang, Omead Pooladzandi, Alexander Branch, Gregory Pottie

**Abstract**: Train-time data poisoning attacks threaten machine learning models by introducing adversarial examples during training, leading to misclassification. Current defense methods often reduce generalization performance, are attack-specific, and impose significant training overhead. To address this, we introduce a set of universal data purification methods using a stochastic transform, $\Psi(x)$, realized via iterative Langevin dynamics of Energy-Based Models (EBMs), Denoising Diffusion Probabilistic Models (DDPMs), or both. These approaches purify poisoned data with minimal impact on classifier generalization. Our specially trained EBMs and DDPMs provide state-of-the-art defense against various attacks (including Narcissus, Bullseye Polytope, Gradient Matching) on CIFAR-10, Tiny-ImageNet, and CINIC-10, without needing attack or classifier-specific information. We discuss performance trade-offs and show that our methods remain highly effective even with poisoned or distributionally shifted generative model training data.



## **44. Constrained Adaptive Attack: Effective Adversarial Attack Against Deep Neural Networks for Tabular Data**

cs.LG

**SubmitDate**: 2024-06-02    [abs](http://arxiv.org/abs/2406.00775v1) [paper-pdf](http://arxiv.org/pdf/2406.00775v1)

**Authors**: Thibault Simonetto, Salah Ghamizi, Maxime Cordy

**Abstract**: State-of-the-art deep learning models for tabular data have recently achieved acceptable performance to be deployed in industrial settings. However, the robustness of these models remains scarcely explored. Contrary to computer vision, there are no effective attacks to properly evaluate the adversarial robustness of deep tabular models due to intrinsic properties of tabular data, such as categorical features, immutability, and feature relationship constraints. To fill this gap, we first propose CAPGD, a gradient attack that overcomes the failures of existing gradient attacks with adaptive mechanisms. This new attack does not require parameter tuning and further degrades the accuracy, up to 81% points compared to the previous gradient attacks. Second, we design CAA, an efficient evasion attack that combines our CAPGD attack and MOEVA, the best search-based attack. We demonstrate the effectiveness of our attacks on five architectures and four critical use cases. Our empirical study demonstrates that CAA outperforms all existing attacks in 17 over the 20 settings, and leads to a drop in the accuracy by up to 96.1% points and 21.9% points compared to CAPGD and MOEVA respectively while being up to five times faster than MOEVA. Given the effectiveness and efficiency of our new attacks, we argue that they should become the minimal test for any new defense or robust architectures in tabular machine learning.



## **45. IBD-PSC: Input-level Backdoor Detection via Parameter-oriented Scaling Consistency**

cs.LG

Accepted to ICML 2024, 31 pages

**SubmitDate**: 2024-06-02    [abs](http://arxiv.org/abs/2405.09786v3) [paper-pdf](http://arxiv.org/pdf/2405.09786v3)

**Authors**: Linshan Hou, Ruili Feng, Zhongyun Hua, Wei Luo, Leo Yu Zhang, Yiming Li

**Abstract**: Deep neural networks (DNNs) are vulnerable to backdoor attacks, where adversaries can maliciously trigger model misclassifications by implanting a hidden backdoor during model training. This paper proposes a simple yet effective input-level backdoor detection (dubbed IBD-PSC) as a `firewall' to filter out malicious testing images. Our method is motivated by an intriguing phenomenon, i.e., parameter-oriented scaling consistency (PSC), where the prediction confidences of poisoned samples are significantly more consistent than those of benign ones when amplifying model parameters. In particular, we provide theoretical analysis to safeguard the foundations of the PSC phenomenon. We also design an adaptive method to select BN layers to scale up for effective detection. Extensive experiments are conducted on benchmark datasets, verifying the effectiveness and efficiency of our IBD-PSC method and its resistance to adaptive attacks. Codes are available at \href{https://github.com/THUYimingLi/BackdoorBox}{BackdoorBox}.



## **46. Jailbreaking Prompt Attack: A Controllable Adversarial Attack against Diffusion Models**

cs.CR

**SubmitDate**: 2024-06-02    [abs](http://arxiv.org/abs/2404.02928v2) [paper-pdf](http://arxiv.org/pdf/2404.02928v2)

**Authors**: Jiachen Ma, Anda Cao, Zhiqing Xiao, Jie Zhang, Chao Ye, Junbo Zhao

**Abstract**: Text-to-Image (T2I) models have received widespread attention due to their remarkable generation capabilities. However, concerns have been raised about the ethical implications of the models in generating Not Safe for Work (NSFW) images because NSFW images may cause discomfort to people or be used for illegal purposes. To mitigate the generation of such images, T2I models deploy various types of safety checkers. However, they still cannot completely prevent the generation of NSFW images. In this paper, we propose the Jailbreak Prompt Attack (JPA) - an automatic attack framework. We aim to maintain prompts that bypass safety checkers while preserving the semantics of the original images. Specifically, we aim to find prompts that can bypass safety checkers because of the robustness of the text space. Our evaluation demonstrates that JPA successfully bypasses both online services with closed-box safety checkers and offline defenses safety checkers to generate NSFW images.



## **47. Generalization Bound and New Algorithm for Clean-Label Backdoor Attack**

cs.LG

**SubmitDate**: 2024-06-02    [abs](http://arxiv.org/abs/2406.00588v1) [paper-pdf](http://arxiv.org/pdf/2406.00588v1)

**Authors**: Lijia Yu, Shuang Liu, Yibo Miao, Xiao-Shan Gao, Lijun Zhang

**Abstract**: The generalization bound is a crucial theoretical tool for assessing the generalizability of learning methods and there exist vast literatures on generalizability of normal learning, adversarial learning, and data poisoning. Unlike other data poison attacks, the backdoor attack has the special property that the poisoned triggers are contained in both the training set and the test set and the purpose of the attack is two-fold. To our knowledge, the generalization bound for the backdoor attack has not been established. In this paper, we fill this gap by deriving algorithm-independent generalization bounds in the clean-label backdoor attack scenario. Precisely, based on the goals of backdoor attack, we give upper bounds for the clean sample population errors and the poison population errors in terms of the empirical error on the poisoned training dataset. Furthermore, based on the theoretical result, a new clean-label backdoor attack is proposed that computes the poisoning trigger by combining adversarial noise and indiscriminate poison. We show its effectiveness in a variety of settings.



## **48. Optimal Transmission Power Scheduling for Networked Control System under DoS Attack**

eess.SY

**SubmitDate**: 2024-06-01    [abs](http://arxiv.org/abs/2406.00540v1) [paper-pdf](http://arxiv.org/pdf/2406.00540v1)

**Authors**: Siyi Wang, Yulong Gao, Sandra Hirche

**Abstract**: Designing networked control systems that are reliable and resilient against adversarial threats, is essential for ensuring the security of cyber-physical systems. This paper addresses the communication-control co-design problem for networked control systems under denial-of-service (DoS) attacks. In the wireless channel, a transmission power scheduler periodically determines the power level for sensory data transmission. Yet DoS attacks render data packets unavailable by disrupting the communication channel. This paper co-designs the control and power scheduling laws in the presence of DoS attacks and aims to minimize the sum of regulation control performance and transmission power consumption. Both finite- and infinite-horizon discounted cost criteria are addressed, respectively. By delving into the information structure between the controller and the power scheduler under attack, the original co-design problem is divided into two subproblems that can be solved individually without compromising optimality. The optimal control is shown to be certainty equivalent, and the optimal transmission power scheduling is solved using a dynamic programming approach. Moreover, in the infinite-horizon scenario, we analyze the performance of the designed scheduling policy and develop an upper bound of the total costs. Finally, a numerical example is provided to demonstrate the theoretical results.



## **49. Intrinsic Biologically Plausible Adversarial Robustness**

cs.LG

**SubmitDate**: 2024-06-01    [abs](http://arxiv.org/abs/2309.17348v5) [paper-pdf](http://arxiv.org/pdf/2309.17348v5)

**Authors**: Matilde Tristany Farinha, Thomas Ortner, Giorgia Dellaferrera, Benjamin Grewe, Angeliki Pantazi

**Abstract**: Artificial Neural Networks (ANNs) trained with Backpropagation (BP) excel in different daily tasks but have a dangerous vulnerability: inputs with small targeted perturbations, also known as adversarial samples, can drastically disrupt their performance. Adversarial training, a technique in which the training dataset is augmented with exemplary adversarial samples, is proven to mitigate this problem but comes at a high computational cost. In contrast to ANNs, humans are not susceptible to misclassifying these same adversarial samples. Thus, one can postulate that biologically-plausible trained ANNs might be more robust against adversarial attacks. In this work, we chose the biologically-plausible learning algorithm Present the Error to Perturb the Input To modulate Activity (PEPITA) as a case study and investigated this question through a comparative analysis with BP-trained ANNs on various computer vision tasks. We observe that PEPITA has a higher intrinsic adversarial robustness and, when adversarially trained, also has a more favorable natural-vs-adversarial performance trade-off. In particular, for the same natural accuracies on the MNIST task, PEPITA's adversarial accuracies decrease on average only by 0.26% while BP's decrease by 8.05%.



## **50. BruSLeAttack: A Query-Efficient Score-Based Black-Box Sparse Adversarial Attack**

cs.LG

Published as a conference paper at the International Conference on  Learning Representations (ICLR 2024). Code is available at  https://brusliattack.github.io/

**SubmitDate**: 2024-06-01    [abs](http://arxiv.org/abs/2404.05311v2) [paper-pdf](http://arxiv.org/pdf/2404.05311v2)

**Authors**: Viet Quoc Vo, Ehsan Abbasnejad, Damith C. Ranasinghe

**Abstract**: We study the unique, less-well understood problem of generating sparse adversarial samples simply by observing the score-based replies to model queries. Sparse attacks aim to discover a minimum number-the l0 bounded-perturbations to model inputs to craft adversarial examples and misguide model decisions. But, in contrast to query-based dense attack counterparts against black-box models, constructing sparse adversarial perturbations, even when models serve confidence score information to queries in a score-based setting, is non-trivial. Because, such an attack leads to i) an NP-hard problem; and ii) a non-differentiable search space. We develop the BruSLeAttack-a new, faster (more query-efficient) Bayesian algorithm for the problem. We conduct extensive attack evaluations including an attack demonstration against a Machine Learning as a Service (MLaaS) offering exemplified by Google Cloud Vision and robustness testing of adversarial training regimes and a recent defense against black-box attacks. The proposed attack scales to achieve state-of-the-art attack success rates and query efficiency on standard computer vision tasks such as ImageNet across different model architectures. Our artefacts and DIY attack samples are available on GitHub. Importantly, our work facilitates faster evaluation of model vulnerabilities and raises our vigilance on the safety, security and reliability of deployed systems.



