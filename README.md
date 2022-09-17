# Latest Adversarial Attack Papers
**update at 2022-09-18 06:38:19**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. Adversarial Training for High-Stakes Reliability**

cs.LG

31 pages, 6 figures, fixed incorrect citation

**SubmitDate**: 2022-09-15    [paper-pdf](http://arxiv.org/pdf/2205.01663v3)

**Authors**: Daniel M. Ziegler, Seraphina Nix, Lawrence Chan, Tim Bauman, Peter Schmidt-Nielsen, Tao Lin, Adam Scherlis, Noa Nabeshima, Ben Weinstein-Raun, Daniel de Haas, Buck Shlegeris, Nate Thomas

**Abstracts**: In the future, powerful AI systems may be deployed in high-stakes settings, where a single failure could be catastrophic. One technique for improving AI safety in high-stakes settings is adversarial training, which uses an adversary to generate examples to train on in order to achieve better worst-case performance.   In this work, we used a language generation task as a testbed for achieving high reliability through adversarial training. We created a series of adversarial training techniques -- including a tool that assists human adversaries -- to find and eliminate failures in a classifier that filters text completions suggested by a generator. In our simple "avoid injuries" task, we determined that we can set very conservative classifier thresholds without significantly impacting the quality of the filtered outputs. With our chosen thresholds, filtering with our baseline classifier decreases the rate of unsafe completions from about 2.4% to 0.003% on in-distribution data, which is near the limit of our ability to measure. We found that adversarial training significantly increased robustness to the adversarial attacks that we trained on, without affecting in-distribution performance. We hope to see further work in the high-stakes reliability setting, including more powerful tools for enhancing human adversaries and better ways to measure high levels of reliability, until we can confidently rule out the possibility of catastrophic deployment-time failures of powerful models.



## **2. How to Attack and Defend NextG Radio Access Network Slicing with Reinforcement Learning**

cs.NI

**SubmitDate**: 2022-09-15    [paper-pdf](http://arxiv.org/pdf/2101.05768v2)

**Authors**: Yi Shi, Yalin E. Sagduyu, Tugba Erpek, M. Cenk Gursoy

**Abstracts**: In this paper, reinforcement learning (RL) for network slicing is considered in NextG radio access networks, where the base station (gNodeB) allocates resource blocks (RBs) to the requests of user equipments and aims to maximize the total reward of accepted requests over time. Based on adversarial machine learning, a novel over-the-air attack is introduced to manipulate the RL algorithm and disrupt NextG network slicing. The adversary observes the spectrum and builds its own RL based surrogate model that selects which RBs to jam subject to an energy budget with the objective of maximizing the number of failed requests due to jammed RBs. By jamming the RBs, the adversary reduces the RL algorithm's reward. As this reward is used as the input to update the RL algorithm, the performance does not recover even after the adversary stops jamming. This attack is evaluated in terms of both the recovery time and the (maximum and total) reward loss, and it is shown to be much more effective than benchmark (random and myopic) jamming attacks. Different reactive and proactive defense schemes (protecting the RL algorithm's updates or misleading the adversary's learning process) are introduced to show that it is viable to defend NextG network slicing against this attack.



## **3. A Light Recipe to Train Robust Vision Transformers**

cs.CV

Code available at https://github.com/dedeswim/vits-robustness-torch

**SubmitDate**: 2022-09-15    [paper-pdf](http://arxiv.org/pdf/2209.07399v1)

**Authors**: Edoardo Debenedetti, Vikash Sehwag, Prateek Mittal

**Abstracts**: In this paper, we ask whether Vision Transformers (ViTs) can serve as an underlying architecture for improving the adversarial robustness of machine learning models against evasion attacks. While earlier works have focused on improving Convolutional Neural Networks, we show that also ViTs are highly suitable for adversarial training to achieve competitive performance. We achieve this objective using a custom adversarial training recipe, discovered using rigorous ablation studies on a subset of the ImageNet dataset. The canonical training recipe for ViTs recommends strong data augmentation, in part to compensate for the lack of vision inductive bias of attention modules, when compared to convolutions. We show that this recipe achieves suboptimal performance when used for adversarial training. In contrast, we find that omitting all heavy data augmentation, and adding some additional bag-of-tricks ($\varepsilon$-warmup and larger weight decay), significantly boosts the performance of robust ViTs. We show that our recipe generalizes to different classes of ViT architectures and large-scale models on full ImageNet-1k. Additionally, investigating the reasons for the robustness of our models, we show that it is easier to generate strong attacks during training when using our recipe and that this leads to better robustness at test time. Finally, we further study one consequence of adversarial training by proposing a way to quantify the semantic nature of adversarial perturbations and highlight its correlation with the robustness of the model. Overall, we recommend that the community should avoid translating the canonical training recipes in ViTs to robust training and rethink common training choices in the context of adversarial training.



## **4. Continuous Patrolling Games**

cs.DM

**SubmitDate**: 2022-09-15    [paper-pdf](http://arxiv.org/pdf/2008.07369v2)

**Authors**: Steve Alpern, Thuy Bui, Thomas Lidbetter, Katerina Papadaki

**Abstracts**: We study a patrolling game played on a network $Q$, considered as a metric space. The Attacker chooses a point of $Q$ (not necessarily a node) to attack during a chosen time interval of fixed duration. The Patroller chooses a unit speed path on $Q$ and intercepts the attack (and wins) if she visits the attacked point during the attack time interval. This zero-sum game models the problem of protecting roads or pipelines from an adversarial attack. The payoff to the maximizing Patroller is the probability that the attack is intercepted. Our results include the following: (i) a solution to the game for any network $Q$, as long as the time required to carry out the attack is sufficiently short, (ii) a solution to the game for all tree networks that satisfy a certain condition on their extremities, and (iii) a solution to the game for any attack duration for stars with one long arc and the remaining arcs equal in length. We present a conjecture on the solution of the game for arbitrary trees and establish it in certain cases.



## **5. Defending From Physically-Realizable Adversarial Attacks Through Internal Over-Activation Analysis**

cs.CV

**SubmitDate**: 2022-09-15    [paper-pdf](http://arxiv.org/pdf/2203.07341v2)

**Authors**: Giulio Rossolini, Federico Nesti, Fabio Brau, Alessandro Biondi, Giorgio Buttazzo

**Abstracts**: This work presents Z-Mask, a robust and effective strategy to improve the adversarial robustness of convolutional networks against physically-realizable adversarial attacks. The presented defense relies on specific Z-score analysis performed on the internal network features to detect and mask the pixels corresponding to adversarial objects in the input image. To this end, spatially contiguous activations are examined in shallow and deep layers to suggest potential adversarial regions. Such proposals are then aggregated through a multi-thresholding mechanism. The effectiveness of Z-Mask is evaluated with an extensive set of experiments carried out on models for both semantic segmentation and object detection. The evaluation is performed with both digital patches added to the input images and printed patches positioned in the real world. The obtained results confirm that Z-Mask outperforms the state-of-the-art methods in terms of both detection accuracy and overall performance of the networks under attack. Additional experiments showed that Z-Mask is also robust against possible defense-aware attacks.



## **6. Decision-based Black-box Attack Against Vision Transformers via Patch-wise Adversarial Removal**

cs.CV

**SubmitDate**: 2022-09-15    [paper-pdf](http://arxiv.org/pdf/2112.03492v2)

**Authors**: Yucheng Shi, Yahong Han, Yu-an Tan, Xiaohui Kuang

**Abstracts**: Vision transformers (ViTs) have demonstrated impressive performance and stronger adversarial robustness compared to Convolutional Neural Networks (CNNs). On the one hand, ViTs' focus on global interaction between individual patches reduces the local noise sensitivity of images. On the other hand, the neglect of noise sensitivity differences between image regions by existing decision-based attacks further compromises the efficiency of noise compression, especially for ViTs. Therefore, validating the black-box adversarial robustness of ViTs when the target model can only be queried still remains a challenging problem. In this paper, we theoretically analyze the limitations of existing decision-based attacks from the perspective of noise sensitivity difference between regions of the image, and propose a new decision-based black-box attack against ViTs, termed Patch-wise Adversarial Removal (PAR). PAR divides images into patches through a coarse-to-fine search process and compresses the noise on each patch separately. PAR records the noise magnitude and noise sensitivity of each patch and selects the patch with the highest query value for noise compression. In addition, PAR can be used as a noise initialization method for other decision-based attacks to improve the noise compression efficiency on both ViTs and CNNs without introducing additional calculations. Extensive experiments on three datasets demonstrate that PAR achieves a much lower noise magnitude with the same number of queries.



## **7. PointACL:Adversarial Contrastive Learning for Robust Point Clouds Representation under Adversarial Attack**

cs.CV

arXiv admin note: text overlap with arXiv:2109.00179 by other authors

**SubmitDate**: 2022-09-14    [paper-pdf](http://arxiv.org/pdf/2209.06971v1)

**Authors**: Junxuan Huang, Yatong An, Lu cheng, Bai Chen, Junsong Yuan, Chunming Qiao

**Abstracts**: Despite recent success of self-supervised based contrastive learning model for 3D point clouds representation, the adversarial robustness of such pre-trained models raised concerns. Adversarial contrastive learning (ACL) is considered an effective way to improve the robustness of pre-trained models. In contrastive learning, the projector is considered an effective component for removing unnecessary feature information during contrastive pretraining and most ACL works also use contrastive loss with projected feature representations to generate adversarial examples in pretraining, while "unprojected " feature representations are used in generating adversarial inputs during inference.Because of the distribution gap between projected and "unprojected" features, their models are constrained of obtaining robust feature representations for downstream tasks. We introduce a new method to generate high-quality 3D adversarial examples for adversarial training by utilizing virtual adversarial loss with "unprojected" feature representations in contrastive learning framework. We present our robust aware loss function to train self-supervised contrastive learning framework adversarially. Furthermore, we find selecting high difference points with the Difference of Normal (DoN) operator as additional input for adversarial self-supervised contrastive learning can significantly improve the adversarial robustness of the pre-trained model. We validate our method, PointACL on downstream tasks, including 3D classification and 3D segmentation with multiple datasets. It obtains comparable robust accuracy over state-of-the-art contrastive adversarial learning methods.



## **8. Finetuning Pretrained Vision-Language Models with Correlation Information Bottleneck for Robust Visual Question Answering**

cs.CV

20 pages, 4 figures, 13 tables

**SubmitDate**: 2022-09-14    [paper-pdf](http://arxiv.org/pdf/2209.06954v1)

**Authors**: Jingjing Jiang, Ziyi Liu, Nanning Zheng

**Abstracts**: Benefiting from large-scale Pretrained Vision-Language Models (VL-PMs), the performance of Visual Question Answering (VQA) has started to approach human oracle performance. However, finetuning large-scale VL-PMs with limited data for VQA usually faces overfitting and poor generalization issues, leading to a lack of robustness. In this paper, we aim to improve the robustness of VQA systems (ie, the ability of the systems to defend against input variations and human-adversarial attacks) from the perspective of Information Bottleneck when finetuning VL-PMs for VQA. Generally, internal representations obtained by VL-PMs inevitably contain irrelevant and redundant information for the downstream VQA task, resulting in statistically spurious correlations and insensitivity to input variations. To encourage representations to converge to a minimal sufficient statistic in vision-language learning, we propose the Correlation Information Bottleneck (CIB) principle, which seeks a tradeoff between representation compression and redundancy by minimizing the mutual information (MI) between the inputs and internal representations while maximizing the MI between the outputs and the representations. Meanwhile, CIB measures the internal correlations among visual and linguistic inputs and representations by a symmetrized joint MI estimation. Extensive experiments on five VQA benchmarks of input robustness and two VQA benchmarks of human-adversarial robustness demonstrate the effectiveness and superiority of the proposed CIB in improving the robustness of VQA systems.



## **9. On the interplay of adversarial robustness and architecture components: patches, convolution and attention**

cs.CV

Presented at the "New Frontiers in Adversarial Machine Learning"  Workshop at ICML 2022

**SubmitDate**: 2022-09-14    [paper-pdf](http://arxiv.org/pdf/2209.06953v1)

**Authors**: Francesco Croce, Matthias Hein

**Abstracts**: In recent years novel architecture components for image classification have been developed, starting with attention and patches used in transformers. While prior works have analyzed the influence of some aspects of architecture components on the robustness to adversarial attacks, in particular for vision transformers, the understanding of the main factors is still limited. We compare several (non)-robust classifiers with different architectures and study their properties, including the effect of adversarial training on the interpretability of the learnt features and robustness to unseen threat models. An ablation from ResNet to ConvNeXt reveals key architectural changes leading to almost $10\%$ higher $\ell_\infty$-robustness.



## **10. Robust Constrained Reinforcement Learning**

cs.LG

**SubmitDate**: 2022-09-14    [paper-pdf](http://arxiv.org/pdf/2209.06866v1)

**Authors**: Yue Wang, Fei Miao, Shaofeng Zou

**Abstracts**: Constrained reinforcement learning is to maximize the expected reward subject to constraints on utilities/costs. However, the training environment may not be the same as the test one, due to, e.g., modeling error, adversarial attack, non-stationarity, resulting in severe performance degradation and more importantly constraint violation. We propose a framework of robust constrained reinforcement learning under model uncertainty, where the MDP is not fixed but lies in some uncertainty set, the goal is to guarantee that constraints on utilities/costs are satisfied for all MDPs in the uncertainty set, and to maximize the worst-case reward performance over the uncertainty set. We design a robust primal-dual approach, and further theoretically develop guarantee on its convergence, complexity and robust feasibility. We then investigate a concrete example of $\delta$-contamination uncertainty set, design an online and model-free algorithm and theoretically characterize its sample complexity.



## **11. Certified Robustness to Word Substitution Ranking Attack for Neural Ranking Models**

cs.IR

Accepted by CIKM2022

**SubmitDate**: 2022-09-14    [paper-pdf](http://arxiv.org/pdf/2209.06691v1)

**Authors**: Chen Wu, Ruqing Zhang, Jiafeng Guo, Wei Chen, Yixing Fan, Maarten de Rijke, Xueqi Cheng

**Abstracts**: Neural ranking models (NRMs) have achieved promising results in information retrieval. NRMs have also been shown to be vulnerable to adversarial examples. A typical Word Substitution Ranking Attack (WSRA) against NRMs was proposed recently, in which an attacker promotes a target document in rankings by adding human-imperceptible perturbations to its text. This raises concerns when deploying NRMs in real-world applications. Therefore, it is important to develop techniques that defend against such attacks for NRMs. In empirical defenses adversarial examples are found during training and used to augment the training set. However, such methods offer no theoretical guarantee on the models' robustness and may eventually be broken by other sophisticated WSRAs. To escape this arms race, rigorous and provable certified defense methods for NRMs are needed.   To this end, we first define the \textit{Certified Top-$K$ Robustness} for ranking models since users mainly care about the top ranked results in real-world scenarios. A ranking model is said to be Certified Top-$K$ Robust on a ranked list when it is guaranteed to keep documents that are out of the top $K$ away from the top $K$ under any attack. Then, we introduce a Certified Defense method, named CertDR, to achieve certified top-$K$ robustness against WSRA, based on the idea of randomized smoothing. Specifically, we first construct a smoothed ranker by applying random word substitutions on the documents, and then leverage the ranking property jointly with the statistical property of the ensemble to provably certify top-$K$ robustness. Extensive experiments on two representative web search datasets demonstrate that CertDR can significantly outperform state-of-the-art empirical defense methods for ranking models.



## **12. Order-Disorder: Imitation Adversarial Attacks for Black-box Neural Ranking Models**

cs.IR

15 pages, 4 figures, accepted by ACM CCS 2022

**SubmitDate**: 2022-09-14    [paper-pdf](http://arxiv.org/pdf/2209.06506v1)

**Authors**: Jiawei Liu, Yangyang Kang, Di Tang, Kaisong Song, Changlong Sun, Xiaofeng Wang, Wei Lu, Xiaozhong Liu

**Abstracts**: Neural text ranking models have witnessed significant advancement and are increasingly being deployed in practice. Unfortunately, they also inherit adversarial vulnerabilities of general neural models, which have been detected but remain underexplored by prior studies. Moreover, the inherit adversarial vulnerabilities might be leveraged by blackhat SEO to defeat better-protected search engines. In this study, we propose an imitation adversarial attack on black-box neural passage ranking models. We first show that the target passage ranking model can be transparentized and imitated by enumerating critical queries/candidates and then train a ranking imitation model. Leveraging the ranking imitation model, we can elaborately manipulate the ranking results and transfer the manipulation attack to the target ranking model. For this purpose, we propose an innovative gradient-based attack method, empowered by the pairwise objective function, to generate adversarial triggers, which causes premeditated disorderliness with very few tokens. To equip the trigger camouflages, we add the next sentence prediction loss and the language model fluency constraint to the objective function. Experimental results on passage ranking demonstrate the effectiveness of the ranking imitation attack model and adversarial triggers against various SOTA neural ranking models. Furthermore, various mitigation analyses and human evaluation show the effectiveness of camouflages when facing potential mitigation approaches. To motivate other scholars to further investigate this novel and important problem, we make the experiment data and code publicly available.



## **13. Targeting interventions for displacement minimization in opinion dynamics**

cs.SI

**SubmitDate**: 2022-09-14    [paper-pdf](http://arxiv.org/pdf/2209.06481v1)

**Authors**: Luca Damonte, Giacomo Como, Fabio Fagnani

**Abstracts**: Social influence is largely recognized as a key factor in opinion formation processes. Recently, the role of external forces in inducing opinion displacement and polarization in social networks has attracted significant attention. This is in particular motivated by the necessity to understand and possibly prevent interference phenomena during political campaigns and elections. In this paper, we formulate and solve a targeted intervention problem for opinion displacement minimization on a social network. Specifically, we consider a min-max problem whereby a social planner (the defender) aims at selecting the optimal network intervention within her given budget constraint in order to minimize the opinion displacement in the system that an adversary (the attacker) is instead trying to maximize. Our results show that the optimal intervention of the defender has two regimes. For large enough budget, the optimal intervention of the social planner acts on all nodes proportionally to a new notion of network centrality. For lower budget values, such optimal intervention has a more delicate structure and is rather concentrated on a few target individuals.



## **14. Private Eye: On the Limits of Textual Screen Peeking via Eyeglass Reflections in Video Conferencing**

cs.CR

**SubmitDate**: 2022-09-14    [paper-pdf](http://arxiv.org/pdf/2205.03971v2)

**Authors**: Yan Long, Chen Yan, Shilin Xiao, Shivan Prasad, Wenyuan Xu, Kevin Fu

**Abstracts**: Using mathematical modeling and human subjects experiments, this research explores the extent to which emerging webcams might leak recognizable textual and graphical information gleaming from eyeglass reflections captured by webcams. The primary goal of our work is to measure, compute, and predict the factors, limits, and thresholds of recognizability as webcam technology evolves in the future. Our work explores and characterizes the viable threat models based on optical attacks using multi-frame super resolution techniques on sequences of video frames. Our models and experimental results in a controlled lab setting show it is possible to reconstruct and recognize with over 75% accuracy on-screen texts that have heights as small as 10 mm with a 720p webcam. We further apply this threat model to web textual contents with varying attacker capabilities to find thresholds at which text becomes recognizable. Our user study with 20 participants suggests present-day 720p webcams are sufficient for adversaries to reconstruct textual content on big-font websites. Our models further show that the evolution towards 4K cameras will tip the threshold of text leakage to reconstruction of most header texts on popular websites. Besides textual targets, a case study on recognizing a closed-world dataset of Alexa top 100 websites with 720p webcams shows a maximum recognition accuracy of 94% with 10 participants even without using machine-learning models. Our research proposes near-term mitigations including a software prototype that users can use to blur the eyeglass areas of their video streams. For possible long-term defenses, we advocate an individual reflection testing procedure to assess threats under various settings, and justify the importance of following the principle of least privilege for privacy-sensitive scenarios.



## **15. TSFool: Crafting High-quality Adversarial Time Series through Multi-objective Optimization to Fool Recurrent Neural Network Classifiers**

cs.LG

9 pages, 5 figures

**SubmitDate**: 2022-09-14    [paper-pdf](http://arxiv.org/pdf/2209.06388v1)

**Authors**: Yanyun Wang, Dehui Du, Yuanhao Liu

**Abstracts**: Deep neural network (DNN) classifiers are vulnerable to adversarial attacks. Although the existing gradient-based attacks have achieved good performance in feed-forward model and image recognition tasks, the extension for time series classification in the recurrent neural network (RNN) remains a dilemma, because the cyclical structure of RNN prevents direct model differentiation and the visual sensitivity to perturbations of time series data challenges the traditional local optimization objective to minimize perturbation. In this paper, an efficient and widely applicable approach called TSFool for crafting high-quality adversarial time series for the RNN classifier is proposed. We propose a novel global optimization objective named Camouflage Coefficient to consider how well the adversarial samples hide in class clusters, and accordingly redefine the high-quality adversarial attack as a multi-objective optimization problem. We also propose a new idea to use intervalized weighted finite automata (IWFA) to capture deeply embedded vulnerable samples having otherness between features and latent manifold to guide the approximation to the optimization solution. Experiments on 22 UCR datasets are conducted to confirm that TSFool is a widely effective, efficient and high-quality approach with 93.22% less local perturbation, 32.33% better global camouflage, and 1.12 times speedup to existing methods.



## **16. PINCH: An Adversarial Extraction Attack Framework for Deep Learning Models**

cs.CR

15 pages, 11 figures, 2 tables

**SubmitDate**: 2022-09-13    [paper-pdf](http://arxiv.org/pdf/2209.06300v1)

**Authors**: William Hackett, Stefan Trawicki, Zhengxin Yu, Neeraj Suri, Peter Garraghan

**Abstracts**: Deep Learning (DL) models increasingly power a diversity of applications. Unfortunately, this pervasiveness also makes them attractive targets for extraction attacks which can steal the architecture, parameters, and hyper-parameters of a targeted DL model. Existing extraction attack studies have observed varying levels of attack success for different DL models and datasets, yet the underlying cause(s) behind their susceptibility often remain unclear. Ascertaining such root-cause weaknesses would help facilitate secure DL systems, though this requires studying extraction attacks in a wide variety of scenarios to identify commonalities across attack success and DL characteristics. The overwhelmingly high technical effort and time required to understand, implement, and evaluate even a single attack makes it infeasible to explore the large number of unique extraction attack scenarios in existence, with current frameworks typically designed to only operate for specific attack types, datasets and hardware platforms. In this paper we present PINCH: an efficient and automated extraction attack framework capable of deploying and evaluating multiple DL models and attacks across heterogeneous hardware platforms. We demonstrate the effectiveness of PINCH by empirically evaluating a large number of previously unexplored extraction attack scenarios, as well as secondary attack staging. Our key findings show that 1) multiple characteristics affect extraction attack success spanning DL model architecture, dataset complexity, hardware, attack type, and 2) partially successful extraction attacks significantly enhance the success of further adversarial attack staging.



## **17. Certified Defences Against Adversarial Patch Attacks on Semantic Segmentation**

cs.CV

**SubmitDate**: 2022-09-13    [paper-pdf](http://arxiv.org/pdf/2209.05980v1)

**Authors**: Maksym Yatsura, Kaspar Sakmann, N. Grace Hua, Matthias Hein, Jan Hendrik Metzen

**Abstracts**: Adversarial patch attacks are an emerging security threat for real world deep learning applications. We present Demasked Smoothing, the first approach (up to our knowledge) to certify the robustness of semantic segmentation models against this threat model. Previous work on certifiably defending against patch attacks has mostly focused on image classification task and often required changes in the model architecture and additional training which is undesirable and computationally expensive. In Demasked Smoothing, any segmentation model can be applied without particular training, fine-tuning, or restriction of the architecture. Using different masking strategies, Demasked Smoothing can be applied both for certified detection and certified recovery. In extensive experiments we show that Demasked Smoothing can on average certify 64% of the pixel predictions for a 1% patch in the detection task and 48% against a 0.5% patch for the recovery task on the ADE20K dataset.



## **18. Adversarial Inter-Group Link Injection Degrades the Fairness of Graph Neural Networks**

cs.LG

A shorter version of this work has been accepted by IEEE ICDM 2022

**SubmitDate**: 2022-09-13    [paper-pdf](http://arxiv.org/pdf/2209.05957v1)

**Authors**: Hussain Hussain, Meng Cao, Sandipan Sikdar, Denis Helic, Elisabeth Lex, Markus Strohmaier, Roman Kern

**Abstracts**: We present evidence for the existence and effectiveness of adversarial attacks on graph neural networks (GNNs) that aim to degrade fairness. These attacks can disadvantage a particular subgroup of nodes in GNN-based node classification, where nodes of the underlying network have sensitive attributes, such as race or gender. We conduct qualitative and experimental analyses explaining how adversarial link injection impairs the fairness of GNN predictions. For example, an attacker can compromise the fairness of GNN-based node classification by injecting adversarial links between nodes belonging to opposite subgroups and opposite class labels. Our experiments on empirical datasets demonstrate that adversarial fairness attacks can significantly degrade the fairness of GNN predictions (attacks are effective) with a low perturbation rate (attacks are efficient) and without a significant drop in accuracy (attacks are deceptive). This work demonstrates the vulnerability of GNN models to adversarial fairness attacks. We hope our findings raise awareness about this issue in our community and lay a foundation for the future development of GNN models that are more robust to such attacks.



## **19. An Evolutionary, Gradient-Free, Query-Efficient, Black-Box Algorithm for Generating Adversarial Instances in Deep Networks**

cs.CV

**SubmitDate**: 2022-09-13    [paper-pdf](http://arxiv.org/pdf/2208.08297v2)

**Authors**: Raz Lapid, Zvika Haramaty, Moshe Sipper

**Abstracts**: Deep neural networks (DNNs) are sensitive to adversarial data in a variety of scenarios, including the black-box scenario, where the attacker is only allowed to query the trained model and receive an output. Existing black-box methods for creating adversarial instances are costly, often using gradient estimation or training a replacement network. This paper introduces \textbf{Qu}ery-Efficient \textbf{E}volutiona\textbf{ry} \textbf{Attack}, \textit{QuEry Attack}, an untargeted, score-based, black-box attack. QuEry Attack is based on a novel objective function that can be used in gradient-free optimization problems. The attack only requires access to the output logits of the classifier and is thus not affected by gradient masking. No additional information is needed, rendering our method more suitable to real-life situations. We test its performance with three different state-of-the-art models -- Inception-v3, ResNet-50, and VGG-16-BN -- against three benchmark datasets: MNIST, CIFAR10 and ImageNet. Furthermore, we evaluate QuEry Attack's performance on non-differential transformation defenses and state-of-the-art robust models. Our results demonstrate the superior performance of QuEry Attack, both in terms of accuracy score and query efficiency.



## **20. Bayesian Pseudo Labels: Expectation Maximization for Robust and Efficient Semi-Supervised Segmentation**

cs.CV

MICCAI 2022 (Early accept, Student Travel Award)

**SubmitDate**: 2022-09-13    [paper-pdf](http://arxiv.org/pdf/2208.04435v3)

**Authors**: Mou-Cheng Xu, Yukun Zhou, Chen Jin, Marius de Groot, Daniel C. Alexander, Neil P. Oxtoby, Yipeng Hu, Joseph Jacob

**Abstracts**: This paper concerns pseudo labelling in segmentation. Our contribution is fourfold. Firstly, we present a new formulation of pseudo-labelling as an Expectation-Maximization (EM) algorithm for clear statistical interpretation. Secondly, we propose a semi-supervised medical image segmentation method purely based on the original pseudo labelling, namely SegPL. We demonstrate SegPL is a competitive approach against state-of-the-art consistency regularisation based methods on semi-supervised segmentation on a 2D multi-class MRI brain tumour segmentation task and a 3D binary CT lung vessel segmentation task. The simplicity of SegPL allows less computational cost comparing to prior methods. Thirdly, we demonstrate that the effectiveness of SegPL may originate from its robustness against out-of-distribution noises and adversarial attacks. Lastly, under the EM framework, we introduce a probabilistic generalisation of SegPL via variational inference, which learns a dynamic threshold for pseudo labelling during the training. We show that SegPL with variational inference can perform uncertainty estimation on par with the gold-standard method Deep Ensemble.



## **21. Adversarial Coreset Selection for Efficient Robust Training**

cs.LG

Extended version of the ECCV2022 paper: arXiv:2112.00378. arXiv admin  note: substantial text overlap with arXiv:2112.00378

**SubmitDate**: 2022-09-13    [paper-pdf](http://arxiv.org/pdf/2209.05785v1)

**Authors**: Hadi M. Dolatabadi, Sarah Erfani, Christopher Leckie

**Abstracts**: Neural networks are vulnerable to adversarial attacks: adding well-crafted, imperceptible perturbations to their input can modify their output. Adversarial training is one of the most effective approaches to training robust models against such attacks. Unfortunately, this method is much slower than vanilla training of neural networks since it needs to construct adversarial examples for the entire training data at every iteration. By leveraging the theory of coreset selection, we show how selecting a small subset of training data provides a principled approach to reducing the time complexity of robust training. To this end, we first provide convergence guarantees for adversarial coreset selection. In particular, we show that the convergence bound is directly related to how well our coresets can approximate the gradient computed over the entire training data. Motivated by our theoretical analysis, we propose using this gradient approximation error as our adversarial coreset selection objective to reduce the training set size effectively. Once built, we run adversarial training over this subset of the training data. Unlike existing methods, our approach can be adapted to a wide variety of training objectives, including TRADES, $\ell_p$-PGD, and Perceptual Adversarial Training. We conduct extensive experiments to demonstrate that our approach speeds up adversarial training by 2-3 times while experiencing a slight degradation in the clean and robust accuracy.



## **22. Adaptive Perturbation Generation for Multiple Backdoors Detection**

cs.CV

7 pages, 5 figures

**SubmitDate**: 2022-09-13    [paper-pdf](http://arxiv.org/pdf/2209.05244v2)

**Authors**: Yuhang Wang, Huafeng Shi, Rui Min, Ruijia Wu, Siyuan Liang, Yichao Wu, Ding Liang, Aishan Liu

**Abstracts**: Extensive evidence has demonstrated that deep neural networks (DNNs) are vulnerable to backdoor attacks, which motivates the development of backdoor detection methods. Existing backdoor detection methods are typically tailored for backdoor attacks with individual specific types (e.g., patch-based or perturbation-based). However, adversaries are likely to generate multiple types of backdoor attacks in practice, which challenges the current detection strategies. Based on the fact that adversarial perturbations are highly correlated with trigger patterns, this paper proposes the Adaptive Perturbation Generation (APG) framework to detect multiple types of backdoor attacks by adaptively injecting adversarial perturbations. Since different trigger patterns turn out to show highly diverse behaviors under the same adversarial perturbations, we first design the global-to-local strategy to fit the multiple types of backdoor triggers via adjusting the region and budget of attacks. To further increase the efficiency of perturbation injection, we introduce a gradient-guided mask generation strategy to search for the optimal regions for adversarial attacks. Extensive experiments conducted on multiple datasets (CIFAR-10, GTSRB, Tiny-ImageNet) demonstrate that our method outperforms state-of-the-art baselines by large margins(+12%).



## **23. A Tale of HodgeRank and Spectral Method: Target Attack Against Rank Aggregation Is the Fixed Point of Adversarial Game**

cs.LG

33 pages,  https://github.com/alphaprime/Target_Attack_Rank_Aggregation

**SubmitDate**: 2022-09-13    [paper-pdf](http://arxiv.org/pdf/2209.05742v1)

**Authors**: Ke Ma, Qianqian Xu, Jinshan Zeng, Guorong Li, Xiaochun Cao, Qingming Huang

**Abstracts**: Rank aggregation with pairwise comparisons has shown promising results in elections, sports competitions, recommendations, and information retrieval. However, little attention has been paid to the security issue of such algorithms, in contrast to numerous research work on the computational and statistical characteristics. Driven by huge profits, the potential adversary has strong motivation and incentives to manipulate the ranking list. Meanwhile, the intrinsic vulnerability of the rank aggregation methods is not well studied in the literature. To fully understand the possible risks, we focus on the purposeful adversary who desires to designate the aggregated results by modifying the pairwise data in this paper. From the perspective of the dynamical system, the attack behavior with a target ranking list is a fixed point belonging to the composition of the adversary and the victim. To perform the targeted attack, we formulate the interaction between the adversary and the victim as a game-theoretic framework consisting of two continuous operators while Nash equilibrium is established. Then two procedures against HodgeRank and RankCentrality are constructed to produce the modification of the original data. Furthermore, we prove that the victims will produce the target ranking list once the adversary masters the complete information. It is noteworthy that the proposed methods allow the adversary only to hold incomplete information or imperfect feedback and perform the purposeful attack. The effectiveness of the suggested target attack strategies is demonstrated by a series of toy simulations and several real-world data experiments. These experimental results show that the proposed methods could achieve the attacker's goal in the sense that the leading candidate of the perturbed ranking list is the designated one by the adversary.



## **24. Sample Complexity of an Adversarial Attack on UCB-based Best-arm Identification Policy**

cs.LG

**SubmitDate**: 2022-09-13    [paper-pdf](http://arxiv.org/pdf/2209.05692v1)

**Authors**: Varsha Pendyala

**Abstracts**: In this work I study the problem of adversarial perturbations to rewards, in a Multi-armed bandit (MAB) setting. Specifically, I focus on an adversarial attack to a UCB type best-arm identification policy applied to a stochastic MAB. The UCB attack presented in [1] results in pulling a target arm K very often. I used the attack model of [1] to derive the sample complexity required for selecting target arm K as the best arm. I have proved that the stopping condition of UCB based best-arm identification algorithm given in [2], can be achieved by the target arm K in T rounds, where T depends only on the total number of arms and $\sigma$ parameter of $\sigma^2-$ sub-Gaussian random rewards of the arms.



## **25. Replay-based Recovery for Autonomous Robotic Vehicles from Sensor Deception Attacks**

cs.RO

**SubmitDate**: 2022-09-13    [paper-pdf](http://arxiv.org/pdf/2209.04554v2)

**Authors**: Pritam Dash, Guanpeng Li, Mehdi Karimibiuki, Karthik Pattabiraman

**Abstracts**: Sensors are crucial for autonomous operation in robotic vehicles (RV). Physical attacks on sensors such as sensor tampering or spoofing can feed erroneous values to RVs through physical channels, which results in mission failures. In this paper, we present DeLorean, a comprehensive diagnosis and recovery framework for securing autonomous RVs from physical attacks. We consider a strong form of physical attack called sensor deception attacks (SDAs), in which the adversary targets multiple sensors of different types simultaneously (even including all sensors). Under SDAs, DeLorean inspects the attack induced errors, identifies the targeted sensors, and prevents the erroneous sensor inputs from being used in RV's feedback control loop. DeLorean replays historic state information in the feedback control loop and recovers the RV from attacks. Our evaluation on four real and two simulated RVs shows that DeLorean can recover RVs from different attacks, and ensure mission success in 94% of the cases (on average), without any crashes. DeLorean incurs low performance, memory and battery overheads.



## **26. Boosting Robustness Verification of Semantic Feature Neighborhoods**

cs.LG

**SubmitDate**: 2022-09-12    [paper-pdf](http://arxiv.org/pdf/2209.05446v1)

**Authors**: Anan Kabaha, Dana Drachsler-Cohen

**Abstracts**: Deep neural networks have been shown to be vulnerable to adversarial attacks that perturb inputs based on semantic features. Existing robustness analyzers can reason about semantic feature neighborhoods to increase the networks' reliability. However, despite the significant progress in these techniques, they still struggle to scale to deep networks and large neighborhoods. In this work, we introduce VeeP, an active learning approach that splits the verification process into a series of smaller verification steps, each is submitted to an existing robustness analyzer. The key idea is to build on prior steps to predict the next optimal step. The optimal step is predicted by estimating the certification velocity and sensitivity via parametric regression. We evaluate VeeP on MNIST, Fashion-MNIST, CIFAR-10 and ImageNet and show that it can analyze neighborhoods of various features: brightness, contrast, hue, saturation, and lightness. We show that, on average, given a 90 minute timeout, VeeP verifies 96% of the maximally certifiable neighborhoods within 29 minutes, while existing splitting approaches verify, on average, 73% of the maximally certifiable neighborhoods within 58 minutes.



## **27. Bilevel Optimization with a Lower-level Contraction: Optimal Sample Complexity without Warm-Start**

stat.ML

35 pages, 2 figures. Code at  https://github.com/CSML-IIT-UCL/bioptexps

**SubmitDate**: 2022-09-12    [paper-pdf](http://arxiv.org/pdf/2202.03397v2)

**Authors**: Riccardo Grazzi, Massimiliano Pontil, Saverio Salzo

**Abstracts**: We analyze a general class of bilevel problems, in which the upper-level problem consists in the minimization of a smooth objective function and the lower-level problem is to find the fixed point of a smooth contraction map. This type of problems include instances of meta-learning, equilibrium models, hyperparameter optimization and data poisoning adversarial attacks. Several recent works have proposed algorithms which warm-start the lower level problem, i.e. they use the previous lower-level approximate solution as a staring point for the lower-level solver. This warm-start procedure allows one to improve the sample complexity in both the stochastic and deterministic settings, achieving in some cases the order-wise optimal sample complexity. However, there are situations, e.g., meta learning and equilibrium models, in which the warm-start procedure is not well-suited or ineffective. In this work we show that without warm-start, it is still possible to achieve order-wise optimal or near-optimal sample complexity. In particular, we propose a simple method which uses stochastic fixed point iterations at the lower-level and projected inexact gradient descent at the upper-level, that reaches an $\epsilon$-stationary point using $O(\epsilon^{-2})$ and $\tilde{O}(\epsilon^{-1})$ samples for the stochastic and the deterministic setting, respectively. Finally, compared to methods using warm-start, our approach yields a simpler analysis that does not need to study the coupled interactions between the upper-level and lower-level iterates



## **28. Fiat-Shamir for Proofs Lacks a Proof Even in the Presence of Shared Entanglement**

quant-ph

62 pages, 2 figures

**SubmitDate**: 2022-09-12    [paper-pdf](http://arxiv.org/pdf/2204.02265v2)

**Authors**: Frédéric Dupuis, Philippe Lamontagne, Louis Salvail

**Abstracts**: We explore the cryptographic power of arbitrary shared physical resources. The most general such resource is access to a fresh entangled quantum state at the outset of each protocol execution. We call this the Common Reference Quantum State (CRQS) model, in analogy to the well-known Common Reference String (CRS). The CRQS model is a natural generalization of the CRS model but appears to be more powerful: in the two-party setting, a CRQS can sometimes exhibit properties associated with a Random Oracle queried once by measuring a maximally entangled state in one of many mutually unbiased bases. We formalize this notion as a Weak One-Time Random Oracle (WOTRO), where we only ask of the $m$--bit output to have some randomness when conditioned on the $n$--bit input.   We show that when $n-m\in\omega(\lg n)$, any protocol for WOTRO in the CRQS model can be attacked by an (inefficient) adversary. Moreover, our adversary is efficiently simulatable, which rules out the possibility of proving the computational security of a scheme by a black-box reduction to a cryptographic game assumption. On the other hand, we introduce a non-game quantum assumption for hash functions that implies WOTRO in the CRQ\$ model (where the CRQS consists only of EPR pairs). We first build a statistically secure WOTRO protocol where $m=n$, then hash the output.   The impossibility of WOTRO has the following consequences. First, we show the black-box impossibility of a quantum Fiat-Shamir transform, extending the impossibility result of Bitansky et al. (TCC '13) to the CRQS model. Second, we show a black-box impossibility result for a strenghtened version of quantum lightning (Zhandry, Eurocrypt '19) where quantum bolts have an additional parameter that cannot be changed without generating new bolts.



## **29. A Survey of Machine Unlearning**

cs.LG

fixed overlaps

**SubmitDate**: 2022-09-12    [paper-pdf](http://arxiv.org/pdf/2209.02299v4)

**Authors**: Thanh Tam Nguyen, Thanh Trung Huynh, Phi Le Nguyen, Alan Wee-Chung Liew, Hongzhi Yin, Quoc Viet Hung Nguyen

**Abstracts**: Computer systems hold a large amount of personal data over decades. On the one hand, such data abundance allows breakthroughs in artificial intelligence (AI), especially machine learning (ML) models. On the other hand, it can threaten the privacy of users and weaken the trust between humans and AI. Recent regulations require that private information about a user can be removed from computer systems in general and from ML models in particular upon request (e.g. the "right to be forgotten"). While removing data from back-end databases should be straightforward, it is not sufficient in the AI context as ML models often "remember" the old data. Existing adversarial attacks proved that we can learn private membership or attributes of the training data from the trained models. This phenomenon calls for a new paradigm, namely machine unlearning, to make ML models forget about particular data. It turns out that recent works on machine unlearning have not been able to solve the problem completely due to the lack of common frameworks and resources. In this survey paper, we seek to provide a thorough investigation of machine unlearning in its definitions, scenarios, mechanisms, and applications. Specifically, as a categorical collection of state-of-the-art research, we hope to provide a broad reference for those seeking a primer on machine unlearning and its various formulations, design requirements, removal requests, algorithms, and uses in a variety of ML applications. Furthermore, we hope to outline key findings and trends in the paradigm as well as highlight new areas of research that have yet to see the application of machine unlearning, but could nonetheless benefit immensely. We hope this survey provides a valuable reference for ML researchers as well as those seeking to innovate privacy technologies. Our resources are at https://github.com/tamlhp/awesome-machine-unlearning.



## **30. GRNN: Generative Regression Neural Network -- A Data Leakage Attack for Federated Learning**

cs.LG

The source code can be found at: https://github.com/Rand2AI/GRNN

**SubmitDate**: 2022-09-12    [paper-pdf](http://arxiv.org/pdf/2105.00529v3)

**Authors**: Hanchi Ren, Jingjing Deng, Xianghua Xie

**Abstracts**: Data privacy has become an increasingly important issue in Machine Learning (ML), where many approaches have been developed to tackle this challenge, e.g. cryptography (Homomorphic Encryption (HE), Differential Privacy (DP), etc.) and collaborative training (Secure Multi-Party Computation (MPC), Distributed Learning and Federated Learning (FL)). These techniques have a particular focus on data encryption or secure local computation. They transfer the intermediate information to the third party to compute the final result. Gradient exchanging is commonly considered to be a secure way of training a robust model collaboratively in Deep Learning (DL). However, recent researches have demonstrated that sensitive information can be recovered from the shared gradient. Generative Adversarial Network (GAN), in particular, has shown to be effective in recovering such information. However, GAN based techniques require additional information, such as class labels which are generally unavailable for privacy-preserved learning. In this paper, we show that, in the FL system, image-based privacy data can be easily recovered in full from the shared gradient only via our proposed Generative Regression Neural Network (GRNN). We formulate the attack to be a regression problem and optimize two branches of the generative model by minimizing the distance between gradients. We evaluate our method on several image classification tasks. The results illustrate that our proposed GRNN outperforms state-of-the-art methods with better stability, stronger robustness, and higher accuracy. It also has no convergence requirement to the global FL model. Moreover, we demonstrate information leakage using face re-identification. Some defense strategies are also discussed in this work.



## **31. Semantic-Preserving Adversarial Code Comprehension**

cs.CL

Accepted by COLING 2022

**SubmitDate**: 2022-09-12    [paper-pdf](http://arxiv.org/pdf/2209.05130v1)

**Authors**: Yiyang Li, Hongqiu Wu, Hai Zhao

**Abstracts**: Based on the tremendous success of pre-trained language models (PrLMs) for source code comprehension tasks, current literature studies either ways to further improve the performance (generalization) of PrLMs, or their robustness against adversarial attacks. However, they have to compromise on the trade-off between the two aspects and none of them consider improving both sides in an effective and practical way. To fill this gap, we propose Semantic-Preserving Adversarial Code Embeddings (SPACE) to find the worst-case semantic-preserving attacks while forcing the model to predict the correct labels under these worst cases. Experiments and analysis demonstrate that SPACE can stay robust against state-of-the-art attacks while boosting the performance of PrLMs for code.



## **32. Passive Triangulation Attack on ORide**

cs.CR

**SubmitDate**: 2022-09-12    [paper-pdf](http://arxiv.org/pdf/2208.12216v2)

**Authors**: Shyam Murthy, Srinivas Vivek

**Abstracts**: Privacy preservation in Ride Hailing Services is intended to protect privacy of drivers and riders. ORide is one of the early RHS proposals published at USENIX Security Symposium 2017. In the ORide protocol, riders and drivers, operating in a zone, encrypt their locations using a Somewhat Homomorphic Encryption scheme (SHE) and forward them to the Service Provider (SP). SP homomorphically computes the squared Euclidean distance between riders and available drivers. Rider receives the encrypted distances and selects the optimal rider after decryption. In order to prevent a triangulation attack, SP randomly permutes the distances before sending them to the rider. In this work, we use propose a passive attack that uses triangulation to determine coordinates of all participating drivers whose permuted distances are available from the points of view of multiple honest-but-curious adversary riders. An attack on ORide was published at SAC 2021. The same paper proposes a countermeasure using noisy Euclidean distances to thwart their attack. We extend our attack to determine locations of drivers when given their permuted and noisy Euclidean distances from multiple points of reference, where the noise perturbation comes from a uniform distribution. We conduct experiments with different number of drivers and for different perturbation values. Our experiments show that we can determine locations of all drivers participating in the ORide protocol. For the perturbed distance version of the ORide protocol, our algorithm reveals locations of about 25% to 50% of participating drivers. Our algorithm runs in time polynomial in number of drivers.



## **33. CARE: Certifiably Robust Learning with Reasoning via Variational Inference**

cs.LG

**SubmitDate**: 2022-09-12    [paper-pdf](http://arxiv.org/pdf/2209.05055v1)

**Authors**: Jiawei Zhang, Linyi Li, Ce Zhang, Bo Li

**Abstracts**: Despite great recent advances achieved by deep neural networks (DNNs), they are often vulnerable to adversarial attacks. Intensive research efforts have been made to improve the robustness of DNNs; however, most empirical defenses can be adaptively attacked again, and the theoretically certified robustness is limited, especially on large-scale datasets. One potential root cause of such vulnerabilities for DNNs is that although they have demonstrated powerful expressiveness, they lack the reasoning ability to make robust and reliable predictions. In this paper, we aim to integrate domain knowledge to enable robust learning with the reasoning paradigm. In particular, we propose a certifiably robust learning with reasoning pipeline (CARE), which consists of a learning component and a reasoning component. Concretely, we use a set of standard DNNs to serve as the learning component to make semantic predictions, and we leverage the probabilistic graphical models, such as Markov logic networks (MLN), to serve as the reasoning component to enable knowledge/logic reasoning. However, it is known that the exact inference of MLN (reasoning) is #P-complete, which limits the scalability of the pipeline. To this end, we propose to approximate the MLN inference via variational inference based on an efficient expectation maximization algorithm. In particular, we leverage graph convolutional networks (GCNs) to encode the posterior distribution during variational inference and update the parameters of GCNs (E-step) and the weights of knowledge rules in MLN (M-step) iteratively. We conduct extensive experiments on different datasets and show that CARE achieves significantly higher certified robustness compared with the state-of-the-art baselines. We additionally conducted different ablation studies to demonstrate the empirical robustness of CARE and the effectiveness of different knowledge integration.



## **34. GFCL: A GRU-based Federated Continual Learning Framework against Data Poisoning Attacks in IoV**

cs.LG

11 pages, 12 figures, 3 tables; This work has been submitted to the  IEEE Transactions on Vehicular Technology for possible publication. Copyright  may be transferred without notice, after which this version may no longer be  accessible

**SubmitDate**: 2022-09-12    [paper-pdf](http://arxiv.org/pdf/2204.11010v2)

**Authors**: Anum Talpur, Mohan Gurusamy

**Abstracts**: Integration of machine learning (ML) in 5G-based Internet of Vehicles (IoV) networks has enabled intelligent transportation and smart traffic management. Nonetheless, the security against adversarial poisoning attacks is also increasingly becoming a challenging task. Specifically, Deep Reinforcement Learning (DRL) is one of the widely used ML designs in IoV applications. The standard ML security techniques are not effective in DRL where the algorithm learns to solve sequential decision-making through continuous interaction with the environment, and the environment is time-varying, dynamic, and mobile. In this paper, we propose a Gated Recurrent Unit (GRU)-based federated continual learning (GFCL) anomaly detection framework against Sybil-based data poisoning attacks in IoV. The objective is to present a lightweight and scalable framework that learns and detects the illegitimate behavior without having a-priori training dataset consisting of attack samples. We use GRU to predict a future data sequence to analyze and detect illegitimate behavior from vehicles in a federated learning-based distributed manner. We investigate the performance of our framework using real-world vehicle mobility traces. The results demonstrate the effectiveness of our proposed solution in terms of different performance metrics.



## **35. Generate novel and robust samples from data: accessible sharing without privacy concerns**

cs.LG

**SubmitDate**: 2022-09-12    [paper-pdf](http://arxiv.org/pdf/2209.06113v1)

**Authors**: David Banh, Alan Huang

**Abstracts**: Generating new samples from data sets can mitigate extra expensive operations, increased invasive procedures, and mitigate privacy issues. These novel samples that are statistically robust can be used as a temporary and intermediate replacement when privacy is a concern. This method can enable better data sharing practices without problems relating to identification issues or biases that are flaws for an adversarial attack.



## **36. Resisting Deep Learning Models Against Adversarial Attack Transferability via Feature Randomization**

cs.CR

**SubmitDate**: 2022-09-11    [paper-pdf](http://arxiv.org/pdf/2209.04930v1)

**Authors**: Ehsan Nowroozi, Mohammadreza Mohammadi, Pargol Golmohammadi, Yassine Mekdad, Mauro Conti, Selcuk Uluagac

**Abstracts**: In the past decades, the rise of artificial intelligence has given us the capabilities to solve the most challenging problems in our day-to-day lives, such as cancer prediction and autonomous navigation. However, these applications might not be reliable if not secured against adversarial attacks. In addition, recent works demonstrated that some adversarial examples are transferable across different models. Therefore, it is crucial to avoid such transferability via robust models that resist adversarial manipulations. In this paper, we propose a feature randomization-based approach that resists eight adversarial attacks targeting deep learning models in the testing phase. Our novel approach consists of changing the training strategy in the target network classifier and selecting random feature samples. We consider the attacker with a Limited-Knowledge and Semi-Knowledge conditions to undertake the most prevalent types of adversarial attacks. We evaluate the robustness of our approach using the well-known UNSW-NB15 datasets that include realistic and synthetic attacks. Afterward, we demonstrate that our strategy outperforms the existing state-of-the-art approach, such as the Most Powerful Attack, which consists of fine-tuning the network model against specific adversarial attacks. Finally, our experimental results show that our methodology can secure the target network and resists adversarial attack transferability by over 60%.



## **37. Detecting Adversarial Perturbations in Multi-Task Perception**

cs.CV

Accepted at IROS 2022

**SubmitDate**: 2022-09-11    [paper-pdf](http://arxiv.org/pdf/2203.01177v2)

**Authors**: Marvin Klingner, Varun Ravi Kumar, Senthil Yogamani, Andreas Bär, Tim Fingscheidt

**Abstracts**: While deep neural networks (DNNs) achieve impressive performance on environment perception tasks, their sensitivity to adversarial perturbations limits their use in practical applications. In this paper, we (i) propose a novel adversarial perturbation detection scheme based on multi-task perception of complex vision tasks (i.e., depth estimation and semantic segmentation). Specifically, adversarial perturbations are detected by inconsistencies between extracted edges of the input image, the depth output, and the segmentation output. To further improve this technique, we (ii) develop a novel edge consistency loss between all three modalities, thereby improving their initial consistency which in turn supports our detection scheme. We verify our detection scheme's effectiveness by employing various known attacks and image noises. In addition, we (iii) develop a multi-task adversarial attack, aiming at fooling both tasks as well as our detection scheme. Experimental evaluation on the Cityscapes and KITTI datasets shows that under an assumption of a 5% false positive rate up to 100% of images are correctly detected as adversarially perturbed, depending on the strength of the perturbation. Code is available at https://github.com/ifnspaml/AdvAttackDet. A short video at https://youtu.be/KKa6gOyWmH4 provides qualitative results.



## **38. Robustness through Cognitive Dissociation Mitigation in Contrastive Adversarial Training**

cs.LG

Accepted to ICMLC 2022

**SubmitDate**: 2022-09-11    [paper-pdf](http://arxiv.org/pdf/2203.08959v3)

**Authors**: Adir Rahamim, Itay Naeh

**Abstracts**: In this paper, we introduce a novel neural network training framework that increases model's adversarial robustness to adversarial attacks while maintaining high clean accuracy by combining contrastive learning (CL) with adversarial training (AT). We propose to improve model robustness to adversarial attacks by learning feature representations that are consistent under both data augmentations and adversarial perturbations. We leverage contrastive learning to improve adversarial robustness by considering an adversarial example as another positive example, and aim to maximize the similarity between random augmentations of data samples and their adversarial example, while constantly updating the classification head in order to avoid a cognitive dissociation between the classification head and the embedding space. This dissociation is caused by the fact that CL updates the network up to the embedding space, while freezing the classification head which is used to generate new positive adversarial examples. We validate our method, Contrastive Learning with Adversarial Features(CLAF), on the CIFAR-10 dataset on which it outperforms both robust accuracy and clean accuracy over alternative supervised and self-supervised adversarial learning methods.



## **39. Scattering Model Guided Adversarial Examples for SAR Target Recognition: Attack and Defense**

cs.CV

**SubmitDate**: 2022-09-11    [paper-pdf](http://arxiv.org/pdf/2209.04779v1)

**Authors**: Bowen Peng, Bo Peng, Jie Zhou, Jianyue Xie, Li Liu

**Abstracts**: Deep Neural Networks (DNNs) based Synthetic Aperture Radar (SAR) Automatic Target Recognition (ATR) systems have shown to be highly vulnerable to adversarial perturbations that are deliberately designed yet almost imperceptible but can bias DNN inference when added to targeted objects. This leads to serious safety concerns when applying DNNs to high-stake SAR ATR applications. Therefore, enhancing the adversarial robustness of DNNs is essential for implementing DNNs to modern real-world SAR ATR systems. Toward building more robust DNN-based SAR ATR models, this article explores the domain knowledge of SAR imaging process and proposes a novel Scattering Model Guided Adversarial Attack (SMGAA) algorithm which can generate adversarial perturbations in the form of electromagnetic scattering response (called adversarial scatterers). The proposed SMGAA consists of two parts: 1) a parametric scattering model and corresponding imaging method and 2) a customized gradient-based optimization algorithm. First, we introduce the effective Attributed Scattering Center Model (ASCM) and a general imaging method to describe the scattering behavior of typical geometric structures in the SAR imaging process. By further devising several strategies to take the domain knowledge of SAR target images into account and relax the greedy search procedure, the proposed method does not need to be prudentially finetuned, but can efficiently to find the effective ASCM parameters to fool the SAR classifiers and facilitate the robust model training. Comprehensive evaluations on the MSTAR dataset show that the adversarial scatterers generated by SMGAA are more robust to perturbations and transformations in the SAR processing chain than the currently studied attacks, and are effective to construct a defensive model against the malicious scatterers.



## **40. Atomic cross-chain exchanges of shared assets**

cs.CR

**SubmitDate**: 2022-09-10    [paper-pdf](http://arxiv.org/pdf/2202.12855v3)

**Authors**: Krishnasuri Narayanam, Venkatraman Ramakrishna, Dhinakaran Vinayagamurthy, Sandeep Nishad

**Abstracts**: A core enabler for blockchain or DLT interoperability is the ability to atomically exchange assets held by mutually untrusting owners on different ledgers. This atomic swap problem has been well-studied, with the Hash Time Locked Contract (HTLC) emerging as a canonical solution. HTLC ensures atomicity of exchange, albeit with caveats for node failure and timeliness of claims. But a bigger limitation of HTLC is that it only applies to a model consisting of two adversarial parties having sole ownership of a single asset in each ledger. Realistic extensions of the model in which assets may be jointly owned by multiple parties, all of whose consents are required for exchanges, or where multiple assets must be exchanged for one, are susceptible to collusion attacks and hence cannot be handled by HTLC. In this paper, we generalize the model of asset exchanges across DLT networks and present a taxonomy of use cases, describe the threat model, and propose MPHTLC, an augmented HTLC protocol for atomic multi-owner-and-asset exchanges. We analyze the correctness, safety, and application scope of MPHTLC. As proof-of-concept, we show how MPHTLC primitives can be implemented in networks built on Hyperledger Fabric and Corda, and how MPHTLC can be implemented in the Hyperledger Labs Weaver framework by augmenting its existing HTLC protocol.



## **41. Phantom Sponges: Exploiting Non-Maximum Suppression to Attack Deep Object Detectors**

cs.CV

**SubmitDate**: 2022-09-10    [paper-pdf](http://arxiv.org/pdf/2205.13618v2)

**Authors**: Avishag Shapira, Alon Zolfi, Luca Demetrio, Battista Biggio, Asaf Shabtai

**Abstracts**: Adversarial attacks against deep learning-based object detectors have been studied extensively in the past few years. Most of the attacks proposed have targeted the model's integrity (i.e., caused the model to make incorrect predictions), while adversarial attacks targeting the model's availability, a critical aspect in safety-critical domains such as autonomous driving, have not yet been explored by the machine learning research community. In this paper, we propose a novel attack that negatively affects the decision latency of an end-to-end object detection pipeline. We craft a universal adversarial perturbation (UAP) that targets a widely used technique integrated in many object detector pipelines -- non-maximum suppression (NMS). Our experiments demonstrate the proposed UAP's ability to increase the processing time of individual frames by adding "phantom" objects that overload the NMS algorithm while preserving the detection of the original objects (which allows the attack to go undetected for a longer period of time).



## **42. Bankrupting DoS Attackers Despite Uncertainty**

cs.CR

**SubmitDate**: 2022-09-10    [paper-pdf](http://arxiv.org/pdf/2205.08287v2)

**Authors**: Trisha Chakraborty, Abir Islam, Valerie King, Daniel Rayborn, Jared Saia, Maxwell Young

**Abstracts**: On-demand provisioning in the cloud allows for services to remain available despite massive denial-of-service (DoS) attacks. Unfortunately, on-demand provisioning is expensive and must be weighed against the costs incurred by an adversary. This leads to a recent threat known as {\it economic denial-of-sustainability (EDoS)}, where the cost for defending a service is higher than that of attacking.   A natural tool for combating EDoS is to impose costs via resource burning (RB). Here, a client must verifiably consume resources -- for example, by solving a computational challenge -- before service is rendered. However, prior RB-based defenses with security guarantees do not account for the cost of on-demand provisioning.   Another common approach is the use of heuristics -- such as a client's reputation score or the geographical location -- to identify and discard spurious job requests. However, these heuristics may err and existing approaches do not provide security guarantees when this occurs.   Here, we propose an EDoS defense, LCharge, that uses resource burning while accounting for on-demand provisioning. LCharge leverages an estimate of the number of job requests from honest clients (i.e., good jobs) in any set $S$ of requests to within an $O(\alpha)$-factor, for any unknown $\alpha>0$, but retains a strong security guarantee despite the uncertainty of this estimate. Specifically, against an adversary that expends $B$ resources to attack, the total cost for defending is $O( \alpha^{5/2}\sqrt{B\,(g+1)} + \alpha^3(g+\alpha))$ where $g$ is the number of good jobs. Notably, for large $B$ relative to $g$ and $\alpha$, the adversary has higher cost, implying that the algorithm has an economic advantage. Finally, we prove a lower bound for our problem of $\Omega(\sqrt{\alpha B g})$, showing that the cost of LCharge is asymptotically tight for $\alpha=\Theta(1)$.



## **43. The Space of Adversarial Strategies**

cs.CR

Accepted to the 32nd USENIX Security Symposium

**SubmitDate**: 2022-09-09    [paper-pdf](http://arxiv.org/pdf/2209.04521v1)

**Authors**: Ryan Sheatsley, Blaine Hoak, Eric Pauley, Patrick McDaniel

**Abstracts**: Adversarial examples, inputs designed to induce worst-case behavior in machine learning models, have been extensively studied over the past decade. Yet, our understanding of this phenomenon stems from a rather fragmented pool of knowledge; at present, there are a handful of attacks, each with disparate assumptions in threat models and incomparable definitions of optimality. In this paper, we propose a systematic approach to characterize worst-case (i.e., optimal) adversaries. We first introduce an extensible decomposition of attacks in adversarial machine learning by atomizing attack components into surfaces and travelers. With our decomposition, we enumerate over components to create 576 attacks (568 of which were previously unexplored). Next, we propose the Pareto Ensemble Attack (PEA): a theoretical attack that upper-bounds attack performance. With our new attacks, we measure performance relative to the PEA on: both robust and non-robust models, seven datasets, and three extended lp-based threat models incorporating compute costs, formalizing the Space of Adversarial Strategies. From our evaluation we find that attack performance to be highly contextual: the domain, model robustness, and threat model can have a profound influence on attack efficacy. Our investigation suggests that future studies measuring the security of machine learning should: (1) be contextualized to the domain & threat models, and (2) go beyond the handful of known attacks used today.



## **44. SoK: Certified Robustness for Deep Neural Networks**

cs.LG

To appear at 2023 IEEE Symposium on Security and Privacy (SP); 14  pages for the main text; benchmark & tool website:  http://sokcertifiedrobustness.github.io/

**SubmitDate**: 2022-09-09    [paper-pdf](http://arxiv.org/pdf/2009.04131v8)

**Authors**: Linyi Li, Tao Xie, Bo Li

**Abstracts**: Great advances in deep neural networks (DNNs) have led to state-of-the-art performance on a wide range of tasks. However, recent studies have shown that DNNs are vulnerable to adversarial attacks, which have brought great concerns when deploying these models to safety-critical applications such as autonomous driving. Different defense approaches have been proposed against adversarial attacks, including: a) empirical defenses, which can usually be adaptively attacked again without providing robustness certification; and b) certifiably robust approaches, which consist of robustness verification providing the lower bound of robust accuracy against any attacks under certain conditions and corresponding robust training approaches. In this paper, we systematize certifiably robust approaches and related practical and theoretical implications and findings. We also provide the first comprehensive benchmark on existing robustness verification and training approaches on different datasets. In particular, we 1) provide a taxonomy for the robustness verification and training approaches, as well as summarize the methodologies for representative algorithms, 2) reveal the characteristics, strengths, limitations, and fundamental connections among these approaches, 3) discuss current research progresses, theoretical barriers, main challenges, and future directions for certifiably robust approaches for DNNs, and 4) provide an open-sourced unified platform to evaluate 20+ representative certifiably robust approaches.



## **45. Adversarial Examples in Constrained Domains**

cs.CR

Accepted to IOS Press Journal of Computer Security

**SubmitDate**: 2022-09-09    [paper-pdf](http://arxiv.org/pdf/2011.01183v3)

**Authors**: Ryan Sheatsley, Nicolas Papernot, Michael Weisman, Gunjan Verma, Patrick McDaniel

**Abstracts**: Machine learning algorithms have been shown to be vulnerable to adversarial manipulation through systematic modification of inputs (e.g., adversarial examples) in domains such as image recognition. Under the default threat model, the adversary exploits the unconstrained nature of images; each feature (pixel) is fully under control of the adversary. However, it is not clear how these attacks translate to constrained domains that limit which and how features can be modified by the adversary (e.g., network intrusion detection). In this paper, we explore whether constrained domains are less vulnerable than unconstrained domains to adversarial example generation algorithms. We create an algorithm for generating adversarial sketches: targeted universal perturbation vectors which encode feature saliency within the envelope of domain constraints. To assess how these algorithms perform, we evaluate them in constrained (e.g., network intrusion detection) and unconstrained (e.g., image recognition) domains. The results demonstrate that our approaches generate misclassification rates in constrained domains that were comparable to those of unconstrained domains (greater than 95%). Our investigation shows that the narrow attack surface exposed by constrained domains is still sufficiently large to craft successful adversarial examples; and thus, constraints do not appear to make a domain robust. Indeed, with as little as five randomly selected features, one can still generate adversarial examples.



## **46. Robust-by-Design Classification via Unitary-Gradient Neural Networks**

cs.LG

Under review

**SubmitDate**: 2022-09-09    [paper-pdf](http://arxiv.org/pdf/2209.04293v1)

**Authors**: Fabio Brau, Giulio Rossolini, Alessandro Biondi, Giorgio Buttazzo

**Abstracts**: The use of neural networks in safety-critical systems requires safe and robust models, due to the existence of adversarial attacks. Knowing the minimal adversarial perturbation of any input x, or, equivalently, knowing the distance of x from the classification boundary, allows evaluating the classification robustness, providing certifiable predictions. Unfortunately, state-of-the-art techniques for computing such a distance are computationally expensive and hence not suited for online applications. This work proposes a novel family of classifiers, namely Signed Distance Classifiers (SDCs), that, from a theoretical perspective, directly output the exact distance of x from the classification boundary, rather than a probability score (e.g., SoftMax). SDCs represent a family of robust-by-design classifiers. To practically address the theoretical requirements of a SDC, a novel network architecture named Unitary-Gradient Neural Network is presented. Experimental results show that the proposed architecture approximates a signed distance classifier, hence allowing an online certifiable classification of x at the cost of a single inference.



## **47. Improving Out-of-Distribution Detection via Epistemic Uncertainty Adversarial Training**

cs.LG

8 pages, 5 figures

**SubmitDate**: 2022-09-09    [paper-pdf](http://arxiv.org/pdf/2209.03148v2)

**Authors**: Derek Everett, Andre T. Nguyen, Luke E. Richards, Edward Raff

**Abstracts**: The quantification of uncertainty is important for the adoption of machine learning, especially to reject out-of-distribution (OOD) data back to human experts for review. Yet progress has been slow, as a balance must be struck between computational efficiency and the quality of uncertainty estimates. For this reason many use deep ensembles of neural networks or Monte Carlo dropout for reasonable uncertainty estimates at relatively minimal compute and memory. Surprisingly, when we focus on the real-world applicable constraint of $\leq 1\%$ false positive rate (FPR), prior methods fail to reliably detect OOD samples as such. Notably, even Gaussian random noise fails to trigger these popular OOD techniques. We help to alleviate this problem by devising a simple adversarial training scheme that incorporates an attack of the epistemic uncertainty predicted by the dropout ensemble. We demonstrate this method improves OOD detection performance on standard data (i.e., not adversarially crafted), and improves the standardized partial AUC from near-random guessing performance to $\geq 0.75$.



## **48. Harnessing Perceptual Adversarial Patches for Crowd Counting**

cs.CV

**SubmitDate**: 2022-09-09    [paper-pdf](http://arxiv.org/pdf/2109.07986v2)

**Authors**: Shunchang Liu, Jiakai Wang, Aishan Liu, Yingwei Li, Yijie Gao, Xianglong Liu, Dacheng Tao

**Abstracts**: Crowd counting, which has been widely adopted for estimating the number of people in safety-critical scenes, is shown to be vulnerable to adversarial examples in the physical world (e.g., adversarial patches). Though harmful, adversarial examples are also valuable for evaluating and better understanding model robustness. However, existing adversarial example generation methods for crowd counting lack strong transferability among different black-box models, which limits their practicability for real-world systems. Motivated by the fact that attacking transferability is positively correlated to the model-invariant characteristics, this paper proposes the Perceptual Adversarial Patch (PAP) generation framework to tailor the adversarial perturbations for crowd counting scenes using the model-shared perceptual features. Specifically, we handcraft an adaptive crowd density weighting approach to capture the invariant scale perception features across various models and utilize the density guided attention to capture the model-shared position perception. Both of them are demonstrated to improve the attacking transferability of our adversarial patches. Extensive experiments show that our PAP could achieve state-of-the-art attacking performance in both the digital and physical world, and outperform previous proposals by large margins (at most +685.7 MAE and +699.5 MSE). Besides, we empirically demonstrate that adversarial training with our PAP can benefit the performance of vanilla models in alleviating several practical challenges in crowd counting scenarios, including generalization across datasets (up to -376.0 MAE and -354.9 MSE) and robustness towards complex backgrounds (up to -10.3 MAE and -16.4 MSE).



## **49. Uncovering the Connection Between Differential Privacy and Certified Robustness of Federated Learning against Poisoning Attacks**

cs.CR

**SubmitDate**: 2022-09-08    [paper-pdf](http://arxiv.org/pdf/2209.04030v1)

**Authors**: Chulin Xie, Yunhui Long, Pin-Yu Chen, Bo Li

**Abstracts**: Federated learning (FL) provides an efficient paradigm to jointly train a global model leveraging data from distributed users. As the local training data come from different users who may not be trustworthy, several studies have shown that FL is vulnerable to poisoning attacks. Meanwhile, to protect the privacy of local users, FL is always trained in a differentially private way (DPFL). Thus, in this paper, we ask: Can we leverage the innate privacy property of DPFL to provide certified robustness against poisoning attacks? Can we further improve the privacy of FL to improve such certification? We first investigate both user-level and instance-level privacy of FL and propose novel mechanisms to achieve improved instance-level privacy. We then provide two robustness certification criteria: certified prediction and certified attack cost for DPFL on both levels. Theoretically, we prove the certified robustness of DPFL under a bounded number of adversarial users or instances. Empirically, we conduct extensive experiments to verify our theories under a range of attacks on different datasets. We show that DPFL with a tighter privacy guarantee always provides stronger robustness certification in terms of certified attack cost, but the optimal certified prediction is achieved under a proper balance between privacy protection and utility loss.



## **50. Evaluating the Security of Aircraft Systems**

cs.CR

38 pages,

**SubmitDate**: 2022-09-08    [paper-pdf](http://arxiv.org/pdf/2209.04028v1)

**Authors**: Edan Habler, Ron Bitton, Asaf Shabtai

**Abstracts**: The sophistication and complexity of cyber attacks and the variety of targeted platforms have been growing in recent years. Various adversaries are abusing an increasing range of platforms, e.g., enterprise platforms, mobile phones, PCs, transportation systems, and industrial control systems. In recent years, we have witnessed various cyber attacks on transportation systems, including attacks on ports, airports, and trains. It is only a matter of time before transportation systems become a more common target of cyber attackers. Due to the enormous potential damage inherent in attacking vehicles carrying many passengers and the lack of security measures applied in traditional airborne systems, the vulnerability of aircraft systems is one of the most concerning topics in the vehicle security domain. This paper provides a comprehensive review of aircraft systems and components and their various networks, emphasizing the cyber threats they are exposed to and the impact of a cyber attack on these components and networks and the essential capabilities of the aircraft. In addition, we present a comprehensive and in-depth taxonomy that standardizes the knowledge and understanding of cyber security in the avionics field from an adversary's perspective. The taxonomy divides techniques into relevant categories (tactics) reflecting the various phases of the adversarial attack lifecycle and maps existing attacks according to the MITRE ATT&CK methodology. Furthermore, we analyze the security risks among the various systems according to the potential threat actors and categorize the threats based on STRIDE threat model. Future work directions are presented as guidelines for industry and academia.



