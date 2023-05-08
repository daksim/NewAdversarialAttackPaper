# Latest Adversarial Attack Papers
**update at 2023-05-08 09:40:08**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. White-Box Multi-Objective Adversarial Attack on Dialogue Generation**

cs.CL

ACL 2023 main conference long paper

**SubmitDate**: 2023-05-05    [abs](http://arxiv.org/abs/2305.03655v1) [paper-pdf](http://arxiv.org/pdf/2305.03655v1)

**Authors**: Yufei Li, Zexin Li, Yingfan Gao, Cong Liu

**Abstract**: Pre-trained transformers are popular in state-of-the-art dialogue generation (DG) systems. Such language models are, however, vulnerable to various adversarial samples as studied in traditional tasks such as text classification, which inspires our curiosity about their robustness in DG systems. One main challenge of attacking DG models is that perturbations on the current sentence can hardly degrade the response accuracy because the unchanged chat histories are also considered for decision-making. Instead of merely pursuing pitfalls of performance metrics such as BLEU, ROUGE, we observe that crafting adversarial samples to force longer generation outputs benefits attack effectiveness -- the generated responses are typically irrelevant, lengthy, and repetitive. To this end, we propose a white-box multi-objective attack method called DGSlow. Specifically, DGSlow balances two objectives -- generation accuracy and length, via a gradient-based multi-objective optimizer and applies an adaptive searching mechanism to iteratively craft adversarial samples with only a few modifications. Comprehensive experiments on four benchmark datasets demonstrate that DGSlow could significantly degrade state-of-the-art DG models with a higher success rate than traditional accuracy-based methods. Besides, our crafted sentences also exhibit strong transferability in attacking other models.



## **2. Verifiable Learning for Robust Tree Ensembles**

cs.LG

17 pages, 3 figures

**SubmitDate**: 2023-05-05    [abs](http://arxiv.org/abs/2305.03626v1) [paper-pdf](http://arxiv.org/pdf/2305.03626v1)

**Authors**: Stefano Calzavara, Lorenzo Cazzaro, Giulio Ermanno Pibiri, Nicola Prezza

**Abstract**: Verifying the robustness of machine learning models against evasion attacks at test time is an important research problem. Unfortunately, prior work established that this problem is NP-hard for decision tree ensembles, hence bound to be intractable for specific inputs. In this paper, we identify a restricted class of decision tree ensembles, called large-spread ensembles, which admit a security verification algorithm running in polynomial time. We then propose a new approach called verifiable learning, which advocates the training of such restricted model classes which are amenable for efficient verification. We show the benefits of this idea by designing a new training algorithm that automatically learns a large-spread decision tree ensemble from labelled data, thus enabling its security verification in polynomial time. Experimental results on publicly available datasets confirm that large-spread ensembles trained using our algorithm can be verified in a matter of seconds, using standard commercial hardware. Moreover, large-spread ensembles are more robust than traditional ensembles against evasion attacks, while incurring in just a relatively small loss of accuracy in the non-adversarial setting.



## **3. Not what you've signed up for: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection**

cs.CR

**SubmitDate**: 2023-05-05    [abs](http://arxiv.org/abs/2302.12173v2) [paper-pdf](http://arxiv.org/pdf/2302.12173v2)

**Authors**: Kai Greshake, Sahar Abdelnabi, Shailesh Mishra, Christoph Endres, Thorsten Holz, Mario Fritz

**Abstract**: Large Language Models (LLMs) are increasingly being integrated into various applications. The functionalities of recent LLMs can be flexibly modulated via natural language prompts. This renders them susceptible to targeted adversarial prompting, e.g., Prompt Injection (PI) attacks enable attackers to override original instructions and employed controls. So far, it was assumed that the user is directly prompting the LLM. But, what if it is not the user prompting? We argue that LLM-Integrated Applications blur the line between data and instructions. We reveal new attack vectors, using Indirect Prompt Injection, that enable adversaries to remotely (without a direct interface) exploit LLM-integrated applications by strategically injecting prompts into data likely to be retrieved. We derive a comprehensive taxonomy from a computer security perspective to systematically investigate impacts and vulnerabilities, including data theft, worming, information ecosystem contamination, and other novel security risks. We demonstrate our attacks' practical viability against both real-world systems, such as Bing's GPT-4 powered Chat and code-completion engines, and synthetic applications built on GPT-4. We show how processing retrieved prompts can act as arbitrary code execution, manipulate the application's functionality, and control how and if other APIs are called. Despite the increasing integration and reliance on LLMs, effective mitigations of these emerging threats are currently lacking. By raising awareness of these vulnerabilities and providing key insights into their implications, we aim to promote the safe and responsible deployment of these powerful models and the development of robust defenses that protect users and systems from potential attacks.



## **4. Exploring the Connection between Robust and Generative Models**

cs.LG

technical report, 6 pages, 6 figures

**SubmitDate**: 2023-05-05    [abs](http://arxiv.org/abs/2304.04033v2) [paper-pdf](http://arxiv.org/pdf/2304.04033v2)

**Authors**: Senad Beadini, Iacopo Masi

**Abstract**: We offer a study that connects robust discriminative classifiers trained with adversarial training (AT) with generative modeling in the form of Energy-based Models (EBM). We do so by decomposing the loss of a discriminative classifier and showing that the discriminative model is also aware of the input data density. Though a common assumption is that adversarial points leave the manifold of the input data, our study finds out that, surprisingly, untargeted adversarial points in the input space are very likely under the generative model hidden inside the discriminative classifier -- have low energy in the EBM. We present two evidence: untargeted attacks are even more likely than the natural data and their likelihood increases as the attack strength increases. This allows us to easily detect them and craft a novel attack called High-Energy PGD that fools the classifier yet has energy similar to the data set.



## **5. Boosting Adversarial Transferability via Fusing Logits of Top-1 Decomposed Feature**

cs.CV

**SubmitDate**: 2023-05-05    [abs](http://arxiv.org/abs/2305.01361v2) [paper-pdf](http://arxiv.org/pdf/2305.01361v2)

**Authors**: Juanjuan Weng, Zhiming Luo, Dazhen Lin, Shaozi Li, Zhun Zhong

**Abstract**: Recent research has shown that Deep Neural Networks (DNNs) are highly vulnerable to adversarial samples, which are highly transferable and can be used to attack other unknown black-box models. To improve the transferability of adversarial samples, several feature-based adversarial attack methods have been proposed to disrupt neuron activation in the middle layers. However, current state-of-the-art feature-based attack methods typically require additional computation costs for estimating the importance of neurons. To address this challenge, we propose a Singular Value Decomposition (SVD)-based feature-level attack method. Our approach is inspired by the discovery that eigenvectors associated with the larger singular values decomposed from the middle layer features exhibit superior generalization and attention properties. Specifically, we conduct the attack by retaining the decomposed Top-1 singular value-associated feature for computing the output logits, which are then combined with the original logits to optimize adversarial examples. Our extensive experimental results verify the effectiveness of our proposed method, which can be easily integrated into various baselines to significantly enhance the transferability of adversarial samples for disturbing normally trained CNNs and advanced defense strategies. The source code of this study is available at \textcolor{blue}{\href{https://anonymous.4open.science/r/SVD-SSA-13BF/README.md}{Link}}.



## **6. Diagnostics for Deep Neural Networks with Automated Copy/Paste Attacks**

cs.LG

Best paper award at the NeurIPS 2022 ML Safety Workshop --  https://neurips2022.mlsafety.org/

**SubmitDate**: 2023-05-05    [abs](http://arxiv.org/abs/2211.10024v3) [paper-pdf](http://arxiv.org/pdf/2211.10024v3)

**Authors**: Stephen Casper, Kaivalya Hariharan, Dylan Hadfield-Menell

**Abstract**: This paper considers the problem of helping humans exercise scalable oversight over deep neural networks (DNNs). Adversarial examples can be useful by helping to reveal weaknesses in DNNs, but they can be difficult to interpret or draw actionable conclusions from. Some previous works have proposed using human-interpretable adversarial attacks including copy/paste attacks in which one natural image pasted into another causes an unexpected misclassification. We build on these with two contributions. First, we introduce Search for Natural Adversarial Features Using Embeddings (SNAFUE) which offers a fully automated method for finding copy/paste attacks. Second, we use SNAFUE to red team an ImageNet classifier. We reproduce copy/paste attacks from previous works and find hundreds of other easily-describable vulnerabilities, all without a human in the loop. Code is available at https://github.com/thestephencasper/snafue



## **7. Efficient Adversarial Contrastive Learning via Robustness-Aware Coreset Selection**

cs.LG

**SubmitDate**: 2023-05-05    [abs](http://arxiv.org/abs/2302.03857v3) [paper-pdf](http://arxiv.org/pdf/2302.03857v3)

**Authors**: Xilie Xu, Jingfeng Zhang, Feng Liu, Masashi Sugiyama, Mohan Kankanhalli

**Abstract**: Adversarial contrastive learning (ACL) does not require expensive data annotations but outputs a robust representation that withstands adversarial attacks and also generalizes to a wide range of downstream tasks. However, ACL needs tremendous running time to generate the adversarial variants of all training data, which limits its scalability to large datasets. To speed up ACL, this paper proposes a robustness-aware coreset selection (RCS) method. RCS does not require label information and searches for an informative subset that minimizes a representational divergence, which is the distance of the representation between natural data and their virtual adversarial variants. The vanilla solution of RCS via traversing all possible subsets is computationally prohibitive. Therefore, we theoretically transform RCS into a surrogate problem of submodular maximization, of which the greedy search is an efficient solution with an optimality guarantee for the original problem. Empirically, our comprehensive results corroborate that RCS can speed up ACL by a large margin without significantly hurting the robustness transferability. Notably, to the best of our knowledge, we are the first to conduct ACL efficiently on the large-scale ImageNet-1K dataset to obtain an effective robust representation via RCS.



## **8. Single Node Injection Label Specificity Attack on Graph Neural Networks via Reinforcement Learning**

cs.LG

**SubmitDate**: 2023-05-04    [abs](http://arxiv.org/abs/2305.02901v1) [paper-pdf](http://arxiv.org/pdf/2305.02901v1)

**Authors**: Dayuan Chen, Jian Zhang, Yuqian Lv, Jinhuan Wang, Hongjie Ni, Shanqing Yu, Zhen Wang, Qi Xuan

**Abstract**: Graph neural networks (GNNs) have achieved remarkable success in various real-world applications. However, recent studies highlight the vulnerability of GNNs to malicious perturbations. Previous adversaries primarily focus on graph modifications or node injections to existing graphs, yielding promising results but with notable limitations. Graph modification attack~(GMA) requires manipulation of the original graph, which is often impractical, while graph injection attack~(GIA) necessitates training a surrogate model in the black-box setting, leading to significant performance degradation due to divergence between the surrogate architecture and the actual victim model. Furthermore, most methods concentrate on a single attack goal and lack a generalizable adversary to develop distinct attack strategies for diverse goals, thus limiting precise control over victim model behavior in real-world scenarios. To address these issues, we present a gradient-free generalizable adversary that injects a single malicious node to manipulate the classification result of a target node in the black-box evasion setting. We propose Gradient-free Generalizable Single Node Injection Attack, namely G$^2$-SNIA, a reinforcement learning framework employing Proximal Policy Optimization. By directly querying the victim model, G$^2$-SNIA learns patterns from exploration to achieve diverse attack goals with extremely limited attack budgets. Through comprehensive experiments over three acknowledged benchmark datasets and four prominent GNNs in the most challenging and realistic scenario, we demonstrate the superior performance of our proposed G$^2$-SNIA over the existing state-of-the-art baselines. Moreover, by comparing G$^2$-SNIA with multiple white-box evasion baselines, we confirm its capacity to generate solutions comparable to those of the best adversaries.



## **9. IMAP: Intrinsically Motivated Adversarial Policy**

cs.LG

**SubmitDate**: 2023-05-04    [abs](http://arxiv.org/abs/2305.02605v1) [paper-pdf](http://arxiv.org/pdf/2305.02605v1)

**Authors**: Xiang Zheng, Xingjun Ma, Shengjie Wang, Xinyu Wang, Chao Shen, Cong Wang

**Abstract**: Reinforcement learning (RL) agents are known to be vulnerable to evasion attacks during deployment. In single-agent environments, attackers can inject imperceptible perturbations on the policy or value network's inputs or outputs; in multi-agent environments, attackers can control an adversarial opponent to indirectly influence the victim's observation. Adversarial policies offer a promising solution to craft such attacks. Still, current approaches either require perfect or partial knowledge of the victim policy or suffer from sample inefficiency due to the sparsity of task-related rewards. To overcome these limitations, we propose the Intrinsically Motivated Adversarial Policy (IMAP) for efficient black-box evasion attacks in single- and multi-agent environments without any knowledge of the victim policy. IMAP uses four intrinsic objectives based on state coverage, policy coverage, risk, and policy divergence to encourage exploration and discover stronger attacking skills. We also design a novel Bias-Reduction (BR) method to boost IMAP further. Our experiments demonstrate the effectiveness of these intrinsic objectives and BR in improving adversarial policy learning in the black-box setting against multiple types of victim agents in various single- and multi-agent MuJoCo environments. Notably, our IMAP reduces the performance of the state-of-the-art robust WocaR-PPO agents by 34\%-54\% and achieves a SOTA attacking success rate of 83.91\% in the two-player zero-sum game YouShallNotPass.



## **10. Madvex: Instrumentation-based Adversarial Attacks on Machine Learning Malware Detection**

cs.CR

20 pages. To be published in The 20th Conference on Detection of  Intrusions and Malware & Vulnerability Assessment (DIMVA 2023)

**SubmitDate**: 2023-05-04    [abs](http://arxiv.org/abs/2305.02559v1) [paper-pdf](http://arxiv.org/pdf/2305.02559v1)

**Authors**: Nils Loose, Felix Mächtle, Claudius Pott, Volodymyr Bezsmertnyi, Thomas Eisenbarth

**Abstract**: WebAssembly (Wasm) is a low-level binary format for web applications, which has found widespread adoption due to its improved performance and compatibility with existing software. However, the popularity of Wasm has also led to its exploitation for malicious purposes, such as cryptojacking, where malicious actors use a victim's computing resources to mine cryptocurrencies without their consent. To counteract this threat, machine learning-based detection methods aiming to identify cryptojacking activities within Wasm code have emerged. It is well-known that neural networks are susceptible to adversarial attacks, where inputs to a classifier are perturbed with minimal changes that result in a crass misclassification. While applying changes in image classification is easy, manipulating binaries in an automated fashion to evade malware classification without changing functionality is non-trivial. In this work, we propose a new approach to include adversarial examples in the code section of binaries via instrumentation. The introduced gadgets allow for the inclusion of arbitrary bytes, enabling efficient adversarial attacks that reliably bypass state-of-the-art machine learning classifiers such as the CNN-based Minos recently proposed at NDSS 2021. We analyze the cost and reliability of instrumentation-based adversarial example generation and show that the approach works reliably at minimal size and performance overheads.



## **11. Detecting Adversarial Faces Using Only Real Face Self-Perturbations**

cs.CV

IJCAI2023

**SubmitDate**: 2023-05-04    [abs](http://arxiv.org/abs/2304.11359v2) [paper-pdf](http://arxiv.org/pdf/2304.11359v2)

**Authors**: Qian Wang, Yongqin Xian, Hefei Ling, Jinyuan Zhang, Xiaorui Lin, Ping Li, Jiazhong Chen, Ning Yu

**Abstract**: Adversarial attacks aim to disturb the functionality of a target system by adding specific noise to the input samples, bringing potential threats to security and robustness when applied to facial recognition systems. Although existing defense techniques achieve high accuracy in detecting some specific adversarial faces (adv-faces), new attack methods especially GAN-based attacks with completely different noise patterns circumvent them and reach a higher attack success rate. Even worse, existing techniques require attack data before implementing the defense, making it impractical to defend newly emerging attacks that are unseen to defenders. In this paper, we investigate the intrinsic generality of adv-faces and propose to generate pseudo adv-faces by perturbing real faces with three heuristically designed noise patterns. We are the first to train an adv-face detector using only real faces and their self-perturbations, agnostic to victim facial recognition systems, and agnostic to unseen attacks. By regarding adv-faces as out-of-distribution data, we then naturally introduce a novel cascaded system for adv-face detection, which consists of training data self-perturbations, decision boundary regularization, and a max-pooling-based binary classifier focusing on abnormal local color aberrations. Experiments conducted on LFW and CelebA-HQ datasets with eight gradient-based and two GAN-based attacks validate that our method generalizes to a variety of unseen adversarial attacks.



## **12. On the Security Risks of Knowledge Graph Reasoning**

cs.CR

In proceedings of USENIX Security'23. Codes:  https://github.com/HarrialX/security-risk-KG-reasoning

**SubmitDate**: 2023-05-03    [abs](http://arxiv.org/abs/2305.02383v1) [paper-pdf](http://arxiv.org/pdf/2305.02383v1)

**Authors**: Zhaohan Xi, Tianyu Du, Changjiang Li, Ren Pang, Shouling Ji, Xiapu Luo, Xusheng Xiao, Fenglong Ma, Ting Wang

**Abstract**: Knowledge graph reasoning (KGR) -- answering complex logical queries over large knowledge graphs -- represents an important artificial intelligence task, entailing a range of applications (e.g., cyber threat hunting). However, despite its surging popularity, the potential security risks of KGR are largely unexplored, which is concerning, given the increasing use of such capability in security-critical domains.   This work represents a solid initial step towards bridging the striking gap. We systematize the security threats to KGR according to the adversary's objectives, knowledge, and attack vectors. Further, we present ROAR, a new class of attacks that instantiate a variety of such threats. Through empirical evaluation in representative use cases (e.g., medical decision support, cyber threat hunting, and commonsense reasoning), we demonstrate that ROAR is highly effective to mislead KGR to suggest pre-defined answers for target queries, yet with negligible impact on non-target ones. Finally, we explore potential countermeasures against ROAR, including filtering of potentially poisoning knowledge and training with adversarially augmented queries, which leads to several promising research directions.



## **13. Privacy-Preserving Federated Recurrent Neural Networks**

cs.CR

Accepted for publication at the 23rd Privacy Enhancing Technologies  Symposium (PETS 2023)

**SubmitDate**: 2023-05-03    [abs](http://arxiv.org/abs/2207.13947v2) [paper-pdf](http://arxiv.org/pdf/2207.13947v2)

**Authors**: Sinem Sav, Abdulrahman Diaa, Apostolos Pyrgelis, Jean-Philippe Bossuat, Jean-Pierre Hubaux

**Abstract**: We present RHODE, a novel system that enables privacy-preserving training of and prediction on Recurrent Neural Networks (RNNs) in a cross-silo federated learning setting by relying on multiparty homomorphic encryption. RHODE preserves the confidentiality of the training data, the model, and the prediction data; and it mitigates federated learning attacks that target the gradients under a passive-adversary threat model. We propose a packing scheme, multi-dimensional packing, for a better utilization of Single Instruction, Multiple Data (SIMD) operations under encryption. With multi-dimensional packing, RHODE enables the efficient processing, in parallel, of a batch of samples. To avoid the exploding gradients problem, RHODE provides several clipping approximations for performing gradient clipping under encryption. We experimentally show that the model performance with RHODE remains similar to non-secure solutions both for homogeneous and heterogeneous data distribution among the data holders. Our experimental evaluation shows that RHODE scales linearly with the number of data holders and the number of timesteps, sub-linearly and sub-quadratically with the number of features and the number of hidden units of RNNs, respectively. To the best of our knowledge, RHODE is the first system that provides the building blocks for the training of RNNs and its variants, under encryption in a federated learning setting.



## **14. New Adversarial Image Detection Based on Sentiment Analysis**

cs.CR

**SubmitDate**: 2023-05-03    [abs](http://arxiv.org/abs/2305.03173v1) [paper-pdf](http://arxiv.org/pdf/2305.03173v1)

**Authors**: Yulong Wang, Tianxiang Li, Shenghong Li, Xin Yuan, Wei Ni

**Abstract**: Deep Neural Networks (DNNs) are vulnerable to adversarial examples, while adversarial attack models, e.g., DeepFool, are on the rise and outrunning adversarial example detection techniques. This paper presents a new adversarial example detector that outperforms state-of-the-art detectors in identifying the latest adversarial attacks on image datasets. Specifically, we propose to use sentiment analysis for adversarial example detection, qualified by the progressively manifesting impact of an adversarial perturbation on the hidden-layer feature maps of a DNN under attack. Accordingly, we design a modularized embedding layer with the minimum learnable parameters to embed the hidden-layer feature maps into word vectors and assemble sentences ready for sentiment analysis. Extensive experiments demonstrate that the new detector consistently surpasses the state-of-the-art detection algorithms in detecting the latest attacks launched against ResNet and Inception neutral networks on the CIFAR-10, CIFAR-100 and SVHN datasets. The detector only has about 2 million parameters, and takes shorter than 4.6 milliseconds to detect an adversarial example generated by the latest attack models using a Tesla K80 GPU card.



## **15. Text Adversarial Purification as Defense against Adversarial Attacks**

cs.CL

Accepted by ACL2023 main conference

**SubmitDate**: 2023-05-03    [abs](http://arxiv.org/abs/2203.14207v2) [paper-pdf](http://arxiv.org/pdf/2203.14207v2)

**Authors**: Linyang Li, Demin Song, Xipeng Qiu

**Abstract**: Adversarial purification is a successful defense mechanism against adversarial attacks without requiring knowledge of the form of the incoming attack. Generally, adversarial purification aims to remove the adversarial perturbations therefore can make correct predictions based on the recovered clean samples. Despite the success of adversarial purification in the computer vision field that incorporates generative models such as energy-based models and diffusion models, using purification as a defense strategy against textual adversarial attacks is rarely explored. In this work, we introduce a novel adversarial purification method that focuses on defending against textual adversarial attacks. With the help of language models, we can inject noise by masking input texts and reconstructing the masked texts based on the masked language models. In this way, we construct an adversarial purification process for textual models against the most widely used word-substitution adversarial attacks. We test our proposed adversarial purification method on several strong adversarial attack methods including Textfooler and BERT-Attack and experimental results indicate that the purification algorithm can successfully defend against strong word-substitution attacks.



## **16. Adversarial Neon Beam: Robust Physical-World Adversarial Attack to DNNs**

cs.CV

**SubmitDate**: 2023-05-03    [abs](http://arxiv.org/abs/2204.00853v2) [paper-pdf](http://arxiv.org/pdf/2204.00853v2)

**Authors**: Chengyin Hu, Kalibinuer Tiliwalidi

**Abstract**: In the physical world, light affects the performance of deep neural networks. Nowadays, many products based on deep neural network have been put into daily life. There are few researches on the effect of light on the performance of deep neural network models. However, the adversarial perturbations generated by light may have extremely dangerous effects on these systems. In this work, we propose an attack method called adversarial neon beam (AdvNB), which can execute the physical attack by obtaining the physical parameters of adversarial neon beams with very few queries. Experiments show that our algorithm can achieve advanced attack effect in both digital test and physical test. In the digital environment, 99.3% attack success rate was achieved, and in the physical environment, 100% attack success rate was achieved. Compared with the most advanced physical attack methods, our method can achieve better physical perturbation concealment. In addition, by analyzing the experimental data, we reveal some new phenomena brought about by the adversarial neon beam attack.



## **17. How Far Are We from Real Synonym Substitution Attacks?**

cs.CL

Findings in ACL 2023

**SubmitDate**: 2023-05-03    [abs](http://arxiv.org/abs/2210.02844v2) [paper-pdf](http://arxiv.org/pdf/2210.02844v2)

**Authors**: Cheng-Han Chiang, Hung-yi Lee

**Abstract**: In this paper, we explore the following question: how far are we from real synonym substitution attacks (SSAs). We approach this question by examining how SSAs replace words in the original sentence and show that there are still unresolved obstacles that make current SSAs generate invalid adversarial samples. We reveal that four widely used word substitution methods generate a large fraction of invalid substitution words that are ungrammatical or do not preserve the original sentence's semantics. Next, we show that the semantic and grammatical constraints used in SSAs for detecting invalid word replacements are highly insufficient in detecting invalid adversarial samples. Our work is an important stepping stone to constructing better SSAs in the future.



## **18. Can Large Language Models Be an Alternative to Human Evaluations?**

cs.CL

ACL 2023 main conference paper. Main content: 10 pages (including  limitations). Appendix: 13 pages

**SubmitDate**: 2023-05-03    [abs](http://arxiv.org/abs/2305.01937v1) [paper-pdf](http://arxiv.org/pdf/2305.01937v1)

**Authors**: Cheng-Han Chiang, Hung-yi Lee

**Abstract**: Human evaluation is indispensable and inevitable for assessing the quality of texts generated by machine learning models or written by humans. However, human evaluation is very difficult to reproduce and its quality is notoriously unstable, hindering fair comparisons among different natural language processing (NLP) models and algorithms. Recently, large language models (LLMs) have demonstrated exceptional performance on unseen tasks when only the task instructions are provided. In this paper, we explore if such an ability of the LLMs can be used as an alternative to human evaluation. We present the LLMs with the exact same instructions, samples to be evaluated, and questions used to conduct human evaluation, and then ask the LLMs to generate responses to those questions; we dub this LLM evaluation. We use human evaluation and LLM evaluation to evaluate the texts in two NLP tasks: open-ended story generation and adversarial attacks. We show that the result of LLM evaluation is consistent with the results obtained by expert human evaluation: the texts rated higher by human experts are also rated higher by the LLMs. We also find that the results of LLM evaluation are stable over different formatting of the task instructions and the sampling algorithm used to generate the answer. We are the first to show the potential of using LLMs to assess the quality of texts and discuss the limitations and ethical considerations of LLM evaluation.



## **19. Towards Imperceptible Document Manipulations against Neural Ranking Models**

cs.IR

Accepted to Findings of ACL 2023

**SubmitDate**: 2023-05-03    [abs](http://arxiv.org/abs/2305.01860v1) [paper-pdf](http://arxiv.org/pdf/2305.01860v1)

**Authors**: Xuanang Chen, Ben He, Zheng Ye, Le Sun, Yingfei Sun

**Abstract**: Adversarial attacks have gained traction in order to identify potential vulnerabilities in neural ranking models (NRMs), but current attack methods often introduce grammatical errors, nonsensical expressions, or incoherent text fragments, which can be easily detected. Additionally, current methods rely heavily on the use of a well-imitated surrogate NRM to guarantee the attack effect, which makes them difficult to use in practice. To address these issues, we propose a framework called Imperceptible DocumEnt Manipulation (IDEM) to produce adversarial documents that are less noticeable to both algorithms and humans. IDEM instructs a well-established generative language model, such as BART, to generate connection sentences without introducing easy-to-detect errors, and employs a separate position-wise merging strategy to balance relevance and coherence of the perturbed text. Experimental results on the popular MS MARCO benchmark demonstrate that IDEM can outperform strong baselines while preserving fluency and correctness of the target documents as evidenced by automatic and human evaluations. Furthermore, the separation of adversarial text generation from the surrogate NRM makes IDEM more robust and less affected by the quality of the surrogate NRM.



## **20. GREAT Score: Global Robustness Evaluation of Adversarial Perturbation using Generative Models**

cs.LG

**SubmitDate**: 2023-05-03    [abs](http://arxiv.org/abs/2304.09875v2) [paper-pdf](http://arxiv.org/pdf/2304.09875v2)

**Authors**: Zaitang Li, Pin-Yu Chen, Tsung-Yi Ho

**Abstract**: Current studies on adversarial robustness mainly focus on aggregating local robustness results from a set of data samples to evaluate and rank different models. However, the local statistics may not well represent the true global robustness of the underlying unknown data distribution. To address this challenge, this paper makes the first attempt to present a new framework, called GREAT Score , for global robustness evaluation of adversarial perturbation using generative models. Formally, GREAT Score carries the physical meaning of a global statistic capturing a mean certified attack-proof perturbation level over all samples drawn from a generative model. For finite-sample evaluation, we also derive a probabilistic guarantee on the sample complexity and the difference between the sample mean and the true mean. GREAT Score has several advantages: (1) Robustness evaluations using GREAT Score are efficient and scalable to large models, by sparing the need of running adversarial attacks. In particular, we show high correlation and significantly reduced computation cost of GREAT Score when compared to the attack-based model ranking on RobustBench (Croce,et. al. 2021). (2) The use of generative models facilitates the approximation of the unknown data distribution. In our ablation study with different generative adversarial networks (GANs), we observe consistency between global robustness evaluation and the quality of GANs. (3) GREAT Score can be used for remote auditing of privacy-sensitive black-box models, as demonstrated by our robustness evaluation on several online facial recognition services.



## **21. Sentiment Perception Adversarial Attacks on Neural Machine Translation Systems**

cs.CL

**SubmitDate**: 2023-05-02    [abs](http://arxiv.org/abs/2305.01437v1) [paper-pdf](http://arxiv.org/pdf/2305.01437v1)

**Authors**: Vyas Raina, Mark Gales

**Abstract**: With the advent of deep learning methods, Neural Machine Translation (NMT) systems have become increasingly powerful. However, deep learning based systems are susceptible to adversarial attacks, where imperceptible changes to the input can cause undesirable changes at the output of the system. To date there has been little work investigating adversarial attacks on sequence-to-sequence systems, such as NMT models. Previous work in NMT has examined attacks with the aim of introducing target phrases in the output sequence. In this work, adversarial attacks for NMT systems are explored from an output perception perspective. Thus the aim of an attack is to change the perception of the output sequence, without altering the perception of the input sequence. For example, an adversary may distort the sentiment of translated reviews to have an exaggerated positive sentiment. In practice it is challenging to run extensive human perception experiments, so a proxy deep-learning classifier applied to the NMT output is used to measure perception changes. Experiments demonstrate that the sentiment perception of NMT systems' output sequences can be changed significantly.



## **22. Improving adversarial robustness by putting more regularizations on less robust samples**

stat.ML

**SubmitDate**: 2023-05-02    [abs](http://arxiv.org/abs/2206.03353v3) [paper-pdf](http://arxiv.org/pdf/2206.03353v3)

**Authors**: Dongyoon Yang, Insung Kong, Yongdai Kim

**Abstract**: Adversarial training, which is to enhance robustness against adversarial attacks, has received much attention because it is easy to generate human-imperceptible perturbations of data to deceive a given deep neural network. In this paper, we propose a new adversarial training algorithm that is theoretically well motivated and empirically superior to other existing algorithms. A novel feature of the proposed algorithm is to apply more regularization to data vulnerable to adversarial attacks than other existing regularization algorithms do. Theoretically, we show that our algorithm can be understood as an algorithm of minimizing the regularized empirical risk motivated from a newly derived upper bound of the robust risk. Numerical experiments illustrate that our proposed algorithm improves the generalization (accuracy on examples) and robustness (accuracy on adversarial attacks) simultaneously to achieve the state-of-the-art performance.



## **23. StyleFool: Fooling Video Classification Systems via Style Transfer**

cs.CV

18 pages, 9 figures. Accepted to S&P 2023

**SubmitDate**: 2023-05-02    [abs](http://arxiv.org/abs/2203.16000v3) [paper-pdf](http://arxiv.org/pdf/2203.16000v3)

**Authors**: Yuxin Cao, Xi Xiao, Ruoxi Sun, Derui Wang, Minhui Xue, Sheng Wen

**Abstract**: Video classification systems are vulnerable to adversarial attacks, which can create severe security problems in video verification. Current black-box attacks need a large number of queries to succeed, resulting in high computational overhead in the process of attack. On the other hand, attacks with restricted perturbations are ineffective against defenses such as denoising or adversarial training. In this paper, we focus on unrestricted perturbations and propose StyleFool, a black-box video adversarial attack via style transfer to fool the video classification system. StyleFool first utilizes color theme proximity to select the best style image, which helps avoid unnatural details in the stylized videos. Meanwhile, the target class confidence is additionally considered in targeted attacks to influence the output distribution of the classifier by moving the stylized video closer to or even across the decision boundary. A gradient-free method is then employed to further optimize the adversarial perturbations. We carry out extensive experiments to evaluate StyleFool on two standard datasets, UCF-101 and HMDB-51. The experimental results demonstrate that StyleFool outperforms the state-of-the-art adversarial attacks in terms of both the number of queries and the robustness against existing defenses. Moreover, 50% of the stylized videos in untargeted attacks do not need any query since they can already fool the video classification model. Furthermore, we evaluate the indistinguishability through a user study to show that the adversarial samples of StyleFool look imperceptible to human eyes, despite unrestricted perturbations.



## **24. Exposing Fine-Grained Adversarial Vulnerability of Face Anti-Spoofing Models**

cs.CV

Accepted by IEEE/CVF Conference on Computer Vision and Pattern  Recognition (CVPR) Workshop, 2023

**SubmitDate**: 2023-05-02    [abs](http://arxiv.org/abs/2205.14851v3) [paper-pdf](http://arxiv.org/pdf/2205.14851v3)

**Authors**: Songlin Yang, Wei Wang, Chenye Xu, Ziwen He, Bo Peng, Jing Dong

**Abstract**: Face anti-spoofing aims to discriminate the spoofing face images (e.g., printed photos) from live ones. However, adversarial examples greatly challenge its credibility, where adding some perturbation noise can easily change the predictions. Previous works conducted adversarial attack methods to evaluate the face anti-spoofing performance without any fine-grained analysis that which model architecture or auxiliary feature is vulnerable to the adversary. To handle this problem, we propose a novel framework to expose the fine-grained adversarial vulnerability of the face anti-spoofing models, which consists of a multitask module and a semantic feature augmentation (SFA) module. The multitask module can obtain different semantic features for further evaluation, but only attacking these semantic features fails to reflect the discrimination-related vulnerability. We then design the SFA module to introduce the data distribution prior for more discrimination-related gradient directions for generating adversarial examples. Comprehensive experiments show that SFA module increases the attack success rate by nearly 40$\%$ on average. We conduct this fine-grained adversarial analysis on different annotations, geometric maps, and backbone networks (e.g., Resnet network). These fine-grained adversarial examples can be used for selecting robust backbone networks and auxiliary features. They also can be used for adversarial training, which makes it practical to further improve the accuracy and robustness of the face anti-spoofing models.



## **25. Stratified Adversarial Robustness with Rejection**

cs.LG

Paper published at International Conference on Machine Learning  (ICML'23)

**SubmitDate**: 2023-05-02    [abs](http://arxiv.org/abs/2305.01139v1) [paper-pdf](http://arxiv.org/pdf/2305.01139v1)

**Authors**: Jiefeng Chen, Jayaram Raghuram, Jihye Choi, Xi Wu, Yingyu Liang, Somesh Jha

**Abstract**: Recently, there is an emerging interest in adversarially training a classifier with a rejection option (also known as a selective classifier) for boosting adversarial robustness. While rejection can incur a cost in many applications, existing studies typically associate zero cost with rejecting perturbed inputs, which can result in the rejection of numerous slightly-perturbed inputs that could be correctly classified. In this work, we study adversarially-robust classification with rejection in the stratified rejection setting, where the rejection cost is modeled by rejection loss functions monotonically non-increasing in the perturbation magnitude. We theoretically analyze the stratified rejection setting and propose a novel defense method -- Adversarial Training with Consistent Prediction-based Rejection (CPR) -- for building a robust selective classifier. Experiments on image datasets demonstrate that the proposed method significantly outperforms existing methods under strong adaptive attacks. For instance, on CIFAR-10, CPR reduces the total robust loss (for different rejection losses) by at least 7.3% under both seen and unseen attacks.



## **26. Randomized Reversible Gate-Based Obfuscation for Secured Compilation of Quantum Circuit**

quant-ph

11 pages, 12 figures, conference

**SubmitDate**: 2023-05-02    [abs](http://arxiv.org/abs/2305.01133v1) [paper-pdf](http://arxiv.org/pdf/2305.01133v1)

**Authors**: Subrata Das, Swaroop Ghosh

**Abstract**: The success of quantum circuits in providing reliable outcomes for a given problem depends on the gate count and depth in near-term noisy quantum computers. Quantum circuit compilers that decompose high-level gates to native gates of the hardware and optimize the circuit play a key role in quantum computing. However, the quality and time complexity of the optimization process can vary significantly especially for practically relevant large-scale quantum circuits. As a result, third-party (often less-trusted/untrusted) compilers have emerged, claiming to provide better and faster optimization of complex quantum circuits than so-called trusted compilers. However, untrusted compilers can pose severe security risks, such as the theft of sensitive intellectual property (IP) embedded within the quantum circuit. We propose an obfuscation technique for quantum circuits using randomized reversible gates to protect them from such attacks during compilation. The idea is to insert a small random circuit into the original circuit and send it to the untrusted compiler. Since the circuit function is corrupted, the adversary may get incorrect IP. However, the user may also get incorrect output post-compilation. To circumvent this issue, we concatenate the inverse of the random circuit in the compiled circuit to recover the original functionality. We demonstrate the practicality of our method by conducting exhaustive experiments on a set of benchmark circuits and measuring the quality of obfuscation by calculating the Total Variation Distance (TVD) metric. Our method achieves TVD of up to 1.92 and performs at least 2X better than a previously reported obfuscation method. We also propose a novel adversarial reverse engineering (RE) approach and show that the proposed obfuscation is resilient against RE attacks. The proposed technique introduces minimal degradation in fidelity (~1% to ~3% on average).



## **27. Evaluating Adversarial Robustness on Document Image Classification**

cs.CV

The 17th International Conference on Document Analysis and  Recognition

**SubmitDate**: 2023-05-01    [abs](http://arxiv.org/abs/2304.12486v2) [paper-pdf](http://arxiv.org/pdf/2304.12486v2)

**Authors**: Timothée Fronteau, Arnaud Paran, Aymen Shabou

**Abstract**: Adversarial attacks and defenses have gained increasing interest on computer vision systems in recent years, but as of today, most investigations are limited to images. However, many artificial intelligence models actually handle documentary data, which is very different from real world images. Hence, in this work, we try to apply the adversarial attack philosophy on documentary and natural data and to protect models against such attacks. We focus our work on untargeted gradient-based, transfer-based and score-based attacks and evaluate the impact of adversarial training, JPEG input compression and grey-scale input transformation on the robustness of ResNet50 and EfficientNetB0 model architectures. To the best of our knowledge, no such work has been conducted by the community in order to study the impact of these attacks on the document image classification task.



## **28. Physical Adversarial Attacks for Surveillance: A Survey**

cs.CV

**SubmitDate**: 2023-05-01    [abs](http://arxiv.org/abs/2305.01074v1) [paper-pdf](http://arxiv.org/pdf/2305.01074v1)

**Authors**: Kien Nguyen, Tharindu Fernando, Clinton Fookes, Sridha Sridharan

**Abstract**: Modern automated surveillance techniques are heavily reliant on deep learning methods. Despite the superior performance, these learning systems are inherently vulnerable to adversarial attacks - maliciously crafted inputs that are designed to mislead, or trick, models into making incorrect predictions. An adversary can physically change their appearance by wearing adversarial t-shirts, glasses, or hats or by specific behavior, to potentially avoid various forms of detection, tracking and recognition of surveillance systems; and obtain unauthorized access to secure properties and assets. This poses a severe threat to the security and safety of modern surveillance systems. This paper reviews recent attempts and findings in learning and designing physical adversarial attacks for surveillance applications. In particular, we propose a framework to analyze physical adversarial attacks and provide a comprehensive survey of physical adversarial attacks on four key surveillance tasks: detection, identification, tracking, and action recognition under this framework. Furthermore, we review and analyze strategies to defend against the physical adversarial attacks and the methods for evaluating the strengths of the defense. The insights in this paper present an important step in building resilience within surveillance systems to physical adversarial attacks.



## **29. IoTFlowGenerator: Crafting Synthetic IoT Device Traffic Flows for Cyber Deception**

cs.CR

FLAIRS-36

**SubmitDate**: 2023-05-01    [abs](http://arxiv.org/abs/2305.00925v1) [paper-pdf](http://arxiv.org/pdf/2305.00925v1)

**Authors**: Joseph Bao, Murat Kantarcioglu, Yevgeniy Vorobeychik, Charles Kamhoua

**Abstract**: Over the years, honeypots emerged as an important security tool to understand attacker intent and deceive attackers to spend time and resources. Recently, honeypots are being deployed for Internet of things (IoT) devices to lure attackers, and learn their behavior. However, most of the existing IoT honeypots, even the high interaction ones, are easily detected by an attacker who can observe honeypot traffic due to lack of real network traffic originating from the honeypot. This implies that, to build better honeypots and enhance cyber deception capabilities, IoT honeypots need to generate realistic network traffic flows. To achieve this goal, we propose a novel deep learning based approach for generating traffic flows that mimic real network traffic due to user and IoT device interactions. A key technical challenge that our approach overcomes is scarcity of device-specific IoT traffic data to effectively train a generator. We address this challenge by leveraging a core generative adversarial learning algorithm for sequences along with domain specific knowledge common to IoT devices. Through an extensive experimental evaluation with 18 IoT devices, we demonstrate that the proposed synthetic IoT traffic generation tool significantly outperforms state of the art sequence and packet generators in remaining indistinguishable from real traffic even to an adaptive attacker.



## **30. Attack-SAM: Towards Evaluating Adversarial Robustness of Segment Anything Model**

cs.CV

The first work to evaluate the adversarial robustness of Segment  Anything Model (ongoing)

**SubmitDate**: 2023-05-01    [abs](http://arxiv.org/abs/2305.00866v1) [paper-pdf](http://arxiv.org/pdf/2305.00866v1)

**Authors**: Chenshuang Zhang, Chaoning Zhang, Taegoo Kang, Donghun Kim, Sung-Ho Bae, In So Kweon

**Abstract**: Segment Anything Model (SAM) has attracted significant attention recently, due to its impressive performance on various downstream tasks in a zero-short manner. Computer vision (CV) area might follow the natural language processing (NLP) area to embark on a path from task-specific vision models toward foundation models. However, previous task-specific models are widely recognized as vulnerable to adversarial examples, which fool the model to make wrong predictions with imperceptible perturbation. Such vulnerability to adversarial attacks causes serious concerns when applying deep models to security-sensitive applications. Therefore, it is critical to know whether the vision foundation model SAM can also be easily fooled by adversarial attacks. To the best of our knowledge, our work is the first of its kind to conduct a comprehensive investigation on how to attack SAM with adversarial examples. Specifically, we find that SAM is vulnerable to white-box attacks while maintaining robustness to some extent in the black-box setting. This is an ongoing project and more results and findings will be updated soon through https://github.com/chenshuang-zhang/attack-sam.



## **31. Visual Prompting for Adversarial Robustness**

cs.CV

ICASSP 2023

**SubmitDate**: 2023-05-01    [abs](http://arxiv.org/abs/2210.06284v4) [paper-pdf](http://arxiv.org/pdf/2210.06284v4)

**Authors**: Aochuan Chen, Peter Lorenz, Yuguang Yao, Pin-Yu Chen, Sijia Liu

**Abstract**: In this work, we leverage visual prompting (VP) to improve adversarial robustness of a fixed, pre-trained model at testing time. Compared to conventional adversarial defenses, VP allows us to design universal (i.e., data-agnostic) input prompting templates, which have plug-and-play capabilities at testing time to achieve desired model performance without introducing much computation overhead. Although VP has been successfully applied to improving model generalization, it remains elusive whether and how it can be used to defend against adversarial attacks. We investigate this problem and show that the vanilla VP approach is not effective in adversarial defense since a universal input prompt lacks the capacity for robust learning against sample-specific adversarial perturbations. To circumvent it, we propose a new VP method, termed Class-wise Adversarial Visual Prompting (C-AVP), to generate class-wise visual prompts so as to not only leverage the strengths of ensemble prompts but also optimize their interrelations to improve model robustness. Our experiments show that C-AVP outperforms the conventional VP method, with 2.1X standard accuracy gain and 2X robust accuracy gain. Compared to classical test-time defenses, C-AVP also yields a 42X inference time speedup.



## **32. Robustness of Graph Neural Networks at Scale**

cs.LG

39 pages, 22 figures, 17 tables NeurIPS 2021

**SubmitDate**: 2023-04-30    [abs](http://arxiv.org/abs/2110.14038v4) [paper-pdf](http://arxiv.org/pdf/2110.14038v4)

**Authors**: Simon Geisler, Tobias Schmidt, Hakan Şirin, Daniel Zügner, Aleksandar Bojchevski, Stephan Günnemann

**Abstract**: Graph Neural Networks (GNNs) are increasingly important given their popularity and the diversity of applications. Yet, existing studies of their vulnerability to adversarial attacks rely on relatively small graphs. We address this gap and study how to attack and defend GNNs at scale. We propose two sparsity-aware first-order optimization attacks that maintain an efficient representation despite optimizing over a number of parameters which is quadratic in the number of nodes. We show that common surrogate losses are not well-suited for global attacks on GNNs. Our alternatives can double the attack strength. Moreover, to improve GNNs' reliability we design a robust aggregation function, Soft Median, resulting in an effective defense at all scales. We evaluate our attacks and defense with standard GNNs on graphs more than 100 times larger compared to previous work. We even scale one order of magnitude further by extending our techniques to a scalable GNN.



## **33. Assessing Vulnerabilities of Adversarial Learning Algorithm through Poisoning Attacks**

cs.CR

**SubmitDate**: 2023-04-30    [abs](http://arxiv.org/abs/2305.00399v1) [paper-pdf](http://arxiv.org/pdf/2305.00399v1)

**Authors**: Jingfeng Zhang, Bo Song, Bo Han, Lei Liu, Gang Niu, Masashi Sugiyama

**Abstract**: Adversarial training (AT) is a robust learning algorithm that can defend against adversarial attacks in the inference phase and mitigate the side effects of corrupted data in the training phase. As such, it has become an indispensable component of many artificial intelligence (AI) systems. However, in high-stake AI applications, it is crucial to understand AT's vulnerabilities to ensure reliable deployment. In this paper, we investigate AT's susceptibility to poisoning attacks, a type of malicious attack that manipulates training data to compromise the performance of the trained model. Previous work has focused on poisoning attacks against standard training, but little research has been done on their effectiveness against AT. To fill this gap, we design and test effective poisoning attacks against AT. Specifically, we investigate and design clean-label poisoning attacks, allowing attackers to imperceptibly modify a small fraction of training data to control the algorithm's behavior on a specific target data point. Additionally, we propose the clean-label untargeted attack, enabling attackers can attach tiny stickers on training data to degrade the algorithm's performance on all test data, where the stickers could serve as a signal against unauthorized data collection. Our experiments demonstrate that AT can still be poisoned, highlighting the need for caution when using vanilla AT algorithms in security-related applications. The code is at https://github.com/zjfheart/Poison-adv-training.git.



## **34. Enhancing Adversarial Contrastive Learning via Adversarial Invariant Regularization**

cs.LG

**SubmitDate**: 2023-04-30    [abs](http://arxiv.org/abs/2305.00374v1) [paper-pdf](http://arxiv.org/pdf/2305.00374v1)

**Authors**: Xilie Xu, Jingfeng Zhang, Feng Liu, Masashi Sugiyama, Mohan Kankanhalli

**Abstract**: Adversarial contrastive learning (ACL), without requiring labels, incorporates adversarial data with standard contrastive learning (SCL) and outputs a robust representation which is generalizable and resistant to adversarial attacks and common corruptions. The style-independence property of representations has been validated to be beneficial in improving robustness transferability. Standard invariant regularization (SIR) has been proposed to make the learned representations via SCL to be independent of the style factors. However, how to equip robust representations learned via ACL with the style-independence property is still unclear so far. To this end, we leverage the technique of causal reasoning to propose an adversarial invariant regularization (AIR) that enforces robust representations learned via ACL to be style-independent. Then, we enhance ACL using invariant regularization (IR), which is a weighted sum of SIR and AIR. Theoretically, we show that AIR implicitly encourages the prediction of adversarial data and consistency between adversarial and natural data to be independent of data augmentations. We also theoretically demonstrate that the style-independence property of robust representation learned via ACL still holds in downstream tasks, providing generalization guarantees. Empirically, our comprehensive experimental results corroborate that IR can significantly improve the performance of ACL and its variants on various datasets.



## **35. MetaShard: A Novel Sharding Blockchain Platform for Metaverse Applications**

cs.CR

**SubmitDate**: 2023-04-30    [abs](http://arxiv.org/abs/2305.00367v1) [paper-pdf](http://arxiv.org/pdf/2305.00367v1)

**Authors**: Cong T. Nguyen, Dinh Thai Hoang, Diep N. Nguyen, Yong Xiao, Dusit Niyato, Eryk Dutkiewicz

**Abstract**: Due to its security, transparency, and flexibility in verifying virtual assets, blockchain has been identified as one of the key technologies for Metaverse. Unfortunately, blockchain-based Metaverse faces serious challenges such as massive resource demands, scalability, and security concerns. To address these issues, this paper proposes a novel sharding-based blockchain framework, namely MetaShard, for Metaverse applications. Particularly, we first develop an effective consensus mechanism, namely Proof-of-Engagement, that can incentivize MUs' data and computing resource contribution. Moreover, to improve the scalability of MetaShard, we propose an innovative sharding management scheme to maximize the network's throughput while protecting the shards from 51% attacks. Since the optimization problem is NP-complete, we develop a hybrid approach that decomposes the problem (using the binary search method) into sub-problems that can be solved effectively by the Lagrangian method. As a result, the proposed approach can obtain solutions in polynomial time, thereby enabling flexible shard reconfiguration and reducing the risk of corruption from the adversary. Extensive numerical experiments show that, compared to the state-of-the-art commercial solvers, our proposed approach can achieve up to 66.6% higher throughput in less than 1/30 running time. Moreover, the proposed approach can achieve global optimal solutions in most experiments.



## **36. FedGrad: Mitigating Backdoor Attacks in Federated Learning Through Local Ultimate Gradients Inspection**

cs.CV

Accepted for presentation at the International Joint Conference on  Neural Networks (IJCNN 2023)

**SubmitDate**: 2023-04-29    [abs](http://arxiv.org/abs/2305.00328v1) [paper-pdf](http://arxiv.org/pdf/2305.00328v1)

**Authors**: Thuy Dung Nguyen, Anh Duy Nguyen, Kok-Seng Wong, Huy Hieu Pham, Thanh Hung Nguyen, Phi Le Nguyen, Truong Thao Nguyen

**Abstract**: Federated learning (FL) enables multiple clients to train a model without compromising sensitive data. The decentralized nature of FL makes it susceptible to adversarial attacks, especially backdoor insertion during training. Recently, the edge-case backdoor attack employing the tail of the data distribution has been proposed as a powerful one, raising questions about the shortfall in current defenses' robustness guarantees. Specifically, most existing defenses cannot eliminate edge-case backdoor attacks or suffer from a trade-off between backdoor-defending effectiveness and overall performance on the primary task. To tackle this challenge, we propose FedGrad, a novel backdoor-resistant defense for FL that is resistant to cutting-edge backdoor attacks, including the edge-case attack, and performs effectively under heterogeneous client data and a large number of compromised clients. FedGrad is designed as a two-layer filtering mechanism that thoroughly analyzes the ultimate layer's gradient to identify suspicious local updates and remove them from the aggregation process. We evaluate FedGrad under different attack scenarios and show that it significantly outperforms state-of-the-art defense mechanisms. Notably, FedGrad can almost 100% correctly detect the malicious participants, thus providing a significant reduction in the backdoor effect (e.g., backdoor accuracy is less than 8%) while not reducing the main accuracy on the primary task.



## **37. Game Theoretic Mixed Experts for Combinational Adversarial Machine Learning**

cs.LG

17pages, 10 figures

**SubmitDate**: 2023-04-29    [abs](http://arxiv.org/abs/2211.14669v2) [paper-pdf](http://arxiv.org/pdf/2211.14669v2)

**Authors**: Ethan Rathbun, Kaleel Mahmood, Sohaib Ahmad, Caiwen Ding, Marten van Dijk

**Abstract**: Recent advances in adversarial machine learning have shown that defenses considered to be robust are actually susceptible to adversarial attacks which are specifically customized to target their weaknesses. These defenses include Barrage of Random Transforms (BaRT), Friendly Adversarial Training (FAT), Trash is Treasure (TiT) and ensemble models made up of Vision Transformers (ViTs), Big Transfer models and Spiking Neural Networks (SNNs). We first conduct a transferability analysis, to demonstrate the adversarial examples generated by customized attacks on one defense, are not often misclassified by another defense.   This finding leads to two important questions. First, how can the low transferability between defenses be utilized in a game theoretic framework to improve the robustness? Second, how can an adversary within this framework develop effective multi-model attacks? In this paper, we provide a game-theoretic framework for ensemble adversarial attacks and defenses. Our framework is called Game theoretic Mixed Experts (GaME). It is designed to find the Mixed-Nash strategy for both a detector based and standard defender, when facing an attacker employing compositional adversarial attacks. We further propose three new attack algorithms, specifically designed to target defenses with randomized transformations, multi-model voting schemes, and adversarial detector architectures. These attacks serve to both strengthen defenses generated by the GaME framework and verify their robustness against unforeseen attacks. Overall, our framework and analyses advance the field of adversarial machine learning by yielding new insights into compositional attack and defense formulations.



## **38. Improving Hyperspectral Adversarial Robustness Under Multiple Attacks**

cs.LG

6 pages, 2 figures, 1 table, 1 algorithm

**SubmitDate**: 2023-04-28    [abs](http://arxiv.org/abs/2210.16346v3) [paper-pdf](http://arxiv.org/pdf/2210.16346v3)

**Authors**: Nicholas Soucy, Salimeh Yasaei Sekeh

**Abstract**: Semantic segmentation models classifying hyperspectral images (HSI) are vulnerable to adversarial examples. Traditional approaches to adversarial robustness focus on training or retraining a single network on attacked data, however, in the presence of multiple attacks these approaches decrease in performance compared to networks trained individually on each attack. To combat this issue we propose an Adversarial Discriminator Ensemble Network (ADE-Net) which focuses on attack type detection and adversarial robustness under a unified model to preserve per data-type weight optimally while robustifiying the overall network. In the proposed method, a discriminator network is used to separate data by attack type into their specific attack-expert ensemble network.



## **39. The Power of Typed Affine Decision Structures: A Case Study**

cs.LG

**SubmitDate**: 2023-04-28    [abs](http://arxiv.org/abs/2304.14888v1) [paper-pdf](http://arxiv.org/pdf/2304.14888v1)

**Authors**: Gerrit Nolte, Maximilian Schlüter, Alnis Murtovi, Bernhard Steffen

**Abstract**: TADS are a novel, concise white-box representation of neural networks. In this paper, we apply TADS to the problem of neural network verification, using them to generate either proofs or concise error characterizations for desirable neural network properties. In a case study, we consider the robustness of neural networks to adversarial attacks, i.e., small changes to an input that drastically change a neural networks perception, and show that TADS can be used to provide precise diagnostics on how and where robustness errors a occur. We achieve these results by introducing Precondition Projection, a technique that yields a TADS describing network behavior precisely on a given subset of its input space, and combining it with PCA, a traditional, well-understood dimensionality reduction technique. We show that PCA is easily compatible with TADS. All analyses can be implemented in a straightforward fashion using the rich algebraic properties of TADS, demonstrating the utility of the TADS framework for neural network explainability and verification. While TADS do not yet scale as efficiently as state-of-the-art neural network verifiers, we show that, using PCA-based simplifications, they can still scale to mediumsized problems and yield concise explanations for potential errors that can be used for other purposes such as debugging a network or generating new training samples.



## **40. Topic-oriented Adversarial Attacks against Black-box Neural Ranking Models**

cs.IR

Accepted by SIGIR 2023

**SubmitDate**: 2023-04-28    [abs](http://arxiv.org/abs/2304.14867v1) [paper-pdf](http://arxiv.org/pdf/2304.14867v1)

**Authors**: Yu-An Liu, Ruqing Zhang, Jiafeng Guo, Maarten de Rijke, Wei Chen, Yixing Fan, Xueqi Cheng

**Abstract**: Neural ranking models (NRMs) have attracted considerable attention in information retrieval. Unfortunately, NRMs may inherit the adversarial vulnerabilities of general neural networks, which might be leveraged by black-hat search engine optimization practitioners. Recently, adversarial attacks against NRMs have been explored in the paired attack setting, generating an adversarial perturbation to a target document for a specific query. In this paper, we focus on a more general type of perturbation and introduce the topic-oriented adversarial ranking attack task against NRMs, which aims to find an imperceptible perturbation that can promote a target document in ranking for a group of queries with the same topic. We define both static and dynamic settings for the task and focus on decision-based black-box attacks. We propose a novel framework to improve topic-oriented attack performance based on a surrogate ranking model. The attack problem is formalized as a Markov decision process (MDP) and addressed using reinforcement learning. Specifically, a topic-oriented reward function guides the policy to find a successful adversarial example that can be promoted in rankings to as many queries as possible in a group. Experimental results demonstrate that the proposed framework can significantly outperform existing attack strategies, and we conclude by re-iterating that there exist potential risks for applying NRMs in the real world.



## **41. False Claims against Model Ownership Resolution**

cs.CR

13pages,3 figures

**SubmitDate**: 2023-04-28    [abs](http://arxiv.org/abs/2304.06607v2) [paper-pdf](http://arxiv.org/pdf/2304.06607v2)

**Authors**: Jian Liu, Rui Zhang, Sebastian Szyller, Kui Ren, N. Asokan

**Abstract**: Deep neural network (DNN) models are valuable intellectual property of model owners, constituting a competitive advantage. Therefore, it is crucial to develop techniques to protect against model theft. Model ownership resolution (MOR) is a class of techniques that can deter model theft. A MOR scheme enables an accuser to assert an ownership claim for a suspect model by presenting evidence, such as a watermark or fingerprint, to show that the suspect model was stolen or derived from a source model owned by the accuser. Most of the existing MOR schemes prioritize robustness against malicious suspects, ensuring that the accuser will win if the suspect model is indeed a stolen model.   In this paper, we show that common MOR schemes in the literature are vulnerable to a different, equally important but insufficiently explored, robustness concern: a malicious accuser. We show how malicious accusers can successfully make false claims against independent suspect models that were not stolen. Our core idea is that a malicious accuser can deviate (without detection) from the specified MOR process by finding (transferable) adversarial examples that successfully serve as evidence against independent suspect models. To this end, we first generalize the procedures of common MOR schemes and show that, under this generalization, defending against false claims is as challenging as preventing (transferable) adversarial examples. Via systematic empirical evaluation we demonstrate that our false claim attacks always succeed in all prominent MOR schemes with realistic configurations, including against a real-world model: Amazon's Rekognition API.



## **42. Certified Robustness of Quantum Classifiers against Adversarial Examples through Quantum Noise**

quant-ph

Accepted to IEEE ICASSP 2023

**SubmitDate**: 2023-04-28    [abs](http://arxiv.org/abs/2211.00887v2) [paper-pdf](http://arxiv.org/pdf/2211.00887v2)

**Authors**: Jhih-Cing Huang, Yu-Lin Tsai, Chao-Han Huck Yang, Cheng-Fang Su, Chia-Mu Yu, Pin-Yu Chen, Sy-Yen Kuo

**Abstract**: Recently, quantum classifiers have been found to be vulnerable to adversarial attacks, in which quantum classifiers are deceived by imperceptible noises, leading to misclassification. In this paper, we propose the first theoretical study demonstrating that adding quantum random rotation noise can improve robustness in quantum classifiers against adversarial attacks. We link the definition of differential privacy and show that the quantum classifier trained with the natural presence of additive noise is differentially private. Finally, we derive a certified robustness bound to enable quantum classifiers to defend against adversarial examples, supported by experimental results simulated with noises from IBM's 7-qubits device.



## **43. Fusion is Not Enough: Single-Modal Attacks to Compromise Fusion Models in Autonomous Driving**

cs.CV

**SubmitDate**: 2023-04-28    [abs](http://arxiv.org/abs/2304.14614v1) [paper-pdf](http://arxiv.org/pdf/2304.14614v1)

**Authors**: Zhiyuan Cheng, Hongjun Choi, James Liang, Shiwei Feng, Guanhong Tao, Dongfang Liu, Michael Zuzak, Xiangyu Zhang

**Abstract**: Multi-sensor fusion (MSF) is widely adopted for perception in autonomous vehicles (AVs), particularly for the task of 3D object detection with camera and LiDAR sensors. The rationale behind fusion is to capitalize on the strengths of each modality while mitigating their limitations. The exceptional and leading performance of fusion models has been demonstrated by advanced deep neural network (DNN)-based fusion techniques. Fusion models are also perceived as more robust to attacks compared to single-modal ones due to the redundant information in multiple modalities. In this work, we challenge this perspective with single-modal attacks that targets the camera modality, which is considered less significant in fusion but more affordable for attackers. We argue that the weakest link of fusion models depends on their most vulnerable modality, and propose an attack framework that targets advanced camera-LiDAR fusion models with adversarial patches. Our approach employs a two-stage optimization-based strategy that first comprehensively assesses vulnerable image areas under adversarial attacks, and then applies customized attack strategies to different fusion models, generating deployable patches. Evaluations with five state-of-the-art camera-LiDAR fusion models on a real-world dataset show that our attacks successfully compromise all models. Our approach can either reduce the mean average precision (mAP) of detection performance from 0.824 to 0.353 or degrade the detection score of the target object from 0.727 to 0.151 on average, demonstrating the effectiveness and practicality of our proposed attack framework.



## **44. Efficient Reward Poisoning Attacks on Online Deep Reinforcement Learning**

cs.LG

**SubmitDate**: 2023-04-28    [abs](http://arxiv.org/abs/2205.14842v2) [paper-pdf](http://arxiv.org/pdf/2205.14842v2)

**Authors**: Yinglun Xu, Qi Zeng, Gagandeep Singh

**Abstract**: We study reward poisoning attacks on online deep reinforcement learning (DRL), where the attacker is oblivious to the learning algorithm used by the agent and the dynamics of the environment. We demonstrate the intrinsic vulnerability of state-of-the-art DRL algorithms by designing a general, black-box reward poisoning framework called adversarial MDP attacks. We instantiate our framework to construct two new attacks which only corrupt the rewards for a small fraction of the total training timesteps and make the agent learn a low-performing policy. We provide a theoretical analysis of the efficiency of our attack and perform an extensive empirical evaluation. Our results show that our attacks efficiently poison agents learning in several popular classical control and MuJoCo environments with a variety of state-of-the-art DRL algorithms, such as DQN, PPO, SAC, etc.



## **45. Adversary Aware Continual Learning**

cs.LG

**SubmitDate**: 2023-04-27    [abs](http://arxiv.org/abs/2304.14483v1) [paper-pdf](http://arxiv.org/pdf/2304.14483v1)

**Authors**: Muhammad Umer, Robi Polikar

**Abstract**: Class incremental learning approaches are useful as they help the model to learn new information (classes) sequentially, while also retaining the previously acquired information (classes). However, it has been shown that such approaches are extremely vulnerable to the adversarial backdoor attacks, where an intelligent adversary can introduce small amount of misinformation to the model in the form of imperceptible backdoor pattern during training to cause deliberate forgetting of a specific task or class at test time. In this work, we propose a novel defensive framework to counter such an insidious attack where, we use the attacker's primary strength-hiding the backdoor pattern by making it imperceptible to humans-against it, and propose to learn a perceptible (stronger) pattern (also during the training) that can overpower the attacker's imperceptible (weaker) pattern. We demonstrate the effectiveness of the proposed defensive mechanism through various commonly used Replay-based (both generative and exact replay-based) class incremental learning algorithms using continual learning benchmark variants of CIFAR-10, CIFAR-100, and MNIST datasets. Most noteworthy, our proposed defensive framework does not assume that the attacker's target task and target class is known to the defender. The defender is also unaware of the shape, size, and location of the attacker's pattern. We show that our proposed defensive framework considerably improves the performance of class incremental learning algorithms with no knowledge of the attacker's target task, attacker's target class, and attacker's imperceptible pattern. We term our defensive framework as Adversary Aware Continual Learning (AACL).



## **46. Attacking Fake News Detectors via Manipulating News Social Engagement**

cs.SI

ACM Web Conference 2023 (WWW'23)

**SubmitDate**: 2023-04-27    [abs](http://arxiv.org/abs/2302.07363v3) [paper-pdf](http://arxiv.org/pdf/2302.07363v3)

**Authors**: Haoran Wang, Yingtong Dou, Canyu Chen, Lichao Sun, Philip S. Yu, Kai Shu

**Abstract**: Social media is one of the main sources for news consumption, especially among the younger generation. With the increasing popularity of news consumption on various social media platforms, there has been a surge of misinformation which includes false information or unfounded claims. As various text- and social context-based fake news detectors are proposed to detect misinformation on social media, recent works start to focus on the vulnerabilities of fake news detectors. In this paper, we present the first adversarial attack framework against Graph Neural Network (GNN)-based fake news detectors to probe their robustness. Specifically, we leverage a multi-agent reinforcement learning (MARL) framework to simulate the adversarial behavior of fraudsters on social media. Research has shown that in real-world settings, fraudsters coordinate with each other to share different news in order to evade the detection of fake news detectors. Therefore, we modeled our MARL framework as a Markov Game with bot, cyborg, and crowd worker agents, which have their own distinctive cost, budget, and influence. We then use deep Q-learning to search for the optimal policy that maximizes the rewards. Extensive experimental results on two real-world fake news propagation datasets demonstrate that our proposed framework can effectively sabotage the GNN-based fake news detector performance. We hope this paper can provide insights for future research on fake news detection.



## **47. On the (In)security of Peer-to-Peer Decentralized Machine Learning**

cs.CR

IEEE S&P'23 (Previous title: "On the Privacy of Decentralized Machine  Learning")

**SubmitDate**: 2023-04-27    [abs](http://arxiv.org/abs/2205.08443v2) [paper-pdf](http://arxiv.org/pdf/2205.08443v2)

**Authors**: Dario Pasquini, Mathilde Raynal, Carmela Troncoso

**Abstract**: In this work, we carry out the first, in-depth, privacy analysis of Decentralized Learning -- a collaborative machine learning framework aimed at addressing the main limitations of federated learning. We introduce a suite of novel attacks for both passive and active decentralized adversaries. We demonstrate that, contrary to what is claimed by decentralized learning proposers, decentralized learning does not offer any security advantage over federated learning. Rather, it increases the attack surface enabling any user in the system to perform privacy attacks such as gradient inversion, and even gain full control over honest users' local model. We also show that, given the state of the art in protections, privacy-preserving configurations of decentralized learning require fully connected networks, losing any practical advantage over the federated setup and therefore completely defeating the objective of the decentralized approach.



## **48. Robust Resilient Signal Reconstruction under Adversarial Attacks**

math.OC

7 pages

**SubmitDate**: 2023-04-27    [abs](http://arxiv.org/abs/1807.08004v2) [paper-pdf](http://arxiv.org/pdf/1807.08004v2)

**Authors**: Yu Zheng, Olugbenga Moses Anubi, Lalit Mestha, Hema Achanta

**Abstract**: We consider the problem of signal reconstruction for a system under sparse signal corruption by a malicious agent. The reconstruction problem follows the standard error coding problem that has been studied extensively in the literature. We include a new challenge of robust estimation of the attack support. The problem is then cast as a constrained optimization problem merging promising techniques in the area of deep learning and estimation theory. A pruning algorithm is developed to reduce the ``false positive" uncertainty of data-driven attack localization results, thereby improving the probability of correct signal reconstruction. Sufficient conditions for the correct reconstruction and the associated reconstruction error bounds are obtained for both exact and inexact attack support estimation. Moreover, a simulation of a water distribution system is presented to validate the proposed techniques.



## **49. QEVSEC: Quick Electric Vehicle SEcure Charging via Dynamic Wireless Power Transfer**

cs.CR

6 pages, conference

**SubmitDate**: 2023-04-27    [abs](http://arxiv.org/abs/2205.10292v2) [paper-pdf](http://arxiv.org/pdf/2205.10292v2)

**Authors**: Tommaso Bianchi, Surudhi Asokraj, Alessandro Brighente, Mauro Conti, Radha Poovendran

**Abstract**: Dynamic Wireless Power Transfer (DWPT) can be used for on-demand recharging of Electric Vehicles (EV) while driving. However, DWPT raises numerous security and privacy concerns. Recently, researchers demonstrated that DWPT systems are vulnerable to adversarial attacks. In an EV charging scenario, an attacker can prevent the authorized customer from charging, obtain a free charge by billing a victim user and track a target vehicle. State-of-the-art authentication schemes relying on centralized solutions are either vulnerable to various attacks or have high computational complexity, making them unsuitable for a dynamic scenario. In this paper, we propose Quick Electric Vehicle SEcure Charging (QEVSEC), a novel, secure, and efficient authentication protocol for the dynamic charging of EVs. Our idea for QEVSEC originates from multiple vulnerabilities we found in the state-of-the-art protocol that allows tracking of user activity and is susceptible to replay attacks. Based on these observations, the proposed protocol solves these issues and achieves lower computational complexity by using only primitive cryptographic operations in a very short message exchange. QEVSEC provides scalability and a reduced cost in each iteration, thus lowering the impact on the power needed from the grid.



## **50. Boosting Big Brother: Attacking Search Engines with Encodings**

cs.CR

**SubmitDate**: 2023-04-27    [abs](http://arxiv.org/abs/2304.14031v1) [paper-pdf](http://arxiv.org/pdf/2304.14031v1)

**Authors**: Nicholas Boucher, Luca Pajola, Ilia Shumailov, Ross Anderson, Mauro Conti

**Abstract**: Search engines are vulnerable to attacks against indexing and searching via text encoding manipulation. By imperceptibly perturbing text using uncommon encoded representations, adversaries can control results across search engines for specific search queries. We demonstrate that this attack is successful against two major commercial search engines - Google and Bing - and one open source search engine - Elasticsearch. We further demonstrate that this attack is successful against LLM chat search including Bing's GPT-4 chatbot and Google's Bard chatbot. We also present a variant of the attack targeting text summarization and plagiarism detection models, two ML tasks closely tied to search. We provide a set of defenses against these techniques and warn that adversaries can leverage these attacks to launch disinformation campaigns against unsuspecting users, motivating the need for search engine maintainers to patch deployed systems.



