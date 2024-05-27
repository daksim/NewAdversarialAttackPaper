# Latest Adversarial Attack Papers
**update at 2024-05-27 09:25:42**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. $$\mathbf{L^2\cdot M = C^2}$$ Large Language Models as Covert Channels... a Systematic Analysis**

cs.CR

**SubmitDate**: 2024-05-24    [abs](http://arxiv.org/abs/2405.15652v1) [paper-pdf](http://arxiv.org/pdf/2405.15652v1)

**Authors**: Simen Gaure, Stefanos Koffas, Stjepan Picek, Sondre Rønjom

**Abstract**: Large Language Models (LLMs) have gained significant popularity in the last few years due to their performance in diverse tasks such as translation, prediction, or content generation. At the same time, the research community has shown that LLMs are susceptible to various attacks but can also improve the security of diverse systems. However, besides enabling more secure systems, how well do open source LLMs behave as covertext distributions to, e.g., facilitate censorship resistant communication?   In this paper, we explore the capabilities of open-source LLM-based covert channels. We approach this problem from the experimental side by empirically measuring the security vs. capacity of the open-source LLM model (Llama-7B) to assess how well it performs as a covert channel. Although our results indicate that such channels are not likely to achieve high practical bitrates, which depend on message length and model entropy, we also show that the chance for an adversary to detect covert communication is low. To ensure that our results can be used with the least effort as a general reference, we employ a conceptually simple and concise scheme and only assume public models.



## **2. Efficient Adversarial Training in LLMs with Continuous Attacks**

cs.LG

19 pages, 4 figures

**SubmitDate**: 2024-05-24    [abs](http://arxiv.org/abs/2405.15589v1) [paper-pdf](http://arxiv.org/pdf/2405.15589v1)

**Authors**: Sophie Xhonneux, Alessandro Sordoni, Stephan Günnemann, Gauthier Gidel, Leo Schwinn

**Abstract**: Large language models (LLMs) are vulnerable to adversarial attacks that can bypass their safety guardrails. In many domains, adversarial training has proven to be one of the most promising methods to reliably improve robustness against such attacks. Yet, in the context of LLMs, current methods for adversarial training are hindered by the high computational costs required to perform discrete adversarial attacks at each training iteration. We address this problem by instead calculating adversarial attacks in the continuous embedding space of the LLM, which is orders of magnitudes more efficient. We propose a fast adversarial training algorithm (C-AdvUL) composed of two losses: the first makes the model robust on continuous embedding attacks computed on an adversarial behaviour dataset; the second ensures the usefulness of the final model by fine-tuning on utility data. Moreover, we introduce C-AdvIPO, an adversarial variant of IPO that does not require utility data for adversarially robust alignment. Our empirical evaluation on four models from different families (Gemma, Phi3, Mistral, Zephyr) and at different scales (2B, 3.8B, 7B) shows that both algorithms substantially enhance LLM robustness against discrete attacks (GCG, AutoDAN, PAIR), while maintaining utility. Our results demonstrate that robustness to continuous perturbations can extrapolate to discrete threat models. Thereby, we present a path toward scalable adversarial training algorithms for robustly aligning LLMs.



## **3. Rethinking Independent Cross-Entropy Loss For Graph-Structured Data**

cs.LG

20 pages, 4 figures

**SubmitDate**: 2024-05-24    [abs](http://arxiv.org/abs/2405.15564v1) [paper-pdf](http://arxiv.org/pdf/2405.15564v1)

**Authors**: Rui Miao, Kaixiong Zhou, Yili Wang, Ninghao Liu, Ying Wang, Xin Wang

**Abstract**: Graph neural networks (GNNs) have exhibited prominent performance in learning graph-structured data. Considering node classification task, based on the i.i.d assumption among node labels, the traditional supervised learning simply sums up cross-entropy losses of the independent training nodes and applies the average loss to optimize GNNs' weights. But different from other data formats, the nodes are naturally connected. It is found that the independent distribution modeling of node labels restricts GNNs' capability to generalize over the entire graph and defend adversarial attacks. In this work, we propose a new framework, termed joint-cluster supervised learning, to model the joint distribution of each node with its corresponding cluster. We learn the joint distribution of node and cluster labels conditioned on their representations, and train GNNs with the obtained joint loss. In this way, the data-label reference signals extracted from the local cluster explicitly strengthen the discrimination ability on the target node. The extensive experiments demonstrate that our joint-cluster supervised learning can effectively bolster GNNs' node classification accuracy. Furthermore, being benefited from the reference signals which may be free from spiteful interference, our learning paradigm significantly protects the node classification from being affected by the adversarial attack.



## **4. AuthNet: Neural Network with Integrated Authentication Logic**

cs.CR

**SubmitDate**: 2024-05-24    [abs](http://arxiv.org/abs/2405.15426v1) [paper-pdf](http://arxiv.org/pdf/2405.15426v1)

**Authors**: Yuling Cai, Fan Xiang, Guozhu Meng, Yinzhi Cao, Kai Chen

**Abstract**: Model stealing, i.e., unauthorized access and exfiltration of deep learning models, has become one of the major threats. Proprietary models may be protected by access controls and encryption. However, in reality, these measures can be compromised due to system breaches, query-based model extraction or a disgruntled insider. Security hardening of neural networks is also suffering from limits, for example, model watermarking is passive, cannot prevent the occurrence of piracy and not robust against transformations. To this end, we propose a native authentication mechanism, called AuthNet, which integrates authentication logic as part of the model without any additional structures. Our key insight is to reuse redundant neurons with low activation and embed authentication bits in an intermediate layer, called a gate layer. Then, AuthNet fine-tunes the layers after the gate layer to embed authentication logic so that only inputs with special secret key can trigger the correct logic of AuthNet. It exhibits two intuitive advantages. It provides the last line of defense, i.e., even being exfiltrated, the model is not usable as the adversary cannot generate valid inputs without the key. Moreover, the authentication logic is difficult to inspect and identify given millions or billions of neurons in the model. We theoretically demonstrate the high sensitivity of AuthNet to the secret key and its high confusion for unauthorized samples. AuthNet is compatible with any convolutional neural network, where our extensive evaluations show that AuthNet successfully achieves the goal in rejecting unauthenticated users (whose average accuracy drops to 22.03%) with a trivial accuracy decrease (1.18% on average) for legitimate users, and is robust against model transformation and adaptive attacks.



## **5. Lost in the Averages: A New Specific Setup to Evaluate Membership Inference Attacks Against Machine Learning Models**

cs.LG

**SubmitDate**: 2024-05-24    [abs](http://arxiv.org/abs/2405.15423v1) [paper-pdf](http://arxiv.org/pdf/2405.15423v1)

**Authors**: Florent Guépin, Nataša Krčo, Matthieu Meeus, Yves-Alexandre de Montjoye

**Abstract**: Membership Inference Attacks (MIAs) are widely used to evaluate the propensity of a machine learning (ML) model to memorize an individual record and the privacy risk releasing the model poses. MIAs are commonly evaluated similarly to ML models: the MIA is performed on a test set of models trained on datasets unseen during training, which are sampled from a larger pool, $D_{eval}$. The MIA is evaluated across all datasets in this test set, and is thus evaluated across the distribution of samples from $D_{eval}$. While this was a natural extension of ML evaluation to MIAs, recent work has shown that a record's risk heavily depends on its specific dataset. For example, outliers are particularly vulnerable, yet an outlier in one dataset may not be one in another. The sources of randomness currently used to evaluate MIAs may thus lead to inaccurate individual privacy risk estimates. We propose a new, specific evaluation setup for MIAs against ML models, using weight initialization as the sole source of randomness. This allows us to accurately evaluate the risk associated with the release of a model trained on a specific dataset. Using SOTA MIAs, we empirically show that the risk estimates given by the current setup lead to many records being misclassified as low risk. We derive theoretical results which, combined with empirical evidence, suggest that the risk calculated in the current setup is an average of the risks specific to each sampled dataset, validating our use of weight initialization as the only source of randomness. Finally, we consider an MIA with a stronger adversary leveraging information about the target dataset to infer membership. Taken together, our results show that current MIA evaluation is averaging the risk across datasets leading to inaccurate risk estimates, and the risk posed by attacks leveraging information about the target dataset to be potentially underestimated.



## **6. Decaf: Data Distribution Decompose Attack against Federated Learning**

cs.LG

**SubmitDate**: 2024-05-24    [abs](http://arxiv.org/abs/2405.15316v1) [paper-pdf](http://arxiv.org/pdf/2405.15316v1)

**Authors**: Zhiyang Dai, Chunyi Zhou, Anmin Fu

**Abstract**: In contrast to prevalent Federated Learning (FL) privacy inference techniques such as generative adversarial networks attacks, membership inference attacks, property inference attacks, and model inversion attacks, we devise an innovative privacy threat: the Data Distribution Decompose Attack on FL, termed Decaf. This attack enables an honest-but-curious FL server to meticulously profile the proportion of each class owned by the victim FL user, divulging sensitive information like local market item distribution and business competitiveness. The crux of Decaf lies in the profound observation that the magnitude of local model gradient changes closely mirrors the underlying data distribution, including the proportion of each class. Decaf addresses two crucial challenges: accurately identify the missing/null class(es) given by any victim user as a premise and then quantify the precise relationship between gradient changes and each remaining non-null class. Notably, Decaf operates stealthily, rendering it entirely passive and undetectable to victim users regarding the infringement of their data distribution privacy. Experimental validation on five benchmark datasets (MNIST, FASHION-MNIST, CIFAR-10, FER-2013, and SkinCancer) employing diverse model architectures, including customized convolutional networks, standardized VGG16, and ResNet18, demonstrates Decaf's efficacy. Results indicate its ability to accurately decompose local user data distribution, regardless of whether it is IID or non-IID distributed. Specifically, the dissimilarity measured using $L_{\infty}$ distance between the distribution decomposed by Decaf and ground truth is consistently below 5\% when no null classes exist. Moreover, Decaf achieves 100\% accuracy in determining any victim user's null classes, validated through formal proof.



## **7. Robust Diffusion Models for Adversarial Purification**

cs.CV

**SubmitDate**: 2024-05-24    [abs](http://arxiv.org/abs/2403.16067v2) [paper-pdf](http://arxiv.org/pdf/2403.16067v2)

**Authors**: Guang Lin, Zerui Tao, Jianhai Zhang, Toshihisa Tanaka, Qibin Zhao

**Abstract**: Diffusion models (DMs) based adversarial purification (AP) has shown to be the most powerful alternative to adversarial training (AT). However, these methods neglect the fact that pre-trained diffusion models themselves are not robust to adversarial attacks as well. Additionally, the diffusion process can easily destroy semantic information and generate a high quality image but totally different from the original input image after the reverse process, leading to degraded standard accuracy. To overcome these issues, a natural idea is to harness adversarial training strategy to retrain or fine-tune the pre-trained diffusion model, which is computationally prohibitive. We propose a novel robust reverse process with adversarial guidance, which is independent of given pre-trained DMs and avoids retraining or fine-tuning the DMs. This robust guidance can not only ensure to generate purified examples retaining more semantic content but also mitigate the accuracy-robustness trade-off of DMs for the first time, which also provides DM-based AP an efficient adaptive ability to new attacks. Extensive experiments are conducted on CIFAR-10, CIFAR-100 and ImageNet to demonstrate that our method achieves the state-of-the-art results and exhibits generalization against different attacks.



## **8. TrojanRAG: Retrieval-Augmented Generation Can Be Backdoor Driver in Large Language Models**

cs.CR

18 pages, 13 figures, 4 tables

**SubmitDate**: 2024-05-24    [abs](http://arxiv.org/abs/2405.13401v2) [paper-pdf](http://arxiv.org/pdf/2405.13401v2)

**Authors**: Pengzhou Cheng, Yidong Ding, Tianjie Ju, Zongru Wu, Wei Du, Ping Yi, Zhuosheng Zhang, Gongshen Liu

**Abstract**: Large language models (LLMs) have raised concerns about potential security threats despite performing significantly in Natural Language Processing (NLP). Backdoor attacks initially verified that LLM is doing substantial harm at all stages, but the cost and robustness have been criticized. Attacking LLMs is inherently risky in security review, while prohibitively expensive. Besides, the continuous iteration of LLMs will degrade the robustness of backdoors. In this paper, we propose TrojanRAG, which employs a joint backdoor attack in the Retrieval-Augmented Generation, thereby manipulating LLMs in universal attack scenarios. Specifically, the adversary constructs elaborate target contexts and trigger sets. Multiple pairs of backdoor shortcuts are orthogonally optimized by contrastive learning, thus constraining the triggering conditions to a parameter subspace to improve the matching. To improve the recall of the RAG for the target contexts, we introduce a knowledge graph to construct structured data to achieve hard matching at a fine-grained level. Moreover, we normalize the backdoor scenarios in LLMs to analyze the real harm caused by backdoors from both attackers' and users' perspectives and further verify whether the context is a favorable tool for jailbreaking models. Extensive experimental results on truthfulness, language understanding, and harmfulness show that TrojanRAG exhibits versatility threats while maintaining retrieval capabilities on normal queries.



## **9. Adversarial Attacks on Hidden Tasks in Multi-Task Learning**

cs.LG

14 pages, 6 figures

**SubmitDate**: 2024-05-24    [abs](http://arxiv.org/abs/2405.15244v1) [paper-pdf](http://arxiv.org/pdf/2405.15244v1)

**Authors**: Yu Zhe, Rei Nagaike, Daiki Nishiyama, Kazuto Fukuchi, Jun Sakuma

**Abstract**: Deep learning models are susceptible to adversarial attacks, where slight perturbations to input data lead to misclassification. Adversarial attacks become increasingly effective with access to information about the targeted classifier. In the context of multi-task learning, where a single model learns multiple tasks simultaneously, attackers may aim to exploit vulnerabilities in specific tasks with limited information. This paper investigates the feasibility of attacking hidden tasks within multi-task classifiers, where model access regarding the hidden target task and labeled data for the hidden target task are not available, but model access regarding the non-target tasks is available. We propose a novel adversarial attack method that leverages knowledge from non-target tasks and the shared backbone network of the multi-task model to force the model to forget knowledge related to the target task. Experimental results on CelebA and DeepFashion datasets demonstrate the effectiveness of our method in degrading the accuracy of hidden tasks while preserving the performance of visible tasks, contributing to the understanding of adversarial vulnerabilities in multi-task classifiers.



## **10. Defensive Unlearning with Adversarial Training for Robust Concept Erasure in Diffusion Models**

cs.CV

Codes are available at https://github.com/OPTML-Group/AdvUnlearn

**SubmitDate**: 2024-05-24    [abs](http://arxiv.org/abs/2405.15234v1) [paper-pdf](http://arxiv.org/pdf/2405.15234v1)

**Authors**: Yimeng Zhang, Xin Chen, Jinghan Jia, Yihua Zhang, Chongyu Fan, Jiancheng Liu, Mingyi Hong, Ke Ding, Sijia Liu

**Abstract**: Diffusion models (DMs) have achieved remarkable success in text-to-image generation, but they also pose safety risks, such as the potential generation of harmful content and copyright violations. The techniques of machine unlearning, also known as concept erasing, have been developed to address these risks. However, these techniques remain vulnerable to adversarial prompt attacks, which can prompt DMs post-unlearning to regenerate undesired images containing concepts (such as nudity) meant to be erased. This work aims to enhance the robustness of concept erasing by integrating the principle of adversarial training (AT) into machine unlearning, resulting in the robust unlearning framework referred to as AdvUnlearn. However, achieving this effectively and efficiently is highly nontrivial. First, we find that a straightforward implementation of AT compromises DMs' image generation quality post-unlearning. To address this, we develop a utility-retaining regularization on an additional retain set, optimizing the trade-off between concept erasure robustness and model utility in AdvUnlearn. Moreover, we identify the text encoder as a more suitable module for robustification compared to UNet, ensuring unlearning effectiveness. And the acquired text encoder can serve as a plug-and-play robust unlearner for various DM types. Empirically, we perform extensive experiments to demonstrate the robustness advantage of AdvUnlearn across various DM unlearning scenarios, including the erasure of nudity, objects, and style concepts. In addition to robustness, AdvUnlearn also achieves a balanced tradeoff with model utility. To our knowledge, this is the first work to systematically explore robust DM unlearning through AT, setting it apart from existing methods that overlook robustness in concept erasing. Codes are available at: https://github.com/OPTML-Group/AdvUnlearn



## **11. TrojanForge: Adversarial Hardware Trojan Examples with Reinforcement Learning**

cs.CR

**SubmitDate**: 2024-05-24    [abs](http://arxiv.org/abs/2405.15184v1) [paper-pdf](http://arxiv.org/pdf/2405.15184v1)

**Authors**: Amin Sarihi, Peter Jamieson, Ahmad Patooghy, Abdel-Hameed A. Badawy

**Abstract**: The Hardware Trojan (HT) problem can be thought of as a continuous game between attackers and defenders, each striving to outsmart the other by leveraging any available means for an advantage. Machine Learning (ML) has recently been key in advancing HT research. Various novel techniques, such as Reinforcement Learning (RL) and Graph Neural Networks (GNNs), have shown HT insertion and detection capabilities. HT insertion with ML techniques, specifically, has seen a spike in research activity due to the shortcomings of conventional HT benchmarks and the inherent human design bias that occurs when we create them. This work continues this innovation by presenting a tool called "TrojanForge", capable of generating HT adversarial examples that defeat HT detectors; demonstrating the capabilities of GAN-like adversarial tools for automatic HT insertion. We introduce an RL environment where the RL insertion agent interacts with HT detectors in an insertion-detection loop where the agent collects rewards based on its success in bypassing HT detectors. Our results show that this process leads to inserted HTs that evade various HT detectors, achieving high attack success percentages. This tool provides insight into why HT insertion fails in some instances and how we can leverage this knowledge in defense.



## **12. Are You Copying My Prompt? Protecting the Copyright of Vision Prompt for VPaaS via Watermark**

cs.CR

11 pages, 7 figures,

**SubmitDate**: 2024-05-24    [abs](http://arxiv.org/abs/2405.15161v1) [paper-pdf](http://arxiv.org/pdf/2405.15161v1)

**Authors**: Huali Ren, Anli Yan, Chong-zhi Gao, Hongyang Yan, Zhenxin Zhang, Jin Li

**Abstract**: Visual Prompt Learning (VPL) differs from traditional fine-tuning methods in reducing significant resource consumption by avoiding updating pre-trained model parameters. Instead, it focuses on learning an input perturbation, a visual prompt, added to downstream task data for making predictions. Since learning generalizable prompts requires expert design and creation, which is technically demanding and time-consuming in the optimization process, developers of Visual Prompts as a Service (VPaaS) have emerged. These developers profit by providing well-crafted prompts to authorized customers. However, a significant drawback is that prompts can be easily copied and redistributed, threatening the intellectual property of VPaaS developers. Hence, there is an urgent need for technology to protect the rights of VPaaS developers. To this end, we present a method named \textbf{WVPrompt} that employs visual prompt watermarking in a black-box way. WVPrompt consists of two parts: prompt watermarking and prompt verification. Specifically, it utilizes a poison-only backdoor attack method to embed a watermark into the prompt and then employs a hypothesis-testing approach for remote verification of prompt ownership. Extensive experiments have been conducted on three well-known benchmark datasets using three popular pre-trained models: RN50, BIT-M, and Instagram. The experimental results demonstrate that WVPrompt is efficient, harmless, and robust to various adversarial operations.



## **13. Quantum Public-Key Encryption with Tamper-Resilient Public Keys from One-Way Functions**

quant-ph

47pages

**SubmitDate**: 2024-05-23    [abs](http://arxiv.org/abs/2304.01800v4) [paper-pdf](http://arxiv.org/pdf/2304.01800v4)

**Authors**: Fuyuki Kitagawa, Tomoyuki Morimae, Ryo Nishimaki, Takashi Yamakawa

**Abstract**: We construct quantum public-key encryption from one-way functions. In our construction, public keys are quantum, but ciphertexts are classical. Quantum public-key encryption from one-way functions (or weaker primitives such as pseudorandom function-like states) are also proposed in some recent works [Morimae-Yamakawa, eprint:2022/1336; Coladangelo, eprint:2023/282; Barooti-Grilo-Malavolta-Sattath-Vu-Walter, eprint:2023/877]. However, they have a huge drawback: they are secure only when quantum public keys can be transmitted to the sender (who runs the encryption algorithm) without being tampered with by the adversary, which seems to require unsatisfactory physical setup assumptions such as secure quantum channels. Our construction is free from such a drawback: it guarantees the secrecy of the encrypted messages even if we assume only unauthenticated quantum channels. Thus, the encryption is done with adversarially tampered quantum public keys. Our construction is the first quantum public-key encryption that achieves the goal of classical public-key encryption, namely, to establish secure communication over insecure channels, based only on one-way functions. Moreover, we show a generic compiler to upgrade security against chosen plaintext attacks (CPA security) into security against chosen ciphertext attacks (CCA security) only using one-way functions. As a result, we obtain CCA secure quantum public-key encryption based only on one-way functions.



## **14. Universal Robustness via Median Randomized Smoothing for Real-World Super-Resolution**

eess.IV

**SubmitDate**: 2024-05-23    [abs](http://arxiv.org/abs/2405.14934v1) [paper-pdf](http://arxiv.org/pdf/2405.14934v1)

**Authors**: Zakariya Chaouai, Mohamed Tamaazousti

**Abstract**: Most of the recent literature on image Super-Resolution (SR) can be classified into two main approaches. The first one involves learning a corruption model tailored to a specific dataset, aiming to mimic the noise and corruption in low-resolution images, such as sensor noise. However, this approach is data-specific, tends to lack adaptability, and its accuracy diminishes when faced with unseen types of image corruptions. A second and more recent approach, referred to as Robust Super-Resolution (RSR), proposes to improve real-world SR by harnessing the generalization capabilities of a model by making it robust to adversarial attacks. To delve further into this second approach, our paper explores the universality of various methods for enhancing the robustness of deep learning SR models. In other words, we inquire: "Which robustness method exhibits the highest degree of adaptability when dealing with a wide range of adversarial attacks ?". Our extensive experimentation on both synthetic and real-world images empirically demonstrates that median randomized smoothing (MRS) is more general in terms of robustness compared to adversarial learning techniques, which tend to focus on specific types of attacks. Furthermore, as expected, we also illustrate that the proposed universal robust method enables the SR model to handle standard corruptions more effectively, such as blur and Gaussian noise, and notably, corruptions naturally present in real-world images. These results support the significance of shifting the paradigm in the development of real-world SR methods towards RSR, especially via MRS.



## **15. Overcoming the Challenges of Batch Normalization in Federated Learning**

cs.LG

**SubmitDate**: 2024-05-23    [abs](http://arxiv.org/abs/2405.14670v1) [paper-pdf](http://arxiv.org/pdf/2405.14670v1)

**Authors**: Rachid Guerraoui, Rafael Pinot, Geovani Rizk, John Stephan, François Taiani

**Abstract**: Batch normalization has proven to be a very beneficial mechanism to accelerate the training and improve the accuracy of deep neural networks in centralized environments. Yet, the scheme faces significant challenges in federated learning, especially under high data heterogeneity. Essentially, the main challenges arise from external covariate shifts and inconsistent statistics across clients. We introduce in this paper Federated BatchNorm (FBN), a novel scheme that restores the benefits of batch normalization in federated learning. Essentially, FBN ensures that the batch normalization during training is consistent with what would be achieved in a centralized execution, hence preserving the distribution of the data, and providing running statistics that accurately approximate the global statistics. FBN thereby reduces the external covariate shift and matches the evaluation performance of the centralized setting. We also show that, with a slight increase in complexity, we can robustify FBN to mitigate erroneous statistics and potentially adversarial attacks.



## **16. A New Formulation for Zeroth-Order Optimization of Adversarial EXEmples in Malware Detection**

cs.LG

**SubmitDate**: 2024-05-23    [abs](http://arxiv.org/abs/2405.14519v1) [paper-pdf](http://arxiv.org/pdf/2405.14519v1)

**Authors**: Marco Rando, Luca Demetrio, Lorenzo Rosasco, Fabio Roli

**Abstract**: Machine learning malware detectors are vulnerable to adversarial EXEmples, i.e. carefully-crafted Windows programs tailored to evade detection. Unlike other adversarial problems, attacks in this context must be functionality-preserving, a constraint which is challenging to address. As a consequence heuristic algorithms are typically used, that inject new content, either randomly-picked or harvested from legitimate programs. In this paper, we show how learning malware detectors can be cast within a zeroth-order optimization framework which allows to incorporate functionality-preserving manipulations. This permits the deployment of sound and efficient gradient-free optimization algorithms, which come with theoretical guarantees and allow for minimal hyper-parameters tuning. As a by-product, we propose and study ZEXE, a novel zero-order attack against Windows malware detection. Compared to state-of-the-art techniques, ZEXE provides drastic improvement in the evasion rate, while reducing to less than one third the size of the injected content.



## **17. SLIFER: Investigating Performance and Robustness of Malware Detection Pipelines**

cs.CR

**SubmitDate**: 2024-05-23    [abs](http://arxiv.org/abs/2405.14478v1) [paper-pdf](http://arxiv.org/pdf/2405.14478v1)

**Authors**: Andrea Ponte, Dmitrijs Trizna, Luca Demetrio, Battista Biggio, Fabio Roli

**Abstract**: As a result of decades of research, Windows malware detection is approached through a plethora of techniques. However, there is an ongoing mismatch between academia -- which pursues an optimal performances in terms of detection rate and low false alarms -- and the requirements of real-world scenarios. In particular, academia focuses on combining static and dynamic analysis within a single or ensemble of models, falling into several pitfalls like (i) firing dynamic analysis without considering the computational burden it requires; (ii) discarding impossible-to-analyse samples; and (iii) analysing robustness against adversarial attacks without considering that malware detectors are complemented with more non-machine-learning components. Thus, in this paper we propose SLIFER, a novel Windows malware detection pipeline sequentially leveraging both static and dynamic analysis, interrupting computations as soon as one module triggers an alarm, requiring dynamic analysis only when needed. Contrary to the state of the art, we investigate how to deal with samples resistance to analysis, showing how much they impact performances, concluding that it is better to flag them as legitimate to not drastically increase false alarms. Lastly, we perform a robustness evaluation of SLIFER leveraging content-injections attacks, and we show that, counter-intuitively, attacks are blocked more by YARA rules than dynamic analysis due to byte artifacts created while optimizing the adversarial strategy.



## **18. BadPart: Unified Black-box Adversarial Patch Attacks against Pixel-wise Regression Tasks**

cs.CV

Paper accepted at ICML 2024

**SubmitDate**: 2024-05-23    [abs](http://arxiv.org/abs/2404.00924v2) [paper-pdf](http://arxiv.org/pdf/2404.00924v2)

**Authors**: Zhiyuan Cheng, Zhaoyi Liu, Tengda Guo, Shiwei Feng, Dongfang Liu, Mingjie Tang, Xiangyu Zhang

**Abstract**: Pixel-wise regression tasks (e.g., monocular depth estimation (MDE) and optical flow estimation (OFE)) have been widely involved in our daily life in applications like autonomous driving, augmented reality and video composition. Although certain applications are security-critical or bear societal significance, the adversarial robustness of such models are not sufficiently studied, especially in the black-box scenario. In this work, we introduce the first unified black-box adversarial patch attack framework against pixel-wise regression tasks, aiming to identify the vulnerabilities of these models under query-based black-box attacks. We propose a novel square-based adversarial patch optimization framework and employ probabilistic square sampling and score-based gradient estimation techniques to generate the patch effectively and efficiently, overcoming the scalability problem of previous black-box patch attacks. Our attack prototype, named BadPart, is evaluated on both MDE and OFE tasks, utilizing a total of 7 models. BadPart surpasses 3 baseline methods in terms of both attack performance and efficiency. We also apply BadPart on the Google online service for portrait depth estimation, causing 43.5% relative distance error with 50K queries. State-of-the-art (SOTA) countermeasures cannot defend our attack effectively.



## **19. Self-playing Adversarial Language Game Enhances LLM Reasoning**

cs.CL

Preprint

**SubmitDate**: 2024-05-23    [abs](http://arxiv.org/abs/2404.10642v2) [paper-pdf](http://arxiv.org/pdf/2404.10642v2)

**Authors**: Pengyu Cheng, Tianhao Hu, Han Xu, Zhisong Zhang, Yong Dai, Lei Han, Nan Du

**Abstract**: We explore the self-play training procedure of large language models (LLMs) in a two-player adversarial language game called Adversarial Taboo. In this game, an attacker and a defender communicate around a target word only visible to the attacker. The attacker aims to induce the defender to speak the target word unconsciously, while the defender tries to infer the target word from the attacker's utterances. To win the game, both players should have sufficient knowledge about the target word and high-level reasoning ability to infer and express in this information-reserved conversation. Hence, we are curious about whether LLMs' reasoning ability can be further enhanced by self-play in this adversarial language game (SPAG). With this goal, we select several open-source LLMs and let each act as the attacker and play with a copy of itself as the defender on an extensive range of target words. Through reinforcement learning on the game outcomes, we observe that the LLMs' performances uniformly improve on a broad range of reasoning benchmarks. Furthermore, iteratively adopting this self-play process can continuously promote LLMs' reasoning abilities. The code is at https://github.com/Linear95/SPAG.



## **20. Eidos: Efficient, Imperceptible Adversarial 3D Point Clouds**

cs.CV

Preprint

**SubmitDate**: 2024-05-23    [abs](http://arxiv.org/abs/2405.14210v1) [paper-pdf](http://arxiv.org/pdf/2405.14210v1)

**Authors**: Hanwei Zhang, Luo Cheng, Qisong He, Wei Huang, Renjue Li, Ronan Sicre, Xiaowei Huang, Holger Hermanns, Lijun Zhang

**Abstract**: Classification of 3D point clouds is a challenging machine learning (ML) task with important real-world applications in a spectrum from autonomous driving and robot-assisted surgery to earth observation from low orbit. As with other ML tasks, classification models are notoriously brittle in the presence of adversarial attacks. These are rooted in imperceptible changes to inputs with the effect that a seemingly well-trained model ends up misclassifying the input. This paper adds to the understanding of adversarial attacks by presenting Eidos, a framework providing Efficient Imperceptible aDversarial attacks on 3D pOint cloudS. Eidos supports a diverse set of imperceptibility metrics. It employs an iterative, two-step procedure to identify optimal adversarial examples, thereby enabling a runtime-imperceptibility trade-off. We provide empirical evidence relative to several popular 3D point cloud classification models and several established 3D attack methods, showing Eidos' superiority with respect to efficiency as well as imperceptibility.



## **21. S-Eval: Automatic and Adaptive Test Generation for Benchmarking Safety Evaluation of Large Language Models**

cs.CR

18 pages, 11 figures

**SubmitDate**: 2024-05-23    [abs](http://arxiv.org/abs/2405.14191v1) [paper-pdf](http://arxiv.org/pdf/2405.14191v1)

**Authors**: Xiaohan Yuan, Jinfeng Li, Dongxia Wang, Yuefeng Chen, Xiaofeng Mao, Longtao Huang, Hui Xue, Wenhai Wang, Kui Ren, Jingyi Wang

**Abstract**: Large Language Models have gained considerable attention for their revolutionary capabilities. However, there is also growing concern on their safety implications, making a comprehensive safety evaluation for LLMs urgently needed before model deployment. In this work, we propose S-Eval, a new comprehensive, multi-dimensional and open-ended safety evaluation benchmark. At the core of S-Eval is a novel LLM-based automatic test prompt generation and selection framework, which trains an expert testing LLM Mt combined with a range of test selection strategies to automatically construct a high-quality test suite for the safety evaluation. The key to the automation of this process is a novel expert safety-critique LLM Mc able to quantify the riskiness score of a LLM's response, and additionally produce risk tags and explanations. Besides, the generation process is also guided by a carefully designed risk taxonomy with four different levels, covering comprehensive and multi-dimensional safety risks of concern. Based on these, we systematically construct a new and large-scale safety evaluation benchmark for LLMs consisting of 220,000 evaluation prompts, including 20,000 base risk prompts (10,000 in Chinese and 10,000 in English) and 200, 000 corresponding attack prompts derived from 10 popular adversarial instruction attacks against LLMs. Moreover, considering the rapid evolution of LLMs and accompanied safety threats, S-Eval can be flexibly configured and adapted to include new risks, attacks and models. S-Eval is extensively evaluated on 20 popular and representative LLMs. The results confirm that S-Eval can better reflect and inform the safety risks of LLMs compared to existing benchmarks. We also explore the impacts of parameter scales, language environments, and decoding parameters on the evaluation, providing a systematic methodology for evaluating the safety of LLMs.



## **22. Certified Robustness against Sparse Adversarial Perturbations via Data Localization**

cs.LG

**SubmitDate**: 2024-05-23    [abs](http://arxiv.org/abs/2405.14176v1) [paper-pdf](http://arxiv.org/pdf/2405.14176v1)

**Authors**: Ambar Pal, René Vidal, Jeremias Sulam

**Abstract**: Recent work in adversarial robustness suggests that natural data distributions are localized, i.e., they place high probability in small volume regions of the input space, and that this property can be utilized for designing classifiers with improved robustness guarantees for $\ell_2$-bounded perturbations. Yet, it is still unclear if this observation holds true for more general metrics. In this work, we extend this theory to $\ell_0$-bounded adversarial perturbations, where the attacker can modify a few pixels of the image but is unrestricted in the magnitude of perturbation, and we show necessary and sufficient conditions for the existence of $\ell_0$-robust classifiers. Theoretical certification approaches in this regime essentially employ voting over a large ensemble of classifiers. Such procedures are combinatorial and expensive or require complicated certification techniques. In contrast, a simple classifier emerges from our theory, dubbed Box-NN, which naturally incorporates the geometry of the problem and improves upon the current state-of-the-art in certified robustness against sparse attacks for the MNIST and Fashion-MNIST datasets.



## **23. Towards Transferable Attacks Against Vision-LLMs in Autonomous Driving with Typography**

cs.CV

12 pages, 5 tables, 5 figures, work in progress

**SubmitDate**: 2024-05-23    [abs](http://arxiv.org/abs/2405.14169v1) [paper-pdf](http://arxiv.org/pdf/2405.14169v1)

**Authors**: Nhat Chung, Sensen Gao, Tuan-Anh Vu, Jie Zhang, Aishan Liu, Yun Lin, Jin Song Dong, Qing Guo

**Abstract**: Vision-Large-Language-Models (Vision-LLMs) are increasingly being integrated into autonomous driving (AD) systems due to their advanced visual-language reasoning capabilities, targeting the perception, prediction, planning, and control mechanisms. However, Vision-LLMs have demonstrated susceptibilities against various types of adversarial attacks, which would compromise their reliability and safety. To further explore the risk in AD systems and the transferability of practical threats, we propose to leverage typographic attacks against AD systems relying on the decision-making capabilities of Vision-LLMs. Different from the few existing works developing general datasets of typographic attacks, this paper focuses on realistic traffic scenarios where these attacks can be deployed, on their potential effects on the decision-making autonomy, and on the practical ways in which these attacks can be physically presented. To achieve the above goals, we first propose a dataset-agnostic framework for automatically generating false answers that can mislead Vision-LLMs' reasoning. Then, we present a linguistic augmentation scheme that facilitates attacks at image-level and region-level reasoning, and we extend it with attack patterns against multiple reasoning tasks simultaneously. Based on these, we conduct a study on how these attacks can be realized in physical traffic scenarios. Through our empirical study, we evaluate the effectiveness, transferability, and realizability of typographic attacks in traffic scenes. Our findings demonstrate particular harmfulness of the typographic attacks against existing Vision-LLMs (e.g., LLaVA, Qwen-VL, VILA, and Imp), thereby raising community awareness of vulnerabilities when incorporating such models into AD systems. We will release our source code upon acceptance.



## **24. Carefully Blending Adversarial Training and Purification Improves Adversarial Robustness**

cs.CV

21 pages, 1 figure, 15 tables

**SubmitDate**: 2024-05-23    [abs](http://arxiv.org/abs/2306.06081v4) [paper-pdf](http://arxiv.org/pdf/2306.06081v4)

**Authors**: Emanuele Ballarin, Alessio Ansuini, Luca Bortolussi

**Abstract**: In this work, we propose a novel adversarial defence mechanism for image classification - CARSO - blending the paradigms of adversarial training and adversarial purification in a synergistic robustness-enhancing way. The method builds upon an adversarially-trained classifier, and learns to map its internal representation associated with a potentially perturbed input onto a distribution of tentative clean reconstructions. Multiple samples from such distribution are classified by the same adversarially-trained model, and an aggregation of its outputs finally constitutes the robust prediction of interest. Experimental evaluation by a well-established benchmark of strong adaptive attacks, across different image datasets, shows that CARSO is able to defend itself against adaptive end-to-end white-box attacks devised for stochastic defences. Paying a modest clean accuracy toll, our method improves by a significant margin the state-of-the-art for CIFAR-10, CIFAR-100, and TinyImageNet-200 $\ell_\infty$ robust classification accuracy against AutoAttack. Code, and instructions to obtain pre-trained models are available at https://github.com/emaballarin/CARSO .



## **25. Breaking Free: How to Hack Safety Guardrails in Black-Box Diffusion Models!**

cs.CV

**SubmitDate**: 2024-05-23    [abs](http://arxiv.org/abs/2402.04699v2) [paper-pdf](http://arxiv.org/pdf/2402.04699v2)

**Authors**: Shashank Kotyan, Po-Yuan Mao, Pin-Yu Chen, Danilo Vasconcellos Vargas

**Abstract**: Deep neural networks can be exploited using natural adversarial samples, which do not impact human perception. Current approaches often rely on deep neural networks' white-box nature to generate these adversarial samples or synthetically alter the distribution of adversarial samples compared to the training distribution. In contrast, we propose EvoSeed, a novel evolutionary strategy-based algorithmic framework for generating photo-realistic natural adversarial samples. Our EvoSeed framework uses auxiliary Conditional Diffusion and Classifier models to operate in a black-box setting. We employ CMA-ES to optimize the search for an initial seed vector, which, when processed by the Conditional Diffusion Model, results in the natural adversarial sample misclassified by the Classifier Model. Experiments show that generated adversarial images are of high image quality, raising concerns about generating harmful content bypassing safety classifiers. Our research opens new avenues to understanding the limitations of current safety mechanisms and the risk of plausible attacks against classifier systems using image generation. Project Website can be accessed at: https://shashankkotyan.github.io/EvoSeed.



## **26. Nearly Tight Black-Box Auditing of Differentially Private Machine Learning**

cs.CR

**SubmitDate**: 2024-05-23    [abs](http://arxiv.org/abs/2405.14106v1) [paper-pdf](http://arxiv.org/pdf/2405.14106v1)

**Authors**: Meenatchi Sundaram Muthu Selva Annamalai, Emiliano De Cristofaro

**Abstract**: This paper presents a nearly tight audit of the Differentially Private Stochastic Gradient Descent (DP-SGD) algorithm in the black-box model. Our auditing procedure empirically estimates the privacy leakage from DP-SGD using membership inference attacks; unlike prior work, the estimates are appreciably close to the theoretical DP bounds. The main intuition is to craft worst-case initial model parameters, as DP-SGD's privacy analysis is agnostic to the choice of the initial model parameters. For models trained with theoretical $\varepsilon=10.0$ on MNIST and CIFAR-10, our auditing procedure yields empirical estimates of $7.21$ and $6.95$, respectively, on 1,000-record samples and $6.48$ and $4.96$ on the full datasets. By contrast, previous work achieved tight audits only in stronger (i.e., less realistic) white-box models that allow the adversary to access the model's inner parameters and insert arbitrary gradients. Our auditing procedure can be used to detect bugs and DP violations more easily and offers valuable insight into how the privacy analysis of DP-SGD can be further improved.



## **27. Adversarial Item Promotion on Visually-Aware Recommender Systems by Guided Diffusion**

cs.IR

Accepted by TOIS 2024

**SubmitDate**: 2024-05-23    [abs](http://arxiv.org/abs/2312.15826v4) [paper-pdf](http://arxiv.org/pdf/2312.15826v4)

**Authors**: Lijian Chen, Wei Yuan, Tong Chen, Guanhua Ye, Quoc Viet Hung Nguyen, Hongzhi Yin

**Abstract**: Visually-aware recommender systems have found widespread application in domains where visual elements significantly contribute to the inference of users' potential preferences. While the incorporation of visual information holds the promise of enhancing recommendation accuracy and alleviating the cold-start problem, it is essential to point out that the inclusion of item images may introduce substantial security challenges. Some existing works have shown that the item provider can manipulate item exposure rates to its advantage by constructing adversarial images. However, these works cannot reveal the real vulnerability of visually-aware recommender systems because (1) The generated adversarial images are markedly distorted, rendering them easily detectable by human observers; (2) The effectiveness of the attacks is inconsistent and even ineffective in some scenarios. To shed light on the real vulnerabilities of visually-aware recommender systems when confronted with adversarial images, this paper introduces a novel attack method, IPDGI (Item Promotion by Diffusion Generated Image). Specifically, IPDGI employs a guided diffusion model to generate adversarial samples designed to deceive visually-aware recommender systems. Taking advantage of accurately modeling benign images' distribution by diffusion models, the generated adversarial images have high fidelity with original images, ensuring the stealth of our IPDGI. To demonstrate the effectiveness of our proposed methods, we conduct extensive experiments on two commonly used e-commerce recommendation datasets (Amazon Beauty and Amazon Baby) with several typical visually-aware recommender systems. The experimental results show that our attack method has a significant improvement in both the performance of promoting the long-tailed (i.e., unpopular) items and the quality of generated adversarial images.



## **28. Learning to Transform Dynamically for Better Adversarial Transferability**

cs.CV

accepted as a poster in CVPR 2024

**SubmitDate**: 2024-05-23    [abs](http://arxiv.org/abs/2405.14077v1) [paper-pdf](http://arxiv.org/pdf/2405.14077v1)

**Authors**: Rongyi Zhu, Zeliang Zhang, Susan Liang, Zhuo Liu, Chenliang Xu

**Abstract**: Adversarial examples, crafted by adding perturbations imperceptible to humans, can deceive neural networks. Recent studies identify the adversarial transferability across various models, \textit{i.e.}, the cross-model attack ability of adversarial samples. To enhance such adversarial transferability, existing input transformation-based methods diversify input data with transformation augmentation. However, their effectiveness is limited by the finite number of available transformations. In our study, we introduce a novel approach named Learning to Transform (L2T). L2T increases the diversity of transformed images by selecting the optimal combination of operations from a pool of candidates, consequently improving adversarial transferability. We conceptualize the selection of optimal transformation combinations as a trajectory optimization problem and employ a reinforcement learning strategy to effectively solve the problem. Comprehensive experiments on the ImageNet dataset, as well as practical tests with Google Vision and GPT-4V, reveal that L2T surpasses current methodologies in enhancing adversarial transferability, thereby confirming its effectiveness and practical significance. The code is available at https://github.com/RongyiZhu/L2T.



## **29. Remote Keylogging Attacks in Multi-user VR Applications**

cs.CR

Accepted for Usenix 2024

**SubmitDate**: 2024-05-22    [abs](http://arxiv.org/abs/2405.14036v1) [paper-pdf](http://arxiv.org/pdf/2405.14036v1)

**Authors**: Zihao Su, Kunlin Cai, Reuben Beeler, Lukas Dresel, Allan Garcia, Ilya Grishchenko, Yuan Tian, Christopher Kruegel, Giovanni Vigna

**Abstract**: As Virtual Reality (VR) applications grow in popularity, they have bridged distances and brought users closer together. However, with this growth, there have been increasing concerns about security and privacy, especially related to the motion data used to create immersive experiences. In this study, we highlight a significant security threat in multi-user VR applications, which are applications that allow multiple users to interact with each other in the same virtual space. Specifically, we propose a remote attack that utilizes the avatar rendering information collected from an adversary's game clients to extract user-typed secrets like credit card information, passwords, or private conversations. We do this by (1) extracting motion data from network packets, and (2) mapping motion data to keystroke entries. We conducted a user study to verify the attack's effectiveness, in which our attack successfully inferred 97.62% of the keystrokes. Besides, we performed an additional experiment to underline that our attack is practical, confirming its effectiveness even when (1) there are multiple users in a room, and (2) the attacker cannot see the victims. Moreover, we replicated our proposed attack on four applications to demonstrate the generalizability of the attack. These results underscore the severity of the vulnerability and its potential impact on millions of VR social platform users.



## **30. Adversarial Training of Two-Layer Polynomial and ReLU Activation Networks via Convex Optimization**

cs.LG

6 pages, 4 figures

**SubmitDate**: 2024-05-22    [abs](http://arxiv.org/abs/2405.14033v1) [paper-pdf](http://arxiv.org/pdf/2405.14033v1)

**Authors**: Daniel Kuelbs, Sanjay Lall, Mert Pilanci

**Abstract**: Training neural networks which are robust to adversarial attacks remains an important problem in deep learning, especially as heavily overparameterized models are adopted in safety-critical settings. Drawing from recent work which reformulates the training problems for two-layer ReLU and polynomial activation networks as convex programs, we devise a convex semidefinite program (SDP) for adversarial training of polynomial activation networks via the S-procedure. We also derive a convex SDP to compute the minimum distance from a correctly classified example to the decision boundary of a polynomial activation network. Adversarial training for two-layer ReLU activation networks has been explored in the literature, but, in contrast to prior work, we present a scalable approach which is compatible with standard machine libraries and GPU acceleration. The adversarial training SDP for polynomial activation networks leads to large increases in robust test accuracy against $\ell^\infty$ attacks on the Breast Cancer Wisconsin dataset from the UCI Machine Learning Repository. For two-layer ReLU networks, we leverage our scalable implementation to retrain the final two fully connected layers of a Pre-Activation ResNet-18 model on the CIFAR-10 dataset. Our 'robustified' model achieves higher clean and robust test accuracies than the same architecture trained with sharpness-aware minimization.



## **31. WordGame: Efficient & Effective LLM Jailbreak via Simultaneous Obfuscation in Query and Response**

cs.LG

**SubmitDate**: 2024-05-22    [abs](http://arxiv.org/abs/2405.14023v1) [paper-pdf](http://arxiv.org/pdf/2405.14023v1)

**Authors**: Tianrong Zhang, Bochuan Cao, Yuanpu Cao, Lu Lin, Prasenjit Mitra, Jinghui Chen

**Abstract**: The recent breakthrough in large language models (LLMs) such as ChatGPT has revolutionized production processes at an unprecedented pace. Alongside this progress also comes mounting concerns about LLMs' susceptibility to jailbreaking attacks, which leads to the generation of harmful or unsafe content. While safety alignment measures have been implemented in LLMs to mitigate existing jailbreak attempts and force them to become increasingly complicated, it is still far from perfect. In this paper, we analyze the common pattern of the current safety alignment and show that it is possible to exploit such patterns for jailbreaking attacks by simultaneous obfuscation in queries and responses. Specifically, we propose WordGame attack, which replaces malicious words with word games to break down the adversarial intent of a query and encourage benign content regarding the games to precede the anticipated harmful content in the response, creating a context that is hardly covered by any corpus used for safety alignment. Extensive experiments demonstrate that WordGame attack can break the guardrails of the current leading proprietary and open-source LLMs, including the latest Claude-3, GPT-4, and Llama-3 models. Further ablation studies on such simultaneous obfuscation in query and response provide evidence of the merits of the attack strategy beyond an individual attack.



## **32. LookHere: Vision Transformers with Directed Attention Generalize and Extrapolate**

cs.CV

**SubmitDate**: 2024-05-22    [abs](http://arxiv.org/abs/2405.13985v1) [paper-pdf](http://arxiv.org/pdf/2405.13985v1)

**Authors**: Anthony Fuller, Daniel G. Kyrollos, Yousef Yassin, James R. Green

**Abstract**: High-resolution images offer more information about scenes that can improve model accuracy. However, the dominant model architecture in computer vision, the vision transformer (ViT), cannot effectively leverage larger images without finetuning -- ViTs poorly extrapolate to more patches at test time, although transformers offer sequence length flexibility. We attribute this shortcoming to the current patch position encoding methods, which create a distribution shift when extrapolating.   We propose a drop-in replacement for the position encoding of plain ViTs that restricts attention heads to fixed fields of view, pointed in different directions, using 2D attention masks. Our novel method, called LookHere, provides translation-equivariance, ensures attention head diversity, and limits the distribution shift that attention heads face when extrapolating. We demonstrate that LookHere improves performance on classification (avg. 1.6%), against adversarial attack (avg. 5.4%), and decreases calibration error (avg. 1.5%) -- on ImageNet without extrapolation. With extrapolation, LookHere outperforms the current SoTA position encoding method, 2D-RoPE, by 21.7% on ImageNet when trained at $224^2$ px and tested at $1024^2$ px. Additionally, we release a high-resolution test set to improve the evaluation of high-resolution image classifiers, called ImageNet-HR.



## **33. Towards Certification of Uncertainty Calibration under Adversarial Attacks**

cs.LG

11 pages main paper, appendix included

**SubmitDate**: 2024-05-22    [abs](http://arxiv.org/abs/2405.13922v1) [paper-pdf](http://arxiv.org/pdf/2405.13922v1)

**Authors**: Cornelius Emde, Francesco Pinto, Thomas Lukasiewicz, Philip H. S. Torr, Adel Bibi

**Abstract**: Since neural classifiers are known to be sensitive to adversarial perturbations that alter their accuracy, \textit{certification methods} have been developed to provide provable guarantees on the insensitivity of their predictions to such perturbations. Furthermore, in safety-critical applications, the frequentist interpretation of the confidence of a classifier (also known as model calibration) can be of utmost importance. This property can be measured via the Brier score or the expected calibration error. We show that attacks can significantly harm calibration, and thus propose certified calibration as worst-case bounds on calibration under adversarial perturbations. Specifically, we produce analytic bounds for the Brier score and approximate bounds via the solution of a mixed-integer program on the expected calibration error. Finally, we propose novel calibration attacks and demonstrate how they can improve model calibration through \textit{adversarial calibration training}.



## **34. DistriBlock: Identifying adversarial audio samples by leveraging characteristics of the output distribution**

cs.SD

**SubmitDate**: 2024-05-22    [abs](http://arxiv.org/abs/2305.17000v4) [paper-pdf](http://arxiv.org/pdf/2305.17000v4)

**Authors**: Matías P. Pizarro B., Dorothea Kolossa, Asja Fischer

**Abstract**: Adversarial attacks can mislead automatic speech recognition (ASR) systems into predicting an arbitrary target text, thus posing a clear security threat. To prevent such attacks, we propose DistriBlock, an efficient detection strategy applicable to any ASR system that predicts a probability distribution over output tokens in each time step. We measure a set of characteristics of this distribution: the median, maximum, and minimum over the output probabilities, the entropy of the distribution, as well as the Kullback-Leibler and the Jensen-Shannon divergence with respect to the distributions of the subsequent time step. Then, by leveraging the characteristics observed for both benign and adversarial data, we apply binary classifiers, including simple threshold-based classification, ensembles of such classifiers, and neural networks. Through extensive analysis across different state-of-the-art ASR systems and language data sets, we demonstrate the supreme performance of this approach, with a mean area under the receiver operating characteristic curve for distinguishing target adversarial examples against clean and noisy data of 99% and 97%, respectively. To assess the robustness of our method, we show that adaptive adversarial examples that can circumvent DistriBlock are much noisier, which makes them easier to detect through filtering and creates another avenue for preserving the system's robustness.



## **35. L-AutoDA: Leveraging Large Language Models for Automated Decision-based Adversarial Attacks**

cs.CR

Camera ready version for GECCO'24 workshop

**SubmitDate**: 2024-05-22    [abs](http://arxiv.org/abs/2401.15335v2) [paper-pdf](http://arxiv.org/pdf/2401.15335v2)

**Authors**: Ping Guo, Fei Liu, Xi Lin, Qingchuan Zhao, Qingfu Zhang

**Abstract**: In the rapidly evolving field of machine learning, adversarial attacks present a significant challenge to model robustness and security. Decision-based attacks, which only require feedback on the decision of a model rather than detailed probabilities or scores, are particularly insidious and difficult to defend against. This work introduces L-AutoDA (Large Language Model-based Automated Decision-based Adversarial Attacks), a novel approach leveraging the generative capabilities of Large Language Models (LLMs) to automate the design of these attacks. By iteratively interacting with LLMs in an evolutionary framework, L-AutoDA automatically designs competitive attack algorithms efficiently without much human effort. We demonstrate the efficacy of L-AutoDA on CIFAR-10 dataset, showing significant improvements over baseline methods in both success rate and computational efficiency. Our findings underscore the potential of language models as tools for adversarial attack generation and highlight new avenues for the development of robust AI systems.



## **36. ALI-DPFL: Differentially Private Federated Learning with Adaptive Local Iterations**

cs.LG

**SubmitDate**: 2024-05-22    [abs](http://arxiv.org/abs/2308.10457v9) [paper-pdf](http://arxiv.org/pdf/2308.10457v9)

**Authors**: Xinpeng Ling, Jie Fu, Kuncan Wang, Haitao Liu, Zhili Chen

**Abstract**: Federated Learning (FL) is a distributed machine learning technique that allows model training among multiple devices or organizations by sharing training parameters instead of raw data. However, adversaries can still infer individual information through inference attacks (e.g. differential attacks) on these training parameters. As a result, Differential Privacy (DP) has been widely used in FL to prevent such attacks.   We consider differentially private federated learning in a resource-constrained scenario, where both privacy budget and communication rounds are constrained. By theoretically analyzing the convergence, we can find the optimal number of local DPSGD iterations for clients between any two sequential global updates. Based on this, we design an algorithm of Differentially Private Federated Learning with Adaptive Local Iterations (ALI-DPFL). We experiment our algorithm on the MNIST, FashionMNIST and Cifar10 datasets, and demonstrate significantly better performances than previous work in the resource-constraint scenario. Code is available at https://github.com/cheng-t/ALI-DPFL.



## **37. Adversarial Training via Adaptive Knowledge Amalgamation of an Ensemble of Teachers**

cs.LG

**SubmitDate**: 2024-05-22    [abs](http://arxiv.org/abs/2405.13324v1) [paper-pdf](http://arxiv.org/pdf/2405.13324v1)

**Authors**: Shayan Mohajer Hamidi, Linfeng Ye

**Abstract**: Adversarial training (AT) is a popular method for training robust deep neural networks (DNNs) against adversarial attacks. Yet, AT suffers from two shortcomings: (i) the robustness of DNNs trained by AT is highly intertwined with the size of the DNNs, posing challenges in achieving robustness in smaller models; and (ii) the adversarial samples employed during the AT process exhibit poor generalization, leaving DNNs vulnerable to unforeseen attack types. To address these dual challenges, this paper introduces adversarial training via adaptive knowledge amalgamation of an ensemble of teachers (AT-AKA). In particular, we generate a diverse set of adversarial samples as the inputs to an ensemble of teachers; and then, we adaptively amalgamate the logtis of these teachers to train a generalized-robust student. Through comprehensive experiments, we illustrate the superior efficacy of AT-AKA over existing AT methods and adversarial robustness distillation techniques against cutting-edge attacks, including AutoAttack.



## **38. Box-Free Model Watermarks Are Prone to Black-Box Removal Attacks**

cs.CV

**SubmitDate**: 2024-05-22    [abs](http://arxiv.org/abs/2405.09863v2) [paper-pdf](http://arxiv.org/pdf/2405.09863v2)

**Authors**: Haonan An, Guang Hua, Zhiping Lin, Yuguang Fang

**Abstract**: Box-free model watermarking is an emerging technique to safeguard the intellectual property of deep learning models, particularly those for low-level image processing tasks. Existing works have verified and improved its effectiveness in several aspects. However, in this paper, we reveal that box-free model watermarking is prone to removal attacks, even under the real-world threat model such that the protected model and the watermark extractor are in black boxes. Under this setting, we carry out three studies. 1) We develop an extractor-gradient-guided (EGG) remover and show its effectiveness when the extractor uses ReLU activation only. 2) More generally, for an unknown extractor, we leverage adversarial attacks and design the EGG remover based on the estimated gradients. 3) Under the most stringent condition that the extractor is inaccessible, we design a transferable remover based on a set of private proxy models. In all cases, the proposed removers can successfully remove embedded watermarks while preserving the quality of the processed images, and we also demonstrate that the EGG remover can even replace the watermarks. Extensive experimental results verify the effectiveness and generalizability of the proposed attacks, revealing the vulnerabilities of the existing box-free methods and calling for further research.



## **39. Cloud-based XAI Services for Assessing Open Repository Models Under Adversarial Attacks**

cs.CR

Accepted by IEEE International Conference on Software Services  Engineering (SSE) 2024

**SubmitDate**: 2024-05-21    [abs](http://arxiv.org/abs/2401.12261v3) [paper-pdf](http://arxiv.org/pdf/2401.12261v3)

**Authors**: Zerui Wang, Yan Liu

**Abstract**: The opacity of AI models necessitates both validation and evaluation before their integration into services. To investigate these models, explainable AI (XAI) employs methods that elucidate the relationship between input features and output predictions. The operations of XAI extend beyond the execution of a single algorithm, involving a series of activities that include preprocessing data, adjusting XAI to align with model parameters, invoking the model to generate predictions, and summarizing the XAI results. Adversarial attacks are well-known threats that aim to mislead AI models. The assessment complexity, especially for XAI, increases when open-source AI models are subject to adversarial attacks, due to various combinations. To automate the numerous entities and tasks involved in XAI-based assessments, we propose a cloud-based service framework that encapsulates computing components as microservices and organizes assessment tasks into pipelines. The current XAI tools are not inherently service-oriented. This framework also integrates open XAI tool libraries as part of the pipeline composition. We demonstrate the application of XAI services for assessing five quality attributes of AI models: (1) computational cost, (2) performance, (3) robustness, (4) explanation deviation, and (5) explanation resilience across computer vision and tabular cases. The service framework generates aggregated analysis that showcases the quality attributes for more than a hundred combination scenarios.



## **40. Inexact Unlearning Needs More Careful Evaluations to Avoid a False Sense of Privacy**

cs.LG

**SubmitDate**: 2024-05-21    [abs](http://arxiv.org/abs/2403.01218v3) [paper-pdf](http://arxiv.org/pdf/2403.01218v3)

**Authors**: Jamie Hayes, Ilia Shumailov, Eleni Triantafillou, Amr Khalifa, Nicolas Papernot

**Abstract**: The high cost of model training makes it increasingly desirable to develop techniques for unlearning. These techniques seek to remove the influence of a training example without having to retrain the model from scratch. Intuitively, once a model has unlearned, an adversary that interacts with the model should no longer be able to tell whether the unlearned example was included in the model's training set or not. In the privacy literature, this is known as membership inference. In this work, we discuss adaptations of Membership Inference Attacks (MIAs) to the setting of unlearning (leading to their "U-MIA" counterparts). We propose a categorization of existing U-MIAs into "population U-MIAs", where the same attacker is instantiated for all examples, and "per-example U-MIAs", where a dedicated attacker is instantiated for each example. We show that the latter category, wherein the attacker tailors its membership prediction to each example under attack, is significantly stronger. Indeed, our results show that the commonly used U-MIAs in the unlearning literature overestimate the privacy protection afforded by existing unlearning techniques on both vision and language models. Our investigation reveals a large variance in the vulnerability of different examples to per-example U-MIAs. In fact, several unlearning algorithms lead to a reduced vulnerability for some, but not all, examples that we wish to unlearn, at the expense of increasing it for other examples. Notably, we find that the privacy protection for the remaining training examples may worsen as a consequence of unlearning. We also discuss the fundamental difficulty of equally protecting all examples using existing unlearning schemes, due to the different rates at which examples are unlearned. We demonstrate that naive attempts at tailoring unlearning stopping criteria to different examples fail to alleviate these issues.



## **41. Adversarial Attacks and Defenses in Automated Control Systems: A Comprehensive Benchmark**

cs.LG

**SubmitDate**: 2024-05-21    [abs](http://arxiv.org/abs/2403.13502v3) [paper-pdf](http://arxiv.org/pdf/2403.13502v3)

**Authors**: Vitaliy Pozdnyakov, Aleksandr Kovalenko, Ilya Makarov, Mikhail Drobyshevskiy, Kirill Lukyanov

**Abstract**: Integrating machine learning into Automated Control Systems (ACS) enhances decision-making in industrial process management. One of the limitations to the widespread adoption of these technologies in industry is the vulnerability of neural networks to adversarial attacks. This study explores the threats in deploying deep learning models for fault diagnosis in ACS using the Tennessee Eastman Process dataset. By evaluating three neural networks with different architectures, we subject them to six types of adversarial attacks and explore five different defense methods. Our results highlight the strong vulnerability of models to adversarial samples and the varying effectiveness of defense strategies. We also propose a novel protection approach by combining multiple defense methods and demonstrate it's efficacy. This research contributes several insights into securing machine learning within ACS, ensuring robust fault diagnosis in industrial processes.



## **42. Rethinking the Vulnerabilities of Face Recognition Systems:From a Practical Perspective**

cs.CR

19 pages

**SubmitDate**: 2024-05-21    [abs](http://arxiv.org/abs/2405.12786v1) [paper-pdf](http://arxiv.org/pdf/2405.12786v1)

**Authors**: Jiahao Chen, Zhiqiang Shen, Yuwen Pu, Chunyi Zhou, Shouling Ji

**Abstract**: Face Recognition Systems (FRS) have increasingly integrated into critical applications, including surveillance and user authentication, highlighting their pivotal role in modern security systems. Recent studies have revealed vulnerabilities in FRS to adversarial (e.g., adversarial patch attacks) and backdoor attacks (e.g., training data poisoning), raising significant concerns about their reliability and trustworthiness. Previous studies primarily focus on traditional adversarial or backdoor attacks, overlooking the resource-intensive or privileged-manipulation nature of such threats, thus limiting their practical generalization, stealthiness, universality and robustness. Correspondingly, in this paper, we delve into the inherent vulnerabilities in FRS through user studies and preliminary explorations. By exploiting these vulnerabilities, we identify a novel attack, facial identity backdoor attack dubbed FIBA, which unveils a potentially more devastating threat against FRS:an enrollment-stage backdoor attack. FIBA circumvents the limitations of traditional attacks, enabling broad-scale disruption by allowing any attacker donning a specific trigger to bypass these systems. This implies that after a single, poisoned example is inserted into the database, the corresponding trigger becomes a universal key for any attackers to spoof the FRS. This strategy essentially challenges the conventional attacks by initiating at the enrollment stage, dramatically transforming the threat landscape by poisoning the feature database rather than the training data.



## **43. Generative AI and Large Language Models for Cyber Security: All Insights You Need**

cs.CR

50 pages, 8 figures

**SubmitDate**: 2024-05-21    [abs](http://arxiv.org/abs/2405.12750v1) [paper-pdf](http://arxiv.org/pdf/2405.12750v1)

**Authors**: Mohamed Amine Ferrag, Fatima Alwahedi, Ammar Battah, Bilel Cherif, Abdechakour Mechri, Norbert Tihanyi

**Abstract**: This paper provides a comprehensive review of the future of cybersecurity through Generative AI and Large Language Models (LLMs). We explore LLM applications across various domains, including hardware design security, intrusion detection, software engineering, design verification, cyber threat intelligence, malware detection, and phishing detection. We present an overview of LLM evolution and its current state, focusing on advancements in models such as GPT-4, GPT-3.5, Mixtral-8x7B, BERT, Falcon2, and LLaMA. Our analysis extends to LLM vulnerabilities, such as prompt injection, insecure output handling, data poisoning, DDoS attacks, and adversarial instructions. We delve into mitigation strategies to protect these models, providing a comprehensive look at potential attack scenarios and prevention techniques. Furthermore, we evaluate the performance of 42 LLM models in cybersecurity knowledge and hardware security, highlighting their strengths and weaknesses. We thoroughly evaluate cybersecurity datasets for LLM training and testing, covering the lifecycle from data creation to usage and identifying gaps for future research. In addition, we review new strategies for leveraging LLMs, including techniques like Half-Quadratic Quantization (HQQ), Reinforcement Learning with Human Feedback (RLHF), Direct Preference Optimization (DPO), Quantized Low-Rank Adapters (QLoRA), and Retrieval-Augmented Generation (RAG). These insights aim to enhance real-time cybersecurity defenses and improve the sophistication of LLM applications in threat detection and response. Our paper provides a foundational understanding and strategic direction for integrating LLMs into future cybersecurity frameworks, emphasizing innovation and robust model deployment to safeguard against evolving cyber threats.



## **44. A GAN-Based Data Poisoning Attack Against Federated Learning Systems and Its Countermeasure**

cs.CR

18 pages, 16 figures

**SubmitDate**: 2024-05-21    [abs](http://arxiv.org/abs/2405.11440v2) [paper-pdf](http://arxiv.org/pdf/2405.11440v2)

**Authors**: Wei Sun, Bo Gao, Ke Xiong, Yuwei Wang

**Abstract**: As a distributed machine learning paradigm, federated learning (FL) is collaboratively carried out on privately owned datasets but without direct data access. Although the original intention is to allay data privacy concerns, "available but not visible" data in FL potentially brings new security threats, particularly poisoning attacks that target such "not visible" local data. Initial attempts have been made to conduct data poisoning attacks against FL systems, but cannot be fully successful due to their high chance of causing statistical anomalies. To unleash the potential for truly "invisible" attacks and build a more deterrent threat model, in this paper, a new data poisoning attack model named VagueGAN is proposed, which can generate seemingly legitimate but noisy poisoned data by untraditionally taking advantage of generative adversarial network (GAN) variants. Capable of manipulating the quality of poisoned data on demand, VagueGAN enables to trade-off attack effectiveness and stealthiness. Furthermore, a cost-effective countermeasure named Model Consistency-Based Defense (MCD) is proposed to identify GAN-poisoned data or models after finding out the consistency of GAN outputs. Extensive experiments on multiple datasets indicate that our attack method is generally much more stealthy as well as more effective in degrading FL performance with low complexity. Our defense method is also shown to be more competent in identifying GAN-poisoned data or models. The source codes are publicly available at \href{https://github.com/SSssWEIssSS/VagueGAN-Data-Poisoning-Attack-and-Its-Countermeasure}{https://github.com/SSssWEIssSS/VagueGAN-Data-Poisoning-Attack-and-Its-Countermeasure}.



## **45. How to Train a Backdoor-Robust Model on a Poisoned Dataset without Auxiliary Data?**

cs.CR

13 pages, under review

**SubmitDate**: 2024-05-21    [abs](http://arxiv.org/abs/2405.12719v1) [paper-pdf](http://arxiv.org/pdf/2405.12719v1)

**Authors**: Yuwen Pu, Jiahao Chen, Chunyi Zhou, Zhou Feng, Qingming Li, Chunqiang Hu, Shouling Ji

**Abstract**: Backdoor attacks have attracted wide attention from academia and industry due to their great security threat to deep neural networks (DNN). Most of the existing methods propose to conduct backdoor attacks by poisoning the training dataset with different strategies, so it's critical to identify the poisoned samples and then train a clean model on the unreliable dataset in the context of defending backdoor attacks. Although numerous backdoor countermeasure researches are proposed, their inherent weaknesses render them limited in practical scenarios, such as the requirement of enough clean samples, unstable defense performance under various attack conditions, poor defense performance against adaptive attacks, and so on.Therefore, in this paper, we are committed to overcome the above limitations and propose a more practical backdoor defense method. Concretely, we first explore the inherent relationship between the potential perturbations and the backdoor trigger, and the theoretical analysis and experimental results demonstrate that the poisoned samples perform more robustness to perturbation than the clean ones. Then, based on our key explorations, we introduce AdvrBD, an Adversarial perturbation-based and robust Backdoor Defense framework, which can effectively identify the poisoned samples and train a clean model on the poisoned dataset. Constructively, our AdvrBD eliminates the requirement for any clean samples or knowledge about the poisoned dataset (e.g., poisoning ratio), which significantly improves the practicality in real-world scenarios.



## **46. Robust Classification via a Single Diffusion Model**

cs.CV

Accepted by ICML 2024

**SubmitDate**: 2024-05-21    [abs](http://arxiv.org/abs/2305.15241v2) [paper-pdf](http://arxiv.org/pdf/2305.15241v2)

**Authors**: Huanran Chen, Yinpeng Dong, Zhengyi Wang, Xiao Yang, Chengqi Duan, Hang Su, Jun Zhu

**Abstract**: Diffusion models have been applied to improve adversarial robustness of image classifiers by purifying the adversarial noises or generating realistic data for adversarial training. However, diffusion-based purification can be evaded by stronger adaptive attacks while adversarial training does not perform well under unseen threats, exhibiting inevitable limitations of these methods. To better harness the expressive power of diffusion models, this paper proposes Robust Diffusion Classifier (RDC), a generative classifier that is constructed from a pre-trained diffusion model to be adversarially robust. RDC first maximizes the data likelihood of a given input and then predicts the class probabilities of the optimized input using the conditional likelihood estimated by the diffusion model through Bayes' theorem. To further reduce the computational cost, we propose a new diffusion backbone called multi-head diffusion and develop efficient sampling strategies. As RDC does not require training on particular adversarial attacks, we demonstrate that it is more generalizable to defend against multiple unseen threats. In particular, RDC achieves $75.67\%$ robust accuracy against various $\ell_\infty$ norm-bounded adaptive attacks with $\epsilon_\infty=8/255$ on CIFAR-10, surpassing the previous state-of-the-art adversarial training models by $+4.77\%$. The results highlight the potential of generative classifiers by employing pre-trained diffusion models for adversarial robustness compared with the commonly studied discriminative classifiers. Code is available at \url{https://github.com/huanranchen/DiffusionClassifier}.



## **47. EmInspector: Combating Backdoor Attacks in Federated Self-Supervised Learning Through Embedding Inspection**

cs.CR

18 pages, 12 figures

**SubmitDate**: 2024-05-21    [abs](http://arxiv.org/abs/2405.13080v1) [paper-pdf](http://arxiv.org/pdf/2405.13080v1)

**Authors**: Yuwen Qian, Shuchi Wu, Kang Wei, Ming Ding, Di Xiao, Tao Xiang, Chuan Ma, Song Guo

**Abstract**: Federated self-supervised learning (FSSL) has recently emerged as a promising paradigm that enables the exploitation of clients' vast amounts of unlabeled data while preserving data privacy. While FSSL offers advantages, its susceptibility to backdoor attacks, a concern identified in traditional federated supervised learning (FSL), has not been investigated. To fill the research gap, we undertake a comprehensive investigation into a backdoor attack paradigm, where unscrupulous clients conspire to manipulate the global model, revealing the vulnerability of FSSL to such attacks. In FSL, backdoor attacks typically build a direct association between the backdoor trigger and the target label. In contrast, in FSSL, backdoor attacks aim to alter the global model's representation for images containing the attacker's specified trigger pattern in favor of the attacker's intended target class, which is less straightforward. In this sense, we demonstrate that existing defenses are insufficient to mitigate the investigated backdoor attacks in FSSL, thus finding an effective defense mechanism is urgent. To tackle this issue, we dive into the fundamental mechanism of backdoor attacks on FSSL, proposing the Embedding Inspector (EmInspector) that detects malicious clients by inspecting the embedding space of local models. In particular, EmInspector assesses the similarity of embeddings from different local models using a small set of inspection images (e.g., ten images of CIFAR100) without specific requirements on sample distribution or labels. We discover that embeddings from backdoored models tend to cluster together in the embedding space for a given inspection image. Evaluation results show that EmInspector can effectively mitigate backdoor attacks on FSSL across various adversary settings. Our code is avaliable at https://github.com/ShuchiWu/EmInspector.



## **48. Fully Randomized Pointers**

cs.CR

24 pages, 3 figures

**SubmitDate**: 2024-05-21    [abs](http://arxiv.org/abs/2405.12513v1) [paper-pdf](http://arxiv.org/pdf/2405.12513v1)

**Authors**: Gregory J. Duck, Sai Dhawal Phaye, Roland H. C. Yap, Trevor E. Carlson

**Abstract**: Software security continues to be a critical concern for programs implemented in low-level programming languages such as C and C++. Many defenses have been proposed in the current literature, each with different trade-offs including performance, compatibility, and attack resistance. One general class of defense is pointer randomization or authentication, where invalid object access (e.g., memory errors) is obfuscated or denied. Many defenses rely on the program termination (e.g., crashing) to abort attacks, with the implicit assumption that an adversary cannot "brute force" the defense with multiple attack attempts. However, such assumptions do not always hold, such as hardware speculative execution attacks or network servers configured to restart on error. In such cases, we argue that most existing defenses provide only weak effective security.   In this paper, we propose Fully Randomized Pointers (FRP) as a stronger memory error defense that is resistant to even brute force attacks. The key idea is to fully randomize pointer bits -- as much as possible while also preserving binary compatibility -- rendering the relationships between pointers highly unpredictable. Furthermore, the very high degree of randomization renders brute force attacks impractical -- providing strong effective security compared to existing work. We design a new FRP encoding that is: (1) compatible with existing binary code (without recompilation); (2) decoupled from the underlying object layout; and (3) can be efficiently decoded on-the-fly to the underlying memory address. We prototype FRP in the form of a software implementation (BlueFat) to test security and compatibility, and a proof-of-concept hardware implementation (GreenFat) to evaluate performance. We show that FRP is secure, practical, and compatible at the binary level, while a hardware implementation can achieve low performance overheads (<10%).



## **49. GPT-4 Jailbreaks Itself with Near-Perfect Success Using Self-Explanation**

cs.CR

**SubmitDate**: 2024-05-21    [abs](http://arxiv.org/abs/2405.13077v1) [paper-pdf](http://arxiv.org/pdf/2405.13077v1)

**Authors**: Govind Ramesh, Yao Dou, Wei Xu

**Abstract**: Research on jailbreaking has been valuable for testing and understanding the safety and security issues of large language models (LLMs). In this paper, we introduce Iterative Refinement Induced Self-Jailbreak (IRIS), a novel approach that leverages the reflective capabilities of LLMs for jailbreaking with only black-box access. Unlike previous methods, IRIS simplifies the jailbreaking process by using a single model as both the attacker and target. This method first iteratively refines adversarial prompts through self-explanation, which is crucial for ensuring that even well-aligned LLMs obey adversarial instructions. IRIS then rates and enhances the output given the refined prompt to increase its harmfulness. We find IRIS achieves jailbreak success rates of 98% on GPT-4 and 92% on GPT-4 Turbo in under 7 queries. It significantly outperforms prior approaches in automatic, black-box and interpretable jailbreaking, while requiring substantially fewer queries, thereby establishing a new standard for interpretable jailbreaking methods.



## **50. Rethinking Robustness Assessment: Adversarial Attacks on Learning-based Quadrupedal Locomotion Controllers**

cs.RO

RSS 2024

**SubmitDate**: 2024-05-21    [abs](http://arxiv.org/abs/2405.12424v1) [paper-pdf](http://arxiv.org/pdf/2405.12424v1)

**Authors**: Fan Shi, Chong Zhang, Takahiro Miki, Joonho Lee, Marco Hutter, Stelian Coros

**Abstract**: Legged locomotion has recently achieved remarkable success with the progress of machine learning techniques, especially deep reinforcement learning (RL). Controllers employing neural networks have demonstrated empirical and qualitative robustness against real-world uncertainties, including sensor noise and external perturbations. However, formally investigating the vulnerabilities of these locomotion controllers remains a challenge. This difficulty arises from the requirement to pinpoint vulnerabilities across a long-tailed distribution within a high-dimensional, temporally sequential space. As a first step towards quantitative verification, we propose a computational method that leverages sequential adversarial attacks to identify weaknesses in learned locomotion controllers. Our research demonstrates that, even state-of-the-art robust controllers can fail significantly under well-designed, low-magnitude adversarial sequence. Through experiments in simulation and on the real robot, we validate our approach's effectiveness, and we illustrate how the results it generates can be used to robustify the original policy and offer valuable insights into the safety of these black-box policies.



