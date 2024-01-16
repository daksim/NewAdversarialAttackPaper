# Latest Adversarial Attack Papers
**update at 2024-01-16 10:21:22**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. How Johnny Can Persuade LLMs to Jailbreak Them: Rethinking Persuasion to Challenge AI Safety by Humanizing LLMs**

cs.CL

14 pages of the main text, qualitative examples of jailbreaks may be  harmful in nature

**SubmitDate**: 2024-01-12    [abs](http://arxiv.org/abs/2401.06373v1) [paper-pdf](http://arxiv.org/pdf/2401.06373v1)

**Authors**: Yi Zeng, Hongpeng Lin, Jingwen Zhang, Diyi Yang, Ruoxi Jia, Weiyan Shi

**Abstract**: Most traditional AI safety research has approached AI models as machines and centered on algorithm-focused attacks developed by security experts. As large language models (LLMs) become increasingly common and competent, non-expert users can also impose risks during daily interactions. This paper introduces a new perspective to jailbreak LLMs as human-like communicators, to explore this overlooked intersection between everyday language interaction and AI safety. Specifically, we study how to persuade LLMs to jailbreak them. First, we propose a persuasion taxonomy derived from decades of social science research. Then, we apply the taxonomy to automatically generate interpretable persuasive adversarial prompts (PAP) to jailbreak LLMs. Results show that persuasion significantly increases the jailbreak performance across all risk categories: PAP consistently achieves an attack success rate of over $92\%$ on Llama 2-7b Chat, GPT-3.5, and GPT-4 in $10$ trials, surpassing recent algorithm-focused attacks. On the defense side, we explore various mechanisms against PAP and, found a significant gap in existing defenses, and advocate for more fundamental mitigation for highly interactive LLMs



## **2. Adversarial Examples are Misaligned in Diffusion Model Manifolds**

cs.CV

under review

**SubmitDate**: 2024-01-12    [abs](http://arxiv.org/abs/2401.06637v1) [paper-pdf](http://arxiv.org/pdf/2401.06637v1)

**Authors**: Peter Lorenz, Ricard Durall, Jansi Keuper

**Abstract**: In recent years, diffusion models (DMs) have drawn significant attention for their success in approximating data distributions, yielding state-of-the-art generative results. Nevertheless, the versatility of these models extends beyond their generative capabilities to encompass various vision applications, such as image inpainting, segmentation, adversarial robustness, among others. This study is dedicated to the investigation of adversarial attacks through the lens of diffusion models. However, our objective does not involve enhancing the adversarial robustness of image classifiers. Instead, our focus lies in utilizing the diffusion model to detect and analyze the anomalies introduced by these attacks on images. To that end, we systematically examine the alignment of the distributions of adversarial examples when subjected to the process of transformation using diffusion models. The efficacy of this approach is assessed across CIFAR-10 and ImageNet datasets, including varying image sizes in the latter. The results demonstrate a notable capacity to discriminate effectively between benign and attacked images, providing compelling evidence that adversarial instances do not align with the learned manifold of the DMs.



## **3. FlashSyn: Flash Loan Attack Synthesis via Counter Example Driven Approximation**

cs.PL

29 pages, 8 figures, conference paper extended version

**SubmitDate**: 2024-01-12    [abs](http://arxiv.org/abs/2206.10708v3) [paper-pdf](http://arxiv.org/pdf/2206.10708v3)

**Authors**: Zhiyang Chen, Sidi Mohamed Beillahi, Fan Long

**Abstract**: In decentralized finance (DeFi), lenders can offer flash loans to borrowers, i.e., loans that are only valid within a blockchain transaction and must be repaid with fees by the end of that transaction. Unlike normal loans, flash loans allow borrowers to borrow large assets without upfront collaterals deposits. Malicious adversaries use flash loans to gather large assets to exploit vulnerable DeFi protocols. In this paper, we introduce a new framework for automated synthesis of adversarial transactions that exploit DeFi protocols using flash loans. To bypass the complexity of a DeFi protocol, we propose a new technique to approximate the DeFi protocol functional behaviors using numerical methods (polynomial linear regression and nearest-neighbor interpolation). We then construct an optimization query using the approximated functions of the DeFi protocol to find an adversarial attack constituted of a sequence of functions invocations with optimal parameters that gives the maximum profit. To improve the accuracy of the approximation, we propose a novel counterexample driven approximation refinement technique. We implement our framework in a tool named FlashSyn. We evaluate FlashSyn on 16 DeFi protocols that were victims to flash loan attacks and 2 DeFi protocols from Damn Vulnerable DeFi challenges. FlashSyn automatically synthesizes an adversarial attack for 16 of the 18 benchmarks. Among the 16 successful cases, FlashSyn identifies attack vectors yielding higher profits than those employed by historical hackers in 3 cases, and also discovers multiple distinct attack vectors in 10 cases, demonstrating its effectiveness in finding possible flash loan attacks.



## **4. When Fairness Meets Privacy: Exploring Privacy Threats in Fair Binary Classifiers through Membership Inference Attacks**

cs.LG

Under review

**SubmitDate**: 2024-01-12    [abs](http://arxiv.org/abs/2311.03865v2) [paper-pdf](http://arxiv.org/pdf/2311.03865v2)

**Authors**: Huan Tian, Guangsheng Zhang, Bo Liu, Tianqing Zhu, Ming Ding, Wanlei Zhou

**Abstract**: Previous studies have developed fairness methods for biased models that exhibit discriminatory behaviors towards specific subgroups. While these models have shown promise in achieving fair predictions, recent research has identified their potential vulnerability to score-based membership inference attacks (MIAs). In these attacks, adversaries can infer whether a particular data sample was used during training by analyzing the model's prediction scores. However, our investigations reveal that these score-based MIAs are ineffective when targeting fairness-enhanced models in binary classifications. The attack models trained to launch the MIAs degrade into simplistic threshold models, resulting in lower attack performance. Meanwhile, we observe that fairness methods often lead to prediction performance degradation for the majority subgroups of the training data. This raises the barrier to successful attacks and widens the prediction gaps between member and non-member data. Building upon these insights, we propose an efficient MIA method against fairness-enhanced models based on fairness discrepancy results (FD-MIA). It leverages the difference in the predictions from both the original and fairness-enhanced models and exploits the observed prediction gaps as attack clues. We also explore potential strategies for mitigating privacy leakages. Extensive experiments validate our findings and demonstrate the efficacy of the proposed method.



## **5. MVPatch: More Vivid Patch for Adversarial Camouflaged Attacks on Object Detectors in the Physical World**

cs.CR

14 pages, 8 figures. This work has been submitted to the IEEE for  possible publication. Copyright may be transferred without notice, after  which this version may no longer be accessible

**SubmitDate**: 2024-01-12    [abs](http://arxiv.org/abs/2312.17431v2) [paper-pdf](http://arxiv.org/pdf/2312.17431v2)

**Authors**: Zheng Zhou, Hongbo Zhao, Ju Liu, Qiaosheng Zhang, Liwei Geng, Shuchang Lyu, Wenquan Feng

**Abstract**: Recent investigations demonstrate that adversarial patches can be utilized to manipulate the result of object detection models. However, the conspicuous patterns on these patches may draw more attention and raise suspicions among humans. Moreover, existing works have primarily focused on enhancing the efficacy of attacks in the physical domain, rather than seeking to optimize their stealth attributes and transferability potential. To address these issues, we introduce a dual-perception-based attack framework that generates an adversarial patch known as the More Vivid Patch (MVPatch). The framework consists of a model-perception degradation method and a human-perception improvement method. To derive the MVPatch, we formulate an iterative process that simultaneously constrains the efficacy of multiple object detectors and refines the visual correlation between the generated adversarial patch and a realistic image. Our method employs a model-perception-based approach that reduces the object confidence scores of several object detectors to boost the transferability of adversarial patches. Further, within the human-perception-based framework, we put forward a lightweight technique for visual similarity measurement that facilitates the development of inconspicuous and natural adversarial patches and eliminates the reliance on additional generative models. Additionally, we introduce the naturalness score and transferability score as metrics for an unbiased assessment of various adversarial patches' natural appearance and transferability capacity. Extensive experiments demonstrate that the proposed MVPatch algorithm achieves superior attack transferability compared to similar algorithms in both digital and physical domains while also exhibiting a more natural appearance. These findings emphasize the remarkable stealthiness and transferability of the proposed MVPatch attack algorithm.



## **6. On the Query Complexity of Training Data Reconstruction in Private Learning**

cs.LG

Updated proof of Thm 10, fixed typos

**SubmitDate**: 2024-01-11    [abs](http://arxiv.org/abs/2303.16372v6) [paper-pdf](http://arxiv.org/pdf/2303.16372v6)

**Authors**: Prateeti Mukherjee, Satya Lokam

**Abstract**: We analyze the number of queries that a whitebox adversary needs to make to a private learner in order to reconstruct its training data. For $(\epsilon, \delta)$ DP learners with training data drawn from any arbitrary compact metric space, we provide the \emph{first known lower bounds on the adversary's query complexity} as a function of the learner's privacy parameters. \emph{Our results are minimax optimal for every $\epsilon \geq 0, \delta \in [0, 1]$, covering both $\epsilon$-DP and $(0, \delta)$ DP as corollaries}. Beyond this, we obtain query complexity lower bounds for $(\alpha, \epsilon)$ R\'enyi DP learners that are valid for any $\alpha > 1, \epsilon \geq 0$. Finally, we analyze data reconstruction attacks on locally compact metric spaces via the framework of Metric DP, a generalization of DP that accounts for the underlying metric structure of the data. In this setting, we provide the first known analysis of data reconstruction in unbounded, high dimensional spaces and obtain query complexity lower bounds that are nearly tight modulo logarithmic factors.



## **7. The Impact of Adversarial Node Placement in Decentralized Federated Learning Networks**

cs.CR

Submitted to ICC 2024 conference

**SubmitDate**: 2024-01-11    [abs](http://arxiv.org/abs/2311.07946v3) [paper-pdf](http://arxiv.org/pdf/2311.07946v3)

**Authors**: Adam Piaseczny, Eric Ruzomberka, Rohit Parasnis, Christopher G. Brinton

**Abstract**: As Federated Learning (FL) grows in popularity, new decentralized frameworks are becoming widespread. These frameworks leverage the benefits of decentralized environments to enable fast and energy-efficient inter-device communication. However, this growing popularity also intensifies the need for robust security measures. While existing research has explored various aspects of FL security, the role of adversarial node placement in decentralized networks remains largely unexplored. This paper addresses this gap by analyzing the performance of decentralized FL for various adversarial placement strategies when adversaries can jointly coordinate their placement within a network. We establish two baseline strategies for placing adversarial node: random placement and network centrality-based placement. Building on this foundation, we propose a novel attack algorithm that prioritizes adversarial spread over adversarial centrality by maximizing the average network distance between adversaries. We show that the new attack algorithm significantly impacts key performance metrics such as testing accuracy, outperforming the baseline frameworks by between 9% and 66.5% for the considered setups. Our findings provide valuable insights into the vulnerabilities of decentralized FL systems, setting the stage for future research aimed at developing more secure and robust decentralized FL frameworks.



## **8. GE-AdvGAN: Improving the transferability of adversarial samples by gradient editing-based adversarial generative model**

cs.CV

Accepted by SIAM International Conference on Data Mining (SDM24)

**SubmitDate**: 2024-01-11    [abs](http://arxiv.org/abs/2401.06031v1) [paper-pdf](http://arxiv.org/pdf/2401.06031v1)

**Authors**: Zhiyu Zhu, Huaming Chen, Xinyi Wang, Jiayu Zhang, Zhibo Jin, Kim-Kwang Raymond Choo

**Abstract**: Adversarial generative models, such as Generative Adversarial Networks (GANs), are widely applied for generating various types of data, i.e., images, text, and audio. Accordingly, its promising performance has led to the GAN-based adversarial attack methods in the white-box and black-box attack scenarios. The importance of transferable black-box attacks lies in their ability to be effective across different models and settings, more closely aligning with real-world applications. However, it remains challenging to retain the performance in terms of transferable adversarial examples for such methods. Meanwhile, we observe that some enhanced gradient-based transferable adversarial attack algorithms require prolonged time for adversarial sample generation. Thus, in this work, we propose a novel algorithm named GE-AdvGAN to enhance the transferability of adversarial samples whilst improving the algorithm's efficiency. The main approach is via optimising the training process of the generator parameters. With the functional and characteristic similarity analysis, we introduce a novel gradient editing (GE) mechanism and verify its feasibility in generating transferable samples on various models. Moreover, by exploring the frequency domain information to determine the gradient editing direction, GE-AdvGAN can generate highly transferable adversarial samples while minimizing the execution time in comparison to the state-of-the-art transferable adversarial attack algorithms. The performance of GE-AdvGAN is comprehensively evaluated by large-scale experiments on different datasets, which results demonstrate the superiority of our algorithm. The code for our algorithm is available at: https://github.com/LMBTough/GE-advGAN



## **9. Combating Adversarial Attacks with Multi-Agent Debate**

cs.CL

**SubmitDate**: 2024-01-11    [abs](http://arxiv.org/abs/2401.05998v1) [paper-pdf](http://arxiv.org/pdf/2401.05998v1)

**Authors**: Steffi Chern, Zhen Fan, Andy Liu

**Abstract**: While state-of-the-art language models have achieved impressive results, they remain susceptible to inference-time adversarial attacks, such as adversarial prompts generated by red teams arXiv:2209.07858. One approach proposed to improve the general quality of language model generations is multi-agent debate, where language models self-evaluate through discussion and feedback arXiv:2305.14325. We implement multi-agent debate between current state-of-the-art language models and evaluate models' susceptibility to red team attacks in both single- and multi-agent settings. We find that multi-agent debate can reduce model toxicity when jailbroken or less capable models are forced to debate with non-jailbroken or more capable models. We also find marginal improvements through the general usage of multi-agent interactions. We further perform adversarial prompt content classification via embedding clustering, and analyze the susceptibility of different models to different types of attack topics.



## **10. Localized adversarial artifacts for compressed sensing MRI**

eess.IV

14 pages, 7 figures

**SubmitDate**: 2024-01-11    [abs](http://arxiv.org/abs/2206.05289v2) [paper-pdf](http://arxiv.org/pdf/2206.05289v2)

**Authors**: Rima Alaifari, Giovanni S. Alberti, Tandri Gauksson

**Abstract**: As interest in deep neural networks (DNNs) for image reconstruction tasks grows, their reliability has been called into question (Antun et al., 2020; Gottschling et al., 2020). However, recent work has shown that, compared to total variation (TV) minimization, when appropriately regularized, DNNs show similar robustness to adversarial noise in terms of $\ell^2$-reconstruction error (Genzel et al., 2022). We consider a different notion of robustness, using the $\ell^\infty$-norm, and argue that localized reconstruction artifacts are a more relevant defect than the $\ell^2$-error. We create adversarial perturbations to undersampled magnetic resonance imaging measurements (in the frequency domain) which induce severe localized artifacts in the TV-regularized reconstruction. Notably, the same attack method is not as effective against DNN based reconstruction. Finally, we show that this phenomenon is inherent to reconstruction methods for which exact recovery can be guaranteed, as with compressed sensing reconstructions with $\ell^1$- or TV-minimization.



## **11. Towards Robust Pruning: An Adaptive Knowledge-Retention Pruning Strategy for Language Models**

cs.CL

**SubmitDate**: 2024-01-11    [abs](http://arxiv.org/abs/2310.13191v3) [paper-pdf](http://arxiv.org/pdf/2310.13191v3)

**Authors**: Jianwei Li, Qi Lei, Wei Cheng, Dongkuan Xu

**Abstract**: The pruning objective has recently extended beyond accuracy and sparsity to robustness in language models. Despite this, existing methods struggle to enhance robustness against adversarial attacks when continually increasing model sparsity and require a retraining process. As humans step into the era of large language models, these issues become increasingly prominent. This paper proposes that the robustness of language models is proportional to the extent of pre-trained knowledge they encompass. Accordingly, we introduce a post-training pruning strategy designed to faithfully replicate the embedding space and feature space of dense language models, aiming to conserve more pre-trained knowledge during the pruning process. In this setup, each layer's reconstruction error not only originates from itself but also includes cumulative error from preceding layers, followed by an adaptive rectification. Compared to other state-of-art baselines, our approach demonstrates a superior balance between accuracy, sparsity, robustness, and pruning cost with BERT on datasets SST2, IMDB, and AGNews, marking a significant stride towards robust pruning in language models.



## **12. Dynamics-aware Adversarial Attack of Adaptive Neural Networks**

cs.CV

arXiv admin note: text overlap with arXiv:2112.09428

**SubmitDate**: 2024-01-11    [abs](http://arxiv.org/abs/2210.08159v4) [paper-pdf](http://arxiv.org/pdf/2210.08159v4)

**Authors**: An Tao, Yueqi Duan, Yingqi Wang, Jiwen Lu, Jie Zhou

**Abstract**: In this paper, we investigate the dynamics-aware adversarial attack problem of adaptive neural networks. Most existing adversarial attack algorithms are designed under a basic assumption -- the network architecture is fixed throughout the attack process. However, this assumption does not hold for many recently proposed adaptive neural networks, which adaptively deactivate unnecessary execution units based on inputs to improve computational efficiency. It results in a serious issue of lagged gradient, making the learned attack at the current step ineffective due to the architecture change afterward. To address this issue, we propose a Leaded Gradient Method (LGM) and show the significant effects of the lagged gradient. More specifically, we reformulate the gradients to be aware of the potential dynamic changes of network architectures, so that the learned attack better "leads" the next step than the dynamics-unaware methods when network architecture changes dynamically. Extensive experiments on representative types of adaptive neural networks for both 2D images and 3D point clouds show that our LGM achieves impressive adversarial attack performance compared with the dynamic-unaware attack methods. Code is available at https://github.com/antao97/LGM.



## **13. The Devil's Advocate: Shattering the Illusion of Unexploitable Data using Diffusion Models**

cs.LG

Accepted to the 2024 IEEE Conference on Secure and Trustworthy  Machine Learning (SatML)

**SubmitDate**: 2024-01-11    [abs](http://arxiv.org/abs/2303.08500v2) [paper-pdf](http://arxiv.org/pdf/2303.08500v2)

**Authors**: Hadi M. Dolatabadi, Sarah Erfani, Christopher Leckie

**Abstract**: Protecting personal data against exploitation of machine learning models is crucial. Recently, availability attacks have shown great promise to provide an extra layer of protection against the unauthorized use of data to train neural networks. These methods aim to add imperceptible noise to clean data so that the neural networks cannot extract meaningful patterns from the protected data, claiming that they can make personal data "unexploitable." This paper provides a strong countermeasure against such approaches, showing that unexploitable data might only be an illusion. In particular, we leverage the power of diffusion models and show that a carefully designed denoising process can counteract the effectiveness of the data-protecting perturbations. We rigorously analyze our algorithm, and theoretically prove that the amount of required denoising is directly related to the magnitude of the data-protecting perturbations. Our approach, called AVATAR, delivers state-of-the-art performance against a suite of recent availability attacks in various scenarios, outperforming adversarial training even under distribution mismatch between the diffusion model and the protected data. Our findings call for more research into making personal data unexploitable, showing that this goal is far from over. Our implementation is available at this repository: https://github.com/hmdolatabadi/AVATAR.



## **14. A general theory for robust clustering via trimmed mean**

math.ST

51 pages

**SubmitDate**: 2024-01-10    [abs](http://arxiv.org/abs/2401.05574v1) [paper-pdf](http://arxiv.org/pdf/2401.05574v1)

**Authors**: Soham Jana, Jianqing Fan, Sanjeev Kulkarni

**Abstract**: Clustering is a fundamental tool in statistical machine learning in the presence of heterogeneous data. Many recent results focus primarily on optimal mislabeling guarantees, when data are distributed around centroids with sub-Gaussian errors. Yet, the restrictive sub-Gaussian model is often invalid in practice, since various real-world applications exhibit heavy tail distributions around the centroids or suffer from possible adversarial attacks that call for robust clustering with a robust data-driven initialization. In this paper, we introduce a hybrid clustering technique with a novel multivariate trimmed mean type centroid estimate to produce mislabeling guarantees under a weak initialization condition for general error distributions around the centroids. A matching lower bound is derived, up to factors depending on the number of clusters. In addition, our approach also produces the optimal mislabeling even in the presence of adversarial outliers. Our results reduce to the sub-Gaussian case when errors follow sub-Gaussian distributions. To solve the problem thoroughly, we also present novel data-driven robust initialization techniques and show that, with probabilities approaching one, these initial centroid estimates are sufficiently good for the subsequent clustering algorithm to achieve the optimal mislabeling rates. Furthermore, we demonstrate that the Lloyd algorithm is suboptimal for more than two clusters even when errors are Gaussian, and for two clusters when errors distributions have heavy tails. Both simulated data and real data examples lend further support to both of our robust initialization procedure and clustering algorithm.



## **15. LimeAttack: Local Explainable Method for Textual Hard-Label Adversarial Attack**

cs.CL

18 pages, 38th AAAI Main Track

**SubmitDate**: 2024-01-10    [abs](http://arxiv.org/abs/2308.00319v2) [paper-pdf](http://arxiv.org/pdf/2308.00319v2)

**Authors**: Hai Zhu, Zhaoqing Yang, Weiwei Shang, Yuren Wu

**Abstract**: Natural language processing models are vulnerable to adversarial examples. Previous textual adversarial attacks adopt gradients or confidence scores to calculate word importance ranking and generate adversarial examples. However, this information is unavailable in the real world. Therefore, we focus on a more realistic and challenging setting, named hard-label attack, in which the attacker can only query the model and obtain a discrete prediction label. Existing hard-label attack algorithms tend to initialize adversarial examples by random substitution and then utilize complex heuristic algorithms to optimize the adversarial perturbation. These methods require a lot of model queries and the attack success rate is restricted by adversary initialization. In this paper, we propose a novel hard-label attack algorithm named LimeAttack, which leverages a local explainable method to approximate word importance ranking, and then adopts beam search to find the optimal solution. Extensive experiments show that LimeAttack achieves the better attacking performance compared with existing hard-label attack under the same query budget. In addition, we evaluate the effectiveness of LimeAttack on large language models, and results indicate that adversarial examples remain a significant threat to large language models. The adversarial examples crafted by LimeAttack are highly transferable and effectively improve model robustness in adversarial training.



## **16. A Theoretical View of Linear Backpropagation and Its Convergence**

cs.LG

This paper is accepted by IEEE Transactions on Pattern Analysis and  Machine Intelligence

**SubmitDate**: 2024-01-10    [abs](http://arxiv.org/abs/2112.11018v2) [paper-pdf](http://arxiv.org/pdf/2112.11018v2)

**Authors**: Ziang Li, Yiwen Guo, Haodi Liu, Changshui Zhang

**Abstract**: Backpropagation (BP) is widely used for calculating gradients in deep neural networks (DNNs). Applied often along with stochastic gradient descent (SGD) or its variants, BP is considered as a de-facto choice in a variety of machine learning tasks including DNN training and adversarial attack/defense. Recently, a linear variant of BP named LinBP was introduced for generating more transferable adversarial examples for performing black-box attacks, by Guo et al. Although it has been shown empirically effective in black-box attacks, theoretical studies and convergence analyses of such a method is lacking. This paper serves as a complement and somewhat an extension to Guo et al.'s paper, by providing theoretical analyses on LinBP in neural-network-involved learning tasks, including adversarial attack and model training. We demonstrate that, somewhat surprisingly, LinBP can lead to faster convergence in these tasks in the same hyper-parameter settings, compared to BP. We confirm our theoretical results with extensive experiments.



## **17. Lyapunov-Stable Deep Equilibrium Models**

cs.LG

**SubmitDate**: 2024-01-10    [abs](http://arxiv.org/abs/2304.12707v3) [paper-pdf](http://arxiv.org/pdf/2304.12707v3)

**Authors**: Haoyu Chu, Shikui Wei, Ting Liu, Yao Zhao, Yuto Miyatake

**Abstract**: Deep equilibrium (DEQ) models have emerged as a promising class of implicit layer models, which abandon traditional depth by solving for the fixed points of a single nonlinear layer. Despite their success, the stability of the fixed points for these models remains poorly understood. By considering DEQ models as nonlinear dynamic systems, we propose a robust DEQ model named LyaDEQ with guaranteed provable stability via Lyapunov theory. The crux of our method is ensuring the Lyapunov stability of the DEQ model's fixed points, which enables the proposed model to resist minor initial perturbations. To avoid poor adversarial defense due to Lyapunov-stable fixed points being located near each other, we orthogonalize the layers after the Lyapunov stability module to separate different fixed points. We evaluate LyaDEQ models under well-known adversarial attacks, and experimental results demonstrate significant improvement in robustness. Furthermore, we show that the LyaDEQ model can be combined with other defense methods, such as adversarial training, to achieve even better adversarial robustness.



## **18. Learning-Based Difficulty Calibration for Enhanced Membership Inference Attacks**

cs.CR

10 pages, 14 figures

**SubmitDate**: 2024-01-10    [abs](http://arxiv.org/abs/2401.04929v1) [paper-pdf](http://arxiv.org/pdf/2401.04929v1)

**Authors**: Haonan Shi, Tu Ouyang, An Wang

**Abstract**: Machine learning models, in particular deep neural networks, are currently an integral part of various applications, from healthcare to finance. However, using sensitive data to train these models raises concerns about privacy and security. One method that has emerged to verify if the trained models are privacy-preserving is Membership Inference Attacks (MIA), which allows adversaries to determine whether a specific data point was part of a model's training dataset. While a series of MIAs have been proposed in the literature, only a few can achieve high True Positive Rates (TPR) in the low False Positive Rate (FPR) region (0.01%~1%). This is a crucial factor to consider for an MIA to be practically useful in real-world settings. In this paper, we present a novel approach to MIA that is aimed at significantly improving TPR at low FPRs. Our method, named learning-based difficulty calibration for MIA(LDC-MIA), characterizes data records by their hardness levels using a neural network classifier to determine membership. The experiment results show that LDC-MIA can improve TPR at low FPR by up to 4x compared to the other difficulty calibration based MIAs. It also has the highest Area Under ROC curve (AUC) across all datasets. Our method's cost is comparable with most of the existing MIAs, but is orders of magnitude more efficient than one of the state-of-the-art methods, LiRA, while achieving similar performance.



## **19. AdvSQLi: Generating Adversarial SQL Injections against Real-world WAF-as-a-service**

cs.CR

Accepted by IEEE Transactions on Information Forensics and Security  (IEEE TIFS)

**SubmitDate**: 2024-01-09    [abs](http://arxiv.org/abs/2401.02615v3) [paper-pdf](http://arxiv.org/pdf/2401.02615v3)

**Authors**: Zhenqing Qu, Xiang Ling, Ting Wang, Xiang Chen, Shouling Ji, Chunming Wu

**Abstract**: As the first defensive layer that attacks would hit, the web application firewall (WAF) plays an indispensable role in defending against malicious web attacks like SQL injection (SQLi). With the development of cloud computing, WAF-as-a-service, as one kind of Security-as-a-service, has been proposed to facilitate the deployment, configuration, and update of WAFs in the cloud. Despite its tremendous popularity, the security vulnerabilities of WAF-as-a-service are still largely unknown, which is highly concerning given its massive usage. In this paper, we propose a general and extendable attack framework, namely AdvSQLi, in which a minimal series of transformations are performed on the hierarchical tree representation of the original SQLi payload, such that the generated SQLi payloads can not only bypass WAF-as-a-service under black-box settings but also keep the same functionality and maliciousness as the original payload. With AdvSQLi, we make it feasible to inspect and understand the security vulnerabilities of WAFs automatically, helping vendors make products more secure. To evaluate the attack effectiveness and efficiency of AdvSQLi, we first employ two public datasets to generate adversarial SQLi payloads, leading to a maximum attack success rate of 100% against state-of-the-art ML-based SQLi detectors. Furthermore, to demonstrate the immediate security threats caused by AdvSQLi, we evaluate the attack effectiveness against 7 WAF-as-a-service solutions from mainstream vendors and find all of them are vulnerable to AdvSQLi. For instance, AdvSQLi achieves an attack success rate of over 79% against the F5 WAF. Through in-depth analysis of the evaluation results, we further condense out several general yet severe flaws of these vendors that cannot be easily patched.



## **20. ROIC-DM: Robust Text Inference and Classification via Diffusion Model**

cs.CL

under review

**SubmitDate**: 2024-01-09    [abs](http://arxiv.org/abs/2401.03514v2) [paper-pdf](http://arxiv.org/pdf/2401.03514v2)

**Authors**: Shilong Yuan, Wei Yuan, Hongzhi Yin, Tieke He

**Abstract**: While language models have made many milestones in text inference and classification tasks, they remain susceptible to adversarial attacks that can lead to unforeseen outcomes. Existing works alleviate this problem by equipping language models with defense patches. However, these defense strategies often rely on impractical assumptions or entail substantial sacrifices in model performance. Consequently, enhancing the resilience of the target model using such defense mechanisms is a formidable challenge. This paper introduces an innovative model for robust text inference and classification, built upon diffusion models (ROIC-DM). Benefiting from its training involving denoising stages, ROIC-DM inherently exhibits greater robustness compared to conventional language models. Moreover, ROIC-DM can attain comparable, and in some cases, superior performance to language models, by effectively incorporating them as advisory components. Extensive experiments conducted with several strong textual adversarial attacks on three datasets demonstrate that (1) ROIC-DM outperforms traditional language models in robustness, even when the latter are fortified with advanced defense mechanisms; (2) ROIC-DM can achieve comparable and even better performance than traditional language models by using them as advisors.



## **21. Exploring Adversarial Robustness of LiDAR-Camera Fusion Model in Autonomous Driving**

cs.RO

**SubmitDate**: 2024-01-09    [abs](http://arxiv.org/abs/2312.01468v2) [paper-pdf](http://arxiv.org/pdf/2312.01468v2)

**Authors**: Bo Yang, Xiaoyu Ji, Zizhi Jin, Yushi Cheng, Wenyuan Xu

**Abstract**: Our study assesses the adversarial robustness of LiDAR-camera fusion models in 3D object detection. We introduce an attack technique that, by simply adding a limited number of physically constrained adversarial points above a car, can make the car undetectable by the fusion model. Experimental results reveal that even without changes to the image data channel, the fusion model can be deceived solely by manipulating the LiDAR data channel. This finding raises safety concerns in the field of autonomous driving. Further, we explore how the quantity of adversarial points, the distance between the front-near car and the LiDAR-equipped car, and various angular factors affect the attack success rate. We believe our research can contribute to the understanding of multi-sensor robustness, offering insights and guidance to enhance the safety of autonomous driving.



## **22. Coupling Graph Neural Networks with Fractional Order Continuous Dynamics: A Robustness Study**

cs.LG

in Proc. AAAI Conference on Artificial Intelligence, Vancouver,  Canada, Feb. 2024

**SubmitDate**: 2024-01-09    [abs](http://arxiv.org/abs/2401.04331v1) [paper-pdf](http://arxiv.org/pdf/2401.04331v1)

**Authors**: Qiyu Kang, Kai Zhao, Yang Song, Yihang Xie, Yanan Zhao, Sijie Wang, Rui She, Wee Peng Tay

**Abstract**: In this work, we rigorously investigate the robustness of graph neural fractional-order differential equation (FDE) models. This framework extends beyond traditional graph neural (integer-order) ordinary differential equation (ODE) models by implementing the time-fractional Caputo derivative. Utilizing fractional calculus allows our model to consider long-term memory during the feature updating process, diverging from the memoryless Markovian updates seen in traditional graph neural ODE models. The superiority of graph neural FDE models over graph neural ODE models has been established in environments free from attacks or perturbations. While traditional graph neural ODE models have been verified to possess a degree of stability and resilience in the presence of adversarial attacks in existing literature, the robustness of graph neural FDE models, especially under adversarial conditions, remains largely unexplored. This paper undertakes a detailed assessment of the robustness of graph neural FDE models. We establish a theoretical foundation outlining the robustness characteristics of graph neural FDE models, highlighting that they maintain more stringent output perturbation bounds in the face of input and graph topology disturbances, compared to their integer-order counterparts. Our empirical evaluations further confirm the enhanced robustness of graph neural FDE models, highlighting their potential in adversarially robust applications.



## **23. Cross-Class Feature Augmentation for Class Incremental Learning**

cs.CV

**SubmitDate**: 2024-01-09    [abs](http://arxiv.org/abs/2304.01899v3) [paper-pdf](http://arxiv.org/pdf/2304.01899v3)

**Authors**: Taehoon Kim, Jaeyoo Park, Bohyung Han

**Abstract**: We propose a novel class incremental learning approach by incorporating a feature augmentation technique motivated by adversarial attacks. We employ a classifier learned in the past to complement training examples rather than simply play a role as a teacher for knowledge distillation towards subsequent models. The proposed approach has a unique perspective to utilize the previous knowledge in class incremental learning since it augments features of arbitrary target classes using examples in other classes via adversarial attacks on a previously learned classifier. By allowing the cross-class feature augmentations, each class in the old tasks conveniently populates samples in the feature space, which alleviates the collapse of the decision boundaries caused by sample deficiency for the previous tasks, especially when the number of stored exemplars is small. This idea can be easily incorporated into existing class incremental learning algorithms without any architecture modification. Extensive experiments on the standard benchmarks show that our method consistently outperforms existing class incremental learning methods by significant margins in various scenarios, especially under an environment with an extremely limited memory budget.



## **24. Vulnerabilities Unveiled: Adversarially Attacking a Multimodal Vision Language Model for Pathology Imaging**

eess.IV

**SubmitDate**: 2024-01-08    [abs](http://arxiv.org/abs/2401.02565v2) [paper-pdf](http://arxiv.org/pdf/2401.02565v2)

**Authors**: Jai Prakash Veerla, Poojitha Thota, Partha Sai Guttikonda, Shirin Nilizadeh, Jacob M. Luber

**Abstract**: In the dynamic landscape of medical artificial intelligence, this study explores the vulnerabilities of the Pathology Language-Image Pretraining (PLIP) model, a Vision Language Foundation model, under targeted adversarial conditions. Leveraging the Kather Colon dataset with 7,180 H&E images across nine tissue types, our investigation employs Projected Gradient Descent (PGD) adversarial attacks to intentionally induce misclassifications. The outcomes reveal a 100% success rate in manipulating PLIP's predictions, underscoring its susceptibility to adversarial perturbations. The qualitative analysis of adversarial examples delves into the interpretability challenges, shedding light on nuanced changes in predictions induced by adversarial manipulations. These findings contribute crucial insights into the interpretability, domain adaptation, and trustworthiness of Vision Language Models in medical imaging. The study emphasizes the pressing need for robust defenses to ensure the reliability of AI models.



## **25. Logits Poisoning Attack in Federated Distillation**

cs.LG

13 pages, 3 figures, 5 tables

**SubmitDate**: 2024-01-08    [abs](http://arxiv.org/abs/2401.03685v1) [paper-pdf](http://arxiv.org/pdf/2401.03685v1)

**Authors**: Yuhan Tang, Zhiyuan Wu, Bo Gao, Tian Wen, Yuwei Wang, Sheng Sun

**Abstract**: Federated Distillation (FD) is a novel and promising distributed machine learning paradigm, where knowledge distillation is leveraged to facilitate a more efficient and flexible cross-device knowledge transfer in federated learning. By optimizing local models with knowledge distillation, FD circumvents the necessity of uploading large-scale model parameters to the central server, simultaneously preserving the raw data on local clients. Despite the growing popularity of FD, there is a noticeable gap in previous works concerning the exploration of poisoning attacks within this framework. This can lead to a scant understanding of the vulnerabilities to potential adversarial actions. To this end, we introduce FDLA, a poisoning attack method tailored for FD. FDLA manipulates logit communications in FD, aiming to significantly degrade model performance on clients through misleading the discrimination of private samples. Through extensive simulation experiments across a variety of datasets, attack scenarios, and FD configurations, we demonstrate that LPA effectively compromises client model accuracy, outperforming established baseline algorithms in this regard. Our findings underscore the critical need for robust defense mechanisms in FD settings to mitigate such adversarial threats.



## **26. Assessing the Influence of Different Types of Probing on Adversarial Decision-Making in a Deception Game**

cs.CR

**SubmitDate**: 2024-01-08    [abs](http://arxiv.org/abs/2310.10662v3) [paper-pdf](http://arxiv.org/pdf/2310.10662v3)

**Authors**: Md Abu Sayed, Mohammad Ariful Islam Khan, Bryant A Allsup, Joshua Zamora, Palvi Aggarwal

**Abstract**: Deception, which includes leading cyber-attackers astray with false information, has shown to be an effective method of thwarting cyber-attacks. There has been little investigation of the effect of probing action costs on adversarial decision-making, despite earlier studies on deception in cybersecurity focusing primarily on variables like network size and the percentage of honeypots utilized in games. Understanding human decision-making when prompted with choices of various costs is essential in many areas such as in cyber security. In this paper, we will use a deception game (DG) to examine different costs of probing on adversarial decisions. To achieve this we utilized an IBLT model and a delayed feedback mechanism to mimic knowledge of human actions. Our results were taken from an even split of deception and no deception to compare each influence. It was concluded that probing was slightly taken less as the cost of probing increased. The proportion of attacks stayed relatively the same as the cost of probing increased. Although a constant cost led to a slight decrease in attacks. Overall, our results concluded that the different probing costs do not have an impact on the proportion of attacks whereas it had a slightly noticeable impact on the proportion of probing.



## **27. Elevating Defenses: Bridging Adversarial Training and Watermarking for Model Resilience**

cs.LG

Accepted at DAI Workshop, AAAI 2024

**SubmitDate**: 2024-01-07    [abs](http://arxiv.org/abs/2312.14260v2) [paper-pdf](http://arxiv.org/pdf/2312.14260v2)

**Authors**: Janvi Thakkar, Giulio Zizzo, Sergio Maffeis

**Abstract**: Machine learning models are being used in an increasing number of critical applications; thus, securing their integrity and ownership is critical. Recent studies observed that adversarial training and watermarking have a conflicting interaction. This work introduces a novel framework to integrate adversarial training with watermarking techniques to fortify against evasion attacks and provide confident model verification in case of intellectual property theft. We use adversarial training together with adversarial watermarks to train a robust watermarked model. The key intuition is to use a higher perturbation budget to generate adversarial watermarks compared to the budget used for adversarial training, thus avoiding conflict. We use the MNIST and Fashion-MNIST datasets to evaluate our proposed technique on various model stealing attacks. The results obtained consistently outperform the existing baseline in terms of robustness performance and further prove the resilience of this defense against pruning and fine-tuning removal attacks.



## **28. Data-Driven Subsampling in the Presence of an Adversarial Actor**

cs.LG

Accepted for publication at ICMLCN 2024

**SubmitDate**: 2024-01-07    [abs](http://arxiv.org/abs/2401.03488v1) [paper-pdf](http://arxiv.org/pdf/2401.03488v1)

**Authors**: Abu Shafin Mohammad Mahdee Jameel, Ahmed P. Mohamed, Jinho Yi, Aly El Gamal, Akshay Malhotra

**Abstract**: Deep learning based automatic modulation classification (AMC) has received significant attention owing to its potential applications in both military and civilian use cases. Recently, data-driven subsampling techniques have been utilized to overcome the challenges associated with computational complexity and training time for AMC. Beyond these direct advantages of data-driven subsampling, these methods also have regularizing properties that may improve the adversarial robustness of the modulation classifier. In this paper, we investigate the effects of an adversarial attack on an AMC system that employs deep learning models both for AMC and for subsampling. Our analysis shows that subsampling itself is an effective deterrent to adversarial attacks. We also uncover the most efficient subsampling strategy when an adversarial attack on both the classifier and the subsampler is anticipated.



## **29. Token-Modification Adversarial Attacks for Natural Language Processing: A Survey**

cs.CL

Version 3: edited and expanded

**SubmitDate**: 2024-01-07    [abs](http://arxiv.org/abs/2103.00676v3) [paper-pdf](http://arxiv.org/pdf/2103.00676v3)

**Authors**: Tom Roth, Yansong Gao, Alsharif Abuadbba, Surya Nepal, Wei Liu

**Abstract**: Many adversarial attacks target natural language processing systems, most of which succeed through modifying the individual tokens of a document. Despite the apparent uniqueness of each of these attacks, fundamentally they are simply a distinct configuration of four components: a goal function, allowable transformations, a search method, and constraints. In this survey, we systematically present the different components used throughout the literature, using an attack-independent framework which allows for easy comparison and categorisation of components. Our work aims to serve as a comprehensive guide for newcomers to the field and to spark targeted research into refining the individual attack components.



## **30. Affinity Uncertainty-based Hard Negative Mining in Graph Contrastive Learning**

cs.LG

Accepted to TNNLS

**SubmitDate**: 2024-01-07    [abs](http://arxiv.org/abs/2301.13340v2) [paper-pdf](http://arxiv.org/pdf/2301.13340v2)

**Authors**: Chaoxi Niu, Guansong Pang, Ling Chen

**Abstract**: Hard negative mining has shown effective in enhancing self-supervised contrastive learning (CL) on diverse data types, including graph CL (GCL). The existing hardness-aware CL methods typically treat negative instances that are most similar to the anchor instance as hard negatives, which helps improve the CL performance, especially on image data. However, this approach often fails to identify the hard negatives but leads to many false negatives on graph data. This is mainly due to that the learned graph representations are not sufficiently discriminative due to oversmooth representations and/or non-independent and identically distributed (non-i.i.d.) issues in graph data. To tackle this problem, this article proposes a novel approach that builds a discriminative model on collective affinity information (i.e., two sets of pairwise affinities between the negative instances and the anchor instance) to mine hard negatives in GCL. In particular, the proposed approach evaluates how confident/uncertain the discriminative model is about the affinity of each negative instance to an anchor instance to determine its hardness weight relative to the anchor instance. This uncertainty information is then incorporated into the existing GCL loss functions via a weighting term to enhance their performance. The enhanced GCL is theoretically grounded that the resulting GCL loss is equivalent to a triplet loss with an adaptive margin being exponentially proportional to the learned uncertainty of each negative instance. Extensive experiments on ten graph datasets show that our approach does the following: 1) consistently enhances different state-of-the-art (SOTA) GCL methods in both graph and node classification tasks and 2) significantly improves their robustness against adversarial attacks. Code is available at https://github.com/mala-lab/AUGCL.



## **31. Data-Dependent Stability Analysis of Adversarial Training**

cs.LG

**SubmitDate**: 2024-01-06    [abs](http://arxiv.org/abs/2401.03156v1) [paper-pdf](http://arxiv.org/pdf/2401.03156v1)

**Authors**: Yihan Wang, Shuang Liu, Xiao-Shan Gao

**Abstract**: Stability analysis is an essential aspect of studying the generalization ability of deep learning, as it involves deriving generalization bounds for stochastic gradient descent-based training algorithms. Adversarial training is the most widely used defense against adversarial example attacks. However, previous generalization bounds for adversarial training have not included information regarding the data distribution. In this paper, we fill this gap by providing generalization bounds for stochastic gradient descent-based adversarial training that incorporate data distribution information. We utilize the concepts of on-average stability and high-order approximate Lipschitz conditions to examine how changes in data distribution and adversarial budget can affect robust generalization gaps. Our derived generalization bounds for both convex and non-convex losses are at least as good as the uniform stability-based counterparts which do not include data distribution information. Furthermore, our findings demonstrate how distribution shifts from data poisoning attacks can impact robust generalization.



## **32. Transferable Learned Image Compression-Resistant Adversarial Perturbations**

cs.CV

Accepted as poster at Data Compression Conference 2024 (DCC 2024)

**SubmitDate**: 2024-01-06    [abs](http://arxiv.org/abs/2401.03115v1) [paper-pdf](http://arxiv.org/pdf/2401.03115v1)

**Authors**: Yang Sui, Zhuohang Li, Ding Ding, Xiang Pan, Xiaozhong Xu, Shan Liu, Zhenzhong Chen

**Abstract**: Adversarial attacks can readily disrupt the image classification system, revealing the vulnerability of DNN-based recognition tasks. While existing adversarial perturbations are primarily applied to uncompressed images or compressed images by the traditional image compression method, i.e., JPEG, limited studies have investigated the robustness of models for image classification in the context of DNN-based image compression. With the rapid evolution of advanced image compression, DNN-based learned image compression has emerged as the promising approach for transmitting images in many security-critical applications, such as cloud-based face recognition and autonomous driving, due to its superior performance over traditional compression. Therefore, there is a pressing need to fully investigate the robustness of a classification system post-processed by learned image compression. To bridge this research gap, we explore the adversarial attack on a new pipeline that targets image classification models that utilize learned image compressors as pre-processing modules. Furthermore, to enhance the transferability of perturbations across various quality levels and architectures of learned image compression models, we introduce a saliency score-based sampling method to enable the fast generation of transferable perturbation. Extensive experiments with popular attack methods demonstrate the enhanced transferability of our proposed method when attacking images that have been post-processed with different learned image compression models.



## **33. Lotto: Secure Participant Selection against Adversarial Servers in Federated Learning**

cs.CR

20 pages, 14 figures

**SubmitDate**: 2024-01-05    [abs](http://arxiv.org/abs/2401.02880v1) [paper-pdf](http://arxiv.org/pdf/2401.02880v1)

**Authors**: Zhifeng Jiang, Peng Ye, Shiqi He, Wei Wang, Ruichuan Chen, Bo Li

**Abstract**: In Federated Learning (FL), common privacy-preserving technologies, such as secure aggregation and distributed differential privacy, rely on the critical assumption of an honest majority among participants to withstand various attacks. In practice, however, servers are not always trusted, and an adversarial server can strategically select compromised clients to create a dishonest majority, thereby undermining the system's security guarantees. In this paper, we present Lotto, an FL system that addresses this fundamental, yet underexplored issue by providing secure participant selection against an adversarial server. Lotto supports two selection algorithms: random and informed. To ensure random selection without a trusted server, Lotto enables each client to autonomously determine their participation using verifiable randomness. For informed selection, which is more vulnerable to manipulation, Lotto approximates the algorithm by employing random selection within a refined client pool. Our theoretical analysis shows that Lotto effectively restricts the number of server-selected compromised clients, thus ensuring an honest majority among participants. Large-scale experiments further reveal that Lotto achieves time-to-accuracy performance comparable to that of insecure selection methods, indicating a low computational overhead for secure selection.



## **34. PromptBench: A Unified Library for Evaluation of Large Language Models**

cs.AI

An extension to PromptBench (arXiv:2306.04528) for unified evaluation  of LLMs using the same name; code: https://github.com/microsoft/promptbench

**SubmitDate**: 2024-01-05    [abs](http://arxiv.org/abs/2312.07910v2) [paper-pdf](http://arxiv.org/pdf/2312.07910v2)

**Authors**: Kaijie Zhu, Qinlin Zhao, Hao Chen, Jindong Wang, Xing Xie

**Abstract**: The evaluation of large language models (LLMs) is crucial to assess their performance and mitigate potential security risks. In this paper, we introduce PromptBench, a unified library to evaluate LLMs. It consists of several key components that are easily used and extended by researchers: prompt construction, prompt engineering, dataset and model loading, adversarial prompt attack, dynamic evaluation protocols, and analysis tools. PromptBench is designed to be an open, general, and flexible codebase for research purposes that can facilitate original study in creating new benchmarks, deploying downstream applications, and designing new evaluation protocols. The code is available at: https://github.com/microsoft/promptbench and will be continuously supported.



## **35. Enhancing targeted transferability via feature space fine-tuning**

cs.CV

9 pages, 10 figures, accepted by 2024ICASSP

**SubmitDate**: 2024-01-05    [abs](http://arxiv.org/abs/2401.02727v1) [paper-pdf](http://arxiv.org/pdf/2401.02727v1)

**Authors**: Hui Zeng, Biwei Chen, Anjie Peng

**Abstract**: Adversarial examples (AEs) have been extensively studied due to their potential for privacy protection and inspiring robust neural networks. However, making a targeted AE transferable across unknown models remains challenging. In this paper, to alleviate the overfitting dilemma common in an AE crafted by existing simple iterative attacks, we propose fine-tuning it in the feature space. Specifically, starting with an AE generated by a baseline attack, we encourage the features that contribute to the target class and discourage the features that contribute to the original class in a middle layer of the source model. Extensive experiments demonstrate that only a few iterations of fine-tuning can boost existing attacks in terms of targeted transferability nontrivially and universally. Our results also verify that the simple iterative attacks can yield comparable or even better transferability than the resource-intensive methods, which rely on training target-specific classifiers or generators with additional data. The code is available at: github.com/zengh5/TA_feature_FT.



## **36. Calibration Attack: A Framework For Adversarial Attacks Targeting Calibration**

cs.LG

**SubmitDate**: 2024-01-05    [abs](http://arxiv.org/abs/2401.02718v1) [paper-pdf](http://arxiv.org/pdf/2401.02718v1)

**Authors**: Stephen Obadinma, Xiaodan Zhu, Hongyu Guo

**Abstract**: We introduce a new framework of adversarial attacks, named calibration attacks, in which the attacks are generated and organized to trap victim models to be miscalibrated without altering their original accuracy, hence seriously endangering the trustworthiness of the models and any decision-making based on their confidence scores. Specifically, we identify four novel forms of calibration attacks: underconfidence attacks, overconfidence attacks, maximum miscalibration attacks, and random confidence attacks, in both the black-box and white-box setups. We then test these new attacks on typical victim models with comprehensive datasets, demonstrating that even with a relatively low number of queries, the attacks can create significant calibration mistakes. We further provide detailed analyses to understand different aspects of calibration attacks. Building on that, we investigate the effectiveness of widely used adversarial defences and calibration methods against these types of attacks, which then inspires us to devise two novel defences against such calibration attacks.



## **37. Game Theory for Adversarial Attacks and Defenses**

cs.LG

With the agreement of my coauthors, I would like to withdraw the  manuscript "Game Theory for Adversarial Attacks and Defenses". Some  experimental procedures were not included in the manuscript, which makes a  part of important claims not meaningful

**SubmitDate**: 2024-01-05    [abs](http://arxiv.org/abs/2110.06166v4) [paper-pdf](http://arxiv.org/pdf/2110.06166v4)

**Authors**: Shorya Sharma

**Abstract**: Adversarial attacks can generate adversarial inputs by applying small but intentionally worst-case perturbations to samples from the dataset, which leads to even state-of-the-art deep neural networks outputting incorrect answers with high confidence. Hence, some adversarial defense techniques are developed to improve the security and robustness of the models and avoid them being attacked. Gradually, a game-like competition between attackers and defenders formed, in which both players would attempt to play their best strategies against each other while maximizing their own payoffs. To solve the game, each player would choose an optimal strategy against the opponent based on the prediction of the opponent's strategy choice. In this work, we are on the defensive side to apply game-theoretic approaches on defending against attacks. We use two randomization methods, random initialization and stochastic activation pruning, to create diversity of networks. Furthermore, we use one denoising technique, super resolution, to improve models' robustness by preprocessing images before attacks. Our experimental results indicate that those three methods can effectively improve the robustness of deep-learning neural networks.



## **38. Secure Control of Connected and Automated Vehicles Using Trust-Aware Robust Event-Triggered Control Barrier Functions**

eess.SY

arXiv admin note: substantial text overlap with arXiv:2305.16818

**SubmitDate**: 2024-01-05    [abs](http://arxiv.org/abs/2401.02306v2) [paper-pdf](http://arxiv.org/pdf/2401.02306v2)

**Authors**: H M Sabbir Ahmad, Ehsan Sabouni, Akua Dickson, Wei Xiao, Christos G. Cassandras, Wenchao Li

**Abstract**: We address the security of a network of Connected and Automated Vehicles (CAVs) cooperating to safely navigate through a conflict area (e.g., traffic intersections, merging roadways, roundabouts). Previous studies have shown that such a network can be targeted by adversarial attacks causing traffic jams or safety violations ending in collisions. We focus on attacks targeting the V2X communication network used to share vehicle data and consider as well uncertainties due to noise in sensor measurements and communication channels. To combat these, motivated by recent work on the safe control of CAVs, we propose a trust-aware robust event-triggered decentralized control and coordination framework that can provably guarantee safety. We maintain a trust metric for each vehicle in the network computed based on their behavior and used to balance the tradeoff between conservativeness (when deeming every vehicle as untrustworthy) and guaranteed safety and security. It is important to highlight that our framework is invariant to the specific choice of the trust framework. Based on this framework, we propose an attack detection and mitigation scheme which has twofold benefits: (i) the trust framework is immune to false positives, and (ii) it provably guarantees safety against false positive cases. We use extensive simulations (in SUMO and CARLA) to validate the theoretical guarantees and demonstrate the efficacy of our proposed scheme to detect and mitigate adversarial attacks.



## **39. MalModel: Hiding Malicious Payload in Mobile Deep Learning Models with Black-box Backdoor Attack**

cs.CR

Due to the limitation "The abstract field cannot be longer than 1,920  characters", the abstract here is shorter than that in the PDF file

**SubmitDate**: 2024-01-05    [abs](http://arxiv.org/abs/2401.02659v1) [paper-pdf](http://arxiv.org/pdf/2401.02659v1)

**Authors**: Jiayi Hua, Kailong Wang, Meizhen Wang, Guangdong Bai, Xiapu Luo, Haoyu Wang

**Abstract**: Mobile malware has become one of the most critical security threats in the era of ubiquitous mobile computing. Despite the intensive efforts from security experts to counteract it, recent years have still witnessed a rapid growth of identified malware samples. This could be partly attributed to the newly-emerged technologies that may constantly open up under-studied attack surfaces for the adversaries. One typical example is the recently-developed mobile machine learning (ML) framework that enables storing and running deep learning (DL) models on mobile devices. Despite obvious advantages, this new feature also inadvertently introduces potential vulnerabilities (e.g., on-device models may be modified for malicious purposes). In this work, we propose a method to generate or transform mobile malware by hiding the malicious payloads inside the parameters of deep learning models, based on a strategy that considers four factors (layer type, layer number, layer coverage and the number of bytes to replace). Utilizing the proposed method, we can run malware in DL mobile applications covertly with little impact on the model performance (i.e., as little as 0.4% drop in accuracy and at most 39ms latency overhead).



## **40. A Random Ensemble of Encrypted models for Enhancing Robustness against Adversarial Examples**

cs.CR

4 pages

**SubmitDate**: 2024-01-05    [abs](http://arxiv.org/abs/2401.02633v1) [paper-pdf](http://arxiv.org/pdf/2401.02633v1)

**Authors**: Ryota Iijima, Sayaka Shiota, Hitoshi Kiya

**Abstract**: Deep neural networks (DNNs) are well known to be vulnerable to adversarial examples (AEs). In addition, AEs have adversarial transferability, which means AEs generated for a source model can fool another black-box model (target model) with a non-trivial probability. In previous studies, it was confirmed that the vision transformer (ViT) is more robust against the property of adversarial transferability than convolutional neural network (CNN) models such as ConvMixer, and moreover encrypted ViT is more robust than ViT without any encryption. In this article, we propose a random ensemble of encrypted ViT models to achieve much more robust models. In experiments, the proposed scheme is verified to be more robust against not only black-box attacks but also white-box ones than convention methods.



## **41. A Practical Survey on Emerging Threats from AI-driven Voice Attacks: How Vulnerable are Commercial Voice Control Systems?**

cs.CR

14 pages

**SubmitDate**: 2024-01-04    [abs](http://arxiv.org/abs/2312.06010v2) [paper-pdf](http://arxiv.org/pdf/2312.06010v2)

**Authors**: Yuanda Wang, Qiben Yan, Nikolay Ivanov, Xun Chen

**Abstract**: The emergence of Artificial Intelligence (AI)-driven audio attacks has revealed new security vulnerabilities in voice control systems. While researchers have introduced a multitude of attack strategies targeting voice control systems (VCS), the continual advancements of VCS have diminished the impact of many such attacks. Recognizing this dynamic landscape, our study endeavors to comprehensively assess the resilience of commercial voice control systems against a spectrum of malicious audio attacks. Through extensive experimentation, we evaluate six prominent attack techniques across a collection of voice control interfaces and devices. Contrary to prevailing narratives, our results suggest that commercial voice control systems exhibit enhanced resistance to existing threats. Particularly, our research highlights the ineffectiveness of white-box attacks in black-box scenarios. Furthermore, the adversaries encounter substantial obstacles in obtaining precise gradient estimations during query-based interactions with commercial systems, such as Apple Siri and Samsung Bixby. Meanwhile, we find that current defense strategies are not completely immune to advanced attacks. Our findings contribute valuable insights for enhancing defense mechanisms in VCS. Through this survey, we aim to raise awareness within the academic community about the security concerns of VCS and advocate for continued research in this crucial area.



## **42. Evasive Hardware Trojan through Adversarial Power Trace**

cs.CR

**SubmitDate**: 2024-01-04    [abs](http://arxiv.org/abs/2401.02342v1) [paper-pdf](http://arxiv.org/pdf/2401.02342v1)

**Authors**: Behnam Omidi, Khaled N. Khasawneh, Ihsen Alouani

**Abstract**: The globalization of the Integrated Circuit (IC) supply chain, driven by time-to-market and cost considerations, has made ICs vulnerable to hardware Trojans (HTs). Against this threat, a promising approach is to use Machine Learning (ML)-based side-channel analysis, which has the advantage of being a non-intrusive method, along with efficiently detecting HTs under golden chip-free settings. In this paper, we question the trustworthiness of ML-based HT detection via side-channel analysis. We introduce a HT obfuscation (HTO) approach to allow HTs to bypass this detection method. Rather than theoretically misleading the model by simulated adversarial traces, a key aspect of our approach is the design and implementation of adversarial noise as part of the circuitry, alongside the HT. We detail HTO methodologies for ASICs and FPGAs, and evaluate our approach using TrustHub benchmark. Interestingly, we found that HTO can be implemented with only a single transistor for ASIC designs to generate adversarial power traces that can fool the defense with 100% efficiency. We also efficiently implemented our approach on a Spartan 6 Xilinx FPGA using 2 different variants: (i) DSP slices-based, and (ii) ring-oscillator-based design. Additionally, we assess the efficiency of countermeasures like spectral domain analysis, and we show that an adaptive attacker can still design evasive HTOs by constraining the design with a spectral noise budget. In addition, while adversarial training (AT) offers higher protection against evasive HTs, AT models suffer from a considerable utility loss, potentially rendering them unsuitable for such security application. We believe this research represents a significant step in understanding and exploiting ML vulnerabilities in a hardware security context, and we make all resources and designs openly available online: https://dev.d18uu4lqwhbmka.amplifyapp.com



## **43. Adversarial Data Poisoning for Fake News Detection: How to Make a Model Misclassify a Target News without Modifying It**

cs.LG

**SubmitDate**: 2024-01-04    [abs](http://arxiv.org/abs/2312.15228v2) [paper-pdf](http://arxiv.org/pdf/2312.15228v2)

**Authors**: Federico Siciliano, Luca Maiano, Lorenzo Papa, Federica Baccini, Irene Amerini, Fabrizio Silvestri

**Abstract**: Fake news detection models are critical to countering disinformation but can be manipulated through adversarial attacks. In this position paper, we analyze how an attacker can compromise the performance of an online learning detector on specific news content without being able to manipulate the original target news. In some contexts, such as social networks, where the attacker cannot exert complete control over all the information, this scenario can indeed be quite plausible. Therefore, we show how an attacker could potentially introduce poisoning data into the training data to manipulate the behavior of an online learning method. Our initial findings reveal varying susceptibility of logistic regression models based on complexity and attack type.



## **44. Attacks in Adversarial Machine Learning: A Systematic Survey from the Life-cycle Perspective**

cs.LG

35 pages, 4 figures, 10 tables, 313 reference papers

**SubmitDate**: 2024-01-04    [abs](http://arxiv.org/abs/2302.09457v2) [paper-pdf](http://arxiv.org/pdf/2302.09457v2)

**Authors**: Baoyuan Wu, Zihao Zhu, Li Liu, Qingshan Liu, Zhaofeng He, Siwei Lyu

**Abstract**: Adversarial machine learning (AML) studies the adversarial phenomenon of machine learning, which may make inconsistent or unexpected predictions with humans. Some paradigms have been recently developed to explore this adversarial phenomenon occurring at different stages of a machine learning system, such as backdoor attack occurring at the pre-training, in-training and inference stage; weight attack occurring at the post-training, deployment and inference stage; adversarial attack occurring at the inference stage. However, although these adversarial paradigms share a common goal, their developments are almost independent, and there is still no big picture of AML. In this work, we aim to provide a unified perspective to the AML community to systematically review the overall progress of this field. We firstly provide a general definition about AML, and then propose a unified mathematical framework to covering existing attack paradigms. According to the proposed unified framework, we build a full taxonomy to systematically categorize and review existing representative methods for each paradigm. Besides, using this unified framework, it is easy to figure out the connections and differences among different attack paradigms, which may inspire future researchers to develop more advanced attack paradigms. Finally, to facilitate the viewing of the built taxonomy and the related literature in adversarial machine learning, we further provide a website, \ie, \url{http://adversarial-ml.com}, where the taxonomies and literature will be continuously updated.



## **45. Fast Certification of Vision-Language Models Using Incremental Randomized Smoothing**

cs.CV

**SubmitDate**: 2024-01-04    [abs](http://arxiv.org/abs/2311.09024v2) [paper-pdf](http://arxiv.org/pdf/2311.09024v2)

**Authors**: A K Nirala, A Joshi, C Hegde, S Sarkar

**Abstract**: A key benefit of deep vision-language models such as CLIP is that they enable zero-shot open vocabulary classification; the user has the ability to define novel class labels via natural language prompts at inference time. However, while CLIP-based zero-shot classifiers have demonstrated competitive performance across a range of domain shifts, they remain highly vulnerable to adversarial attacks. Therefore, ensuring the robustness of such models is crucial for their reliable deployment in the wild.   In this work, we introduce Open Vocabulary Certification (OVC), a fast certification method designed for open-vocabulary models like CLIP via randomized smoothing techniques. Given a base "training" set of prompts and their corresponding certified CLIP classifiers, OVC relies on the observation that a classifier with a novel prompt can be viewed as a perturbed version of nearby classifiers in the base training set. Therefore, OVC can rapidly certify the novel classifier using a variation of incremental randomized smoothing. By using a caching trick, we achieve approximately two orders of magnitude acceleration in the certification process for novel prompts. To achieve further (heuristic) speedups, OVC approximates the embedding space at a given input using a multivariate normal distribution bypassing the need for sampling via forward passes through the vision backbone. We demonstrate the effectiveness of OVC on through experimental evaluation using multiple vision-language backbones on the CIFAR-10 and ImageNet test datasets.



## **46. InstructTA: Instruction-Tuned Targeted Attack for Large Vision-Language Models**

cs.CV

**SubmitDate**: 2024-01-04    [abs](http://arxiv.org/abs/2312.01886v2) [paper-pdf](http://arxiv.org/pdf/2312.01886v2)

**Authors**: Xunguang Wang, Zhenlan Ji, Pingchuan Ma, Zongjie Li, Shuai Wang

**Abstract**: Large vision-language models (LVLMs) have demonstrated their incredible capability in image understanding and response generation. However, this rich visual interaction also makes LVLMs vulnerable to adversarial examples. In this paper, we formulate a novel and practical gray-box attack scenario that the adversary can only access the visual encoder of the victim LVLM, without the knowledge of its prompts (which are often proprietary for service providers and not publicly available) and its underlying large language model (LLM). This practical setting poses challenges to the cross-prompt and cross-model transferability of targeted adversarial attack, which aims to confuse the LVLM to output a response that is semantically similar to the attacker's chosen target text. To this end, we propose an instruction-tuned targeted attack (dubbed InstructTA) to deliver the targeted adversarial attack on LVLMs with high transferability. Initially, we utilize a public text-to-image generative model to "reverse" the target response into a target image, and employ GPT-4 to infer a reasonable instruction $\boldsymbol{p}^\prime$ from the target response. We then form a local surrogate model (sharing the same visual encoder with the victim LVLM) to extract instruction-aware features of an adversarial image example and the target image, and minimize the distance between these two features to optimize the adversarial example. To further improve the transferability, we augment the instruction $\boldsymbol{p}^\prime$ with instructions paraphrased from an LLM. Extensive experiments demonstrate the superiority of our proposed method in targeted attack performance and transferability.



## **47. DiffAttack: Evasion Attacks Against Diffusion-Based Adversarial Purification**

cs.CR

Accepted to NeurIPS 2023

**SubmitDate**: 2024-01-04    [abs](http://arxiv.org/abs/2311.16124v2) [paper-pdf](http://arxiv.org/pdf/2311.16124v2)

**Authors**: Mintong Kang, Dawn Song, Bo Li

**Abstract**: Diffusion-based purification defenses leverage diffusion models to remove crafted perturbations of adversarial examples and achieve state-of-the-art robustness. Recent studies show that even advanced attacks cannot break such defenses effectively, since the purification process induces an extremely deep computational graph which poses the potential problem of gradient obfuscation, high memory cost, and unbounded randomness. In this paper, we propose a unified framework DiffAttack to perform effective and efficient attacks against diffusion-based purification defenses, including both DDPM and score-based approaches. In particular, we propose a deviated-reconstruction loss at intermediate diffusion steps to induce inaccurate density gradient estimation to tackle the problem of vanishing/exploding gradients. We also provide a segment-wise forwarding-backwarding algorithm, which leads to memory-efficient gradient backpropagation. We validate the attack effectiveness of DiffAttack compared with existing adaptive attacks on CIFAR-10 and ImageNet. We show that DiffAttack decreases the robust accuracy of models compared with SOTA attacks by over 20% on CIFAR-10 under $\ell_\infty$ attack $(\epsilon=8/255)$, and over 10% on ImageNet under $\ell_\infty$ attack $(\epsilon=4/255)$. We conduct a series of ablations studies, and we find 1) DiffAttack with the deviated-reconstruction loss added over uniformly sampled time steps is more effective than that added over only initial/final steps, and 2) diffusion-based purification with a moderate diffusion length is more robust under DiffAttack.



## **48. CBD: A Certified Backdoor Detector Based on Local Dominant Probability**

cs.LG

Accepted to NeurIPS 2023

**SubmitDate**: 2024-01-04    [abs](http://arxiv.org/abs/2310.17498v2) [paper-pdf](http://arxiv.org/pdf/2310.17498v2)

**Authors**: Zhen Xiang, Zidi Xiong, Bo Li

**Abstract**: Backdoor attack is a common threat to deep neural networks. During testing, samples embedded with a backdoor trigger will be misclassified as an adversarial target by a backdoored model, while samples without the backdoor trigger will be correctly classified. In this paper, we present the first certified backdoor detector (CBD), which is based on a novel, adjustable conformal prediction scheme based on our proposed statistic local dominant probability. For any classifier under inspection, CBD provides 1) a detection inference, 2) the condition under which the attacks are guaranteed to be detectable for the same classification domain, and 3) a probabilistic upper bound for the false positive rate. Our theoretical results show that attacks with triggers that are more resilient to test-time noise and have smaller perturbation magnitudes are more likely to be detected with guarantees. Moreover, we conduct extensive experiments on four benchmark datasets considering various backdoor types, such as BadNet, CB, and Blend. CBD achieves comparable or even higher detection accuracy than state-of-the-art detectors, and it in addition provides detection certification. Notably, for backdoor attacks with random perturbation triggers bounded by $\ell_2\leq0.75$ which achieves more than 90\% attack success rate, CBD achieves 100\% (98\%), 100\% (84\%), 98\% (98\%), and 72\% (40\%) empirical (certified) detection true positive rates on the four benchmark datasets GTSRB, SVHN, CIFAR-10, and TinyImageNet, respectively, with low false positive rates.



## **49. DeepTaster: Adversarial Perturbation-Based Fingerprinting to Identify Proprietary Dataset Use in Deep Neural Networks**

cs.CR

**SubmitDate**: 2024-01-04    [abs](http://arxiv.org/abs/2211.13535v2) [paper-pdf](http://arxiv.org/pdf/2211.13535v2)

**Authors**: Seonhye Park, Alsharif Abuadbba, Shuo Wang, Kristen Moore, Yansong Gao, Hyoungshick Kim, Surya Nepal

**Abstract**: Training deep neural networks (DNNs) requires large datasets and powerful computing resources, which has led some owners to restrict redistribution without permission. Watermarking techniques that embed confidential data into DNNs have been used to protect ownership, but these can degrade model performance and are vulnerable to watermark removal attacks. Recently, DeepJudge was introduced as an alternative approach to measuring the similarity between a suspect and a victim model. While DeepJudge shows promise in addressing the shortcomings of watermarking, it primarily addresses situations where the suspect model copies the victim's architecture. In this study, we introduce DeepTaster, a novel DNN fingerprinting technique, to address scenarios where a victim's data is unlawfully used to build a suspect model. DeepTaster can effectively identify such DNN model theft attacks, even when the suspect model's architecture deviates from the victim's. To accomplish this, DeepTaster generates adversarial images with perturbations, transforms them into the Fourier frequency domain, and uses these transformed images to identify the dataset used in a suspect model. The underlying premise is that adversarial images can capture the unique characteristics of DNNs built with a specific dataset. To demonstrate the effectiveness of DeepTaster, we evaluated the effectiveness of DeepTaster by assessing its detection accuracy on three datasets (CIFAR10, MNIST, and Tiny-ImageNet) across three model architectures (ResNet18, VGG16, and DenseNet161). We conducted experiments under various attack scenarios, including transfer learning, pruning, fine-tuning, and data augmentation. Specifically, in the Multi-Architecture Attack scenario, DeepTaster was able to identify all the stolen cases across all datasets, while DeepJudge failed to detect any of the cases.



## **50. Integrated Cyber-Physical Resiliency for Power Grids under IoT-Enabled Dynamic Botnet Attacks**

eess.SY

**SubmitDate**: 2024-01-03    [abs](http://arxiv.org/abs/2401.01963v1) [paper-pdf](http://arxiv.org/pdf/2401.01963v1)

**Authors**: Yuhan Zhao, Juntao Chen, Quanyan Zhu

**Abstract**: The wide adoption of Internet of Things (IoT)-enabled energy devices improves the quality of life, but simultaneously, it enlarges the attack surface of the power grid system. The adversary can gain illegitimate control of a large number of these devices and use them as a means to compromise the physical grid operation, a mechanism known as the IoT botnet attack. This paper aims to improve the resiliency of cyber-physical power grids to such attacks. Specifically, we use an epidemic model to understand the dynamic botnet formation, which facilitates the assessment of the cyber layer vulnerability of the grid. The attacker aims to exploit this vulnerability to enable a successful physical compromise, while the system operator's goal is to ensure a normal operation of the grid by mitigating cyber risks. We develop a cross-layer game-theoretic framework for strategic decision-making to enhance cyber-physical grid resiliency. The cyber-layer game guides the system operator on how to defend against the botnet attacker as the first layer of defense, while the dynamic game strategy at the physical layer further counteracts the adversarial behavior in real time for improved physical resilience. A number of case studies on the IEEE-39 bus system are used to corroborate the devised approach.



