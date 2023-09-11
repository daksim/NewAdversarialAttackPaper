# Latest Adversarial Attack Papers
**update at 2023-09-11 14:41:28**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. Avoid Adversarial Adaption in Federated Learning by Multi-Metric Investigations**

cs.LG

25 pages, 14 figures, 23 tables, 11 equations

**SubmitDate**: 2023-09-08    [abs](http://arxiv.org/abs/2306.03600v2) [paper-pdf](http://arxiv.org/pdf/2306.03600v2)

**Authors**: Torsten Krauß, Alexandra Dmitrienko

**Abstract**: Federated Learning (FL) facilitates decentralized machine learning model training, preserving data privacy, lowering communication costs, and boosting model performance through diversified data sources. Yet, FL faces vulnerabilities such as poisoning attacks, undermining model integrity with both untargeted performance degradation and targeted backdoor attacks. Preventing backdoors proves especially challenging due to their stealthy nature.   Prominent mitigation techniques against poisoning attacks rely on monitoring certain metrics and filtering malicious model updates. While shown effective in evaluations, we argue that previous works didn't consider realistic real-world adversaries and data distributions. We define a new notion of strong adaptive adversaries, capable of adapting to multiple objectives simultaneously. Through extensive empirical tests, we show that existing defense methods can be easily circumvented in this adversary model. We also demonstrate, that existing defenses have limited effectiveness when no assumptions are made about underlying data distributions.   We introduce Metric-Cascades (MESAS), a novel defense method for more realistic scenarios and adversary models. MESAS employs multiple detection metrics simultaneously to identify poisoned model updates, creating a complex multi-objective optimization problem for adaptive attackers. In our extensive evaluation featuring nine backdoors and three datasets, MESAS consistently detects even strong adaptive attackers. Furthermore, MESAS outperforms existing defenses in distinguishing backdoors from data distribution-related distortions within and across clients. MESAS is the first defense robust against strong adaptive adversaries, effective in real-world data scenarios, with an average overhead of just 24.37 seconds.



## **2. Verifiable Learning for Robust Tree Ensembles**

cs.LG

19 pages, 5 figures; full version of the revised paper accepted at  ACM CCS 2023

**SubmitDate**: 2023-09-08    [abs](http://arxiv.org/abs/2305.03626v2) [paper-pdf](http://arxiv.org/pdf/2305.03626v2)

**Authors**: Stefano Calzavara, Lorenzo Cazzaro, Giulio Ermanno Pibiri, Nicola Prezza

**Abstract**: Verifying the robustness of machine learning models against evasion attacks at test time is an important research problem. Unfortunately, prior work established that this problem is NP-hard for decision tree ensembles, hence bound to be intractable for specific inputs. In this paper, we identify a restricted class of decision tree ensembles, called large-spread ensembles, which admit a security verification algorithm running in polynomial time. We then propose a new approach called verifiable learning, which advocates the training of such restricted model classes which are amenable for efficient verification. We show the benefits of this idea by designing a new training algorithm that automatically learns a large-spread decision tree ensemble from labelled data, thus enabling its security verification in polynomial time. Experimental results on public datasets confirm that large-spread ensembles trained using our algorithm can be verified in a matter of seconds, using standard commercial hardware. Moreover, large-spread ensembles are more robust than traditional ensembles against evasion attacks, at the cost of an acceptable loss of accuracy in the non-adversarial setting.



## **3. FIVA: Facial Image and Video Anonymization and Anonymization Defense**

cs.CV

Accepted to ICCVW 2023 - DFAD 2023

**SubmitDate**: 2023-09-08    [abs](http://arxiv.org/abs/2309.04228v1) [paper-pdf](http://arxiv.org/pdf/2309.04228v1)

**Authors**: Felix Rosberg, Eren Erdal Aksoy, Cristofer Englund, Fernando Alonso-Fernandez

**Abstract**: In this paper, we present a new approach for facial anonymization in images and videos, abbreviated as FIVA. Our proposed method is able to maintain the same face anonymization consistently over frames with our suggested identity-tracking and guarantees a strong difference from the original face. FIVA allows for 0 true positives for a false acceptance rate of 0.001. Our work considers the important security issue of reconstruction attacks and investigates adversarial noise, uniform noise, and parameter noise to disrupt reconstruction attacks. In this regard, we apply different defense and protection methods against these privacy threats to demonstrate the scalability of FIVA. On top of this, we also show that reconstruction attack models can be used for detection of deep fakes. Last but not least, we provide experimental results showing how FIVA can even enable face swapping, which is purely trained on a single target image.



## **4. Counterfactual Explanations via Locally-guided Sequential Algorithmic Recourse**

cs.LG

7 pages, 5 figures, 3 appendix pages

**SubmitDate**: 2023-09-08    [abs](http://arxiv.org/abs/2309.04211v1) [paper-pdf](http://arxiv.org/pdf/2309.04211v1)

**Authors**: Edward A. Small, Jeffrey N. Clark, Christopher J. McWilliams, Kacper Sokol, Jeffrey Chan, Flora D. Salim, Raul Santos-Rodriguez

**Abstract**: Counterfactuals operationalised through algorithmic recourse have become a powerful tool to make artificial intelligence systems explainable. Conceptually, given an individual classified as y -- the factual -- we seek actions such that their prediction becomes the desired class y' -- the counterfactual. This process offers algorithmic recourse that is (1) easy to customise and interpret, and (2) directly aligned with the goals of each individual. However, the properties of a "good" counterfactual are still largely debated; it remains an open challenge to effectively locate a counterfactual along with its corresponding recourse. Some strategies use gradient-driven methods, but these offer no guarantees on the feasibility of the recourse and are open to adversarial attacks on carefully created manifolds. This can lead to unfairness and lack of robustness. Other methods are data-driven, which mostly addresses the feasibility problem at the expense of privacy, security and secrecy as they require access to the entire training data set. Here, we introduce LocalFACE, a model-agnostic technique that composes feasible and actionable counterfactual explanations using locally-acquired information at each step of the algorithmic recourse. Our explainer preserves the privacy of users by only leveraging data that it specifically requires to construct actionable algorithmic recourse, and protects the model by offering transparency solely in the regions deemed necessary for the intervention.



## **5. Blades: A Unified Benchmark Suite for Byzantine Attacks and Defenses in Federated Learning**

cs.CR

**SubmitDate**: 2023-09-07    [abs](http://arxiv.org/abs/2206.05359v2) [paper-pdf](http://arxiv.org/pdf/2206.05359v2)

**Authors**: Shenghui Li, Edith Ngai, Fanghua Ye, Li Ju, Tianru Zhang, Thiemo Voigt

**Abstract**: Federated learning (FL) facilitates distributed training across clients, safeguarding the privacy of their data. The inherent distributed structure of FL introduces vulnerabilities, especially from adversarial (Byzantine) clients aiming to skew local updates to their advantage. Despite the plethora of research focusing on Byzantine-resilient FL, the academic community has yet to establish a comprehensive benchmark suite, pivotal for impartial assessment and comparison of different techniques.   This paper investigates existing techniques in Byzantine-resilient FL and introduces an open-source benchmark suite for convenient and fair performance comparisons. Our investigation begins with a systematic study of Byzantine attack and defense strategies. Subsequently, we present \ours, a scalable, extensible, and easily configurable benchmark suite that supports researchers and developers in efficiently implementing and validating novel strategies against baseline algorithms in Byzantine-resilient FL. The design of \ours incorporates key characteristics derived from our systematic study, encompassing the attacker's capabilities and knowledge, defense strategy categories, and factors influencing robustness. Blades contains built-in implementations of representative attack and defense strategies and offers user-friendly interfaces for seamlessly integrating new ideas.



## **6. Node Injection for Class-specific Network Poisoning**

cs.LG

28 pages, 5 figures

**SubmitDate**: 2023-09-07    [abs](http://arxiv.org/abs/2301.12277v2) [paper-pdf](http://arxiv.org/pdf/2301.12277v2)

**Authors**: Ansh Kumar Sharma, Rahul Kukreja, Mayank Kharbanda, Tanmoy Chakraborty

**Abstract**: Graph Neural Networks (GNNs) are powerful in learning rich network representations that aid the performance of downstream tasks. However, recent studies showed that GNNs are vulnerable to adversarial attacks involving node injection and network perturbation. Among these, node injection attacks are more practical as they don't require manipulation in the existing network and can be performed more realistically. In this paper, we propose a novel problem statement - a class-specific poison attack on graphs in which the attacker aims to misclassify specific nodes in the target class into a different class using node injection. Additionally, nodes are injected in such a way that they camouflage as benign nodes. We propose NICKI, a novel attacking strategy that utilizes an optimization-based approach to sabotage the performance of GNN-based node classifiers. NICKI works in two phases - it first learns the node representation and then generates the features and edges of the injected nodes. Extensive experiments and ablation studies on four benchmark networks show that NICKI is consistently better than four baseline attacking strategies for misclassifying nodes in the target class. We also show that the injected nodes are properly camouflaged as benign, thus making the poisoned graph indistinguishable from its clean version w.r.t various topological properties.



## **7. Experimental Study of Adversarial Attacks on ML-based xApps in O-RAN**

cs.NI

Accepted for Globecom 2023

**SubmitDate**: 2023-09-07    [abs](http://arxiv.org/abs/2309.03844v1) [paper-pdf](http://arxiv.org/pdf/2309.03844v1)

**Authors**: Naveen Naik Sapavath, Brian Kim, Kaushik Chowdhury, Vijay K Shah

**Abstract**: Open Radio Access Network (O-RAN) is considered as a major step in the evolution of next-generation cellular networks given its support for open interfaces and utilization of artificial intelligence (AI) into the deployment, operation, and maintenance of RAN. However, due to the openness of the O-RAN architecture, such AI models are inherently vulnerable to various adversarial machine learning (ML) attacks, i.e., adversarial attacks which correspond to slight manipulation of the input to the ML model. In this work, we showcase the vulnerability of an example ML model used in O-RAN, and experimentally deploy it in the near-real time (near-RT) RAN intelligent controller (RIC). Our ML-based interference classifier xApp (extensible application in near-RT RIC) tries to classify the type of interference to mitigate the interference effect on the O-RAN system. We demonstrate the first-ever scenario of how such an xApp can be impacted through an adversarial attack by manipulating the data stored in a shared database inside the near-RT RIC. Through a rigorous performance analysis deployed on a laboratory O-RAN testbed, we evaluate the performance in terms of capacity and the prediction accuracy of the interference classifier xApp using both clean and perturbed data. We show that even small adversarial attacks can significantly decrease the accuracy of ML application in near-RT RIC, which can directly impact the performance of the entire O-RAN deployment.



## **8. Adversarially Robust Deep Learning with Optimal-Transport-Regularized Divergences**

cs.LG

30 pages, 5 figures

**SubmitDate**: 2023-09-07    [abs](http://arxiv.org/abs/2309.03791v1) [paper-pdf](http://arxiv.org/pdf/2309.03791v1)

**Authors**: Jeremiah Birrell, Mohammadreza Ebrahimi

**Abstract**: We introduce the $ARMOR_D$ methods as novel approaches to enhancing the adversarial robustness of deep learning models. These methods are based on a new class of optimal-transport-regularized divergences, constructed via an infimal convolution between an information divergence and an optimal-transport (OT) cost. We use these as tools to enhance adversarial robustness by maximizing the expected loss over a neighborhood of distributions, a technique known as distributionally robust optimization. Viewed as a tool for constructing adversarial samples, our method allows samples to be both transported, according to the OT cost, and re-weighted, according to the information divergence. We demonstrate the effectiveness of our method on malware detection and image recognition applications and find that, to our knowledge, it outperforms existing methods at enhancing the robustness against adversarial attacks. $ARMOR_D$ yields the robustified accuracy of $98.29\%$ against $FGSM$ and $98.18\%$ against $PGD^{40}$ on the MNIST dataset, reducing the error rate by more than $19.7\%$ and $37.2\%$ respectively compared to prior methods. Similarly, in malware detection, a discrete (binary) data domain, $ARMOR_D$ improves the robustified accuracy under $rFGSM^{50}$ attack compared to the previous best-performing adversarial training methods by $37.0\%$ while lowering false negative and false positive rates by $51.1\%$ and $57.53\%$, respectively.



## **9. DiffDefense: Defending against Adversarial Attacks via Diffusion Models**

cs.LG

Paper published at ICIAP23

**SubmitDate**: 2023-09-07    [abs](http://arxiv.org/abs/2309.03702v1) [paper-pdf](http://arxiv.org/pdf/2309.03702v1)

**Authors**: Hondamunige Prasanna Silva, Lorenzo Seidenari, Alberto Del Bimbo

**Abstract**: This paper presents a novel reconstruction method that leverages Diffusion Models to protect machine learning classifiers against adversarial attacks, all without requiring any modifications to the classifiers themselves. The susceptibility of machine learning models to minor input perturbations renders them vulnerable to adversarial attacks. While diffusion-based methods are typically disregarded for adversarial defense due to their slow reverse process, this paper demonstrates that our proposed method offers robustness against adversarial threats while preserving clean accuracy, speed, and plug-and-play compatibility. Code at: https://github.com/HondamunigePrasannaSilva/DiffDefence.



## **10. Improving Visual Quality and Transferability of Adversarial Attacks on Face Recognition Simultaneously with Adversarial Restoration**

cs.CV

**SubmitDate**: 2023-09-07    [abs](http://arxiv.org/abs/2309.01582v2) [paper-pdf](http://arxiv.org/pdf/2309.01582v2)

**Authors**: Fengfan Zhou, Hefei Ling, Yuxuan Shi, Jiazhong Chen, Ping Li

**Abstract**: Adversarial face examples possess two critical properties: Visual Quality and Transferability. However, existing approaches rarely address these properties simultaneously, leading to subpar results. To address this issue, we propose a novel adversarial attack technique known as Adversarial Restoration (AdvRestore), which enhances both visual quality and transferability of adversarial face examples by leveraging a face restoration prior. In our approach, we initially train a Restoration Latent Diffusion Model (RLDM) designed for face restoration. Subsequently, we employ the inference process of RLDM to generate adversarial face examples. The adversarial perturbations are applied to the intermediate features of RLDM. Additionally, by treating RLDM face restoration as a sibling task, the transferability of the generated adversarial face examples is further improved. Our experimental results validate the effectiveness of the proposed attack method.



## **11. Impact Sensitivity Analysis of Cooperative Adaptive Cruise Control Against Resource-Limited Adversaries**

eess.SY

**SubmitDate**: 2023-09-07    [abs](http://arxiv.org/abs/2304.02395v2) [paper-pdf](http://arxiv.org/pdf/2304.02395v2)

**Authors**: Mischa Huisman, Carlos Murguia, Erjen Lefeber, Nathan van de Wouw

**Abstract**: Cooperative Adaptive Cruise Control (CACC) is a technology that allows groups of vehicles to form in automated, tightly-coupled platoons. CACC schemes exploit Vehicle-to-Vehicle (V2V) wireless communications to exchange information between vehicles. However, the use of communication networks brings security concerns as it exposes network access points that the adversary can exploit to disrupt the vehicles' operation and even cause crashes. In this manuscript, we present a sensitivity analysis of CACC schemes against a class of resource-limited attacks. We present a modelling framework that allows us to systematically compute outer ellipsoidal approximations of reachable sets induced by attacks. We use the size of these sets as a security metric to quantify the potential damage of attacks affecting different signals in a CACC-controlled vehicle and study how two key system parameters change this metric. We carry out a sensitivity analysis for two different controller implementations (as given the available sensors there is an infinite number of realizations of the same controller) and show how different controller realizations can significantly affect the impact of attacks. We present extensive simulation experiments to illustrate the results.



## **12. How adversarial attacks can disrupt seemingly stable accurate classifiers**

cs.LG

11 pages, 8 figures, additional supplementary materials

**SubmitDate**: 2023-09-07    [abs](http://arxiv.org/abs/2309.03665v1) [paper-pdf](http://arxiv.org/pdf/2309.03665v1)

**Authors**: Oliver J. Sutton, Qinghua Zhou, Ivan Y. Tyukin, Alexander N. Gorban, Alexander Bastounis, Desmond J. Higham

**Abstract**: Adversarial attacks dramatically change the output of an otherwise accurate learning system using a seemingly inconsequential modification to a piece of input data. Paradoxically, empirical evidence indicates that even systems which are robust to large random perturbations of the input data remain susceptible to small, easily constructed, adversarial perturbations of their inputs. Here, we show that this may be seen as a fundamental feature of classifiers working with high dimensional input data. We introduce a simple generic and generalisable framework for which key behaviours observed in practical systems arise with high probability -- notably the simultaneous susceptibility of the (otherwise accurate) model to easily constructed adversarial attacks, and robustness to random perturbations of the input data. We confirm that the same phenomena are directly observed in practical neural networks trained on standard image classification problems, where even large additive random noise fails to trigger the adversarial instability of the network. A surprising takeaway is that even small margins separating a classifier's decision surface from training and testing data can hide adversarial susceptibility from being detected using randomly sampled perturbations. Counterintuitively, using additive noise during training or testing is therefore inefficient for eradicating or detecting adversarial examples, and more demanding adversarial training is required.



## **13. Unlearnable Examples Give a False Sense of Security: Piercing through Unexploitable Data with Learnable Examples**

cs.LG

Accepted in MM 2023

**SubmitDate**: 2023-09-07    [abs](http://arxiv.org/abs/2305.09241v4) [paper-pdf](http://arxiv.org/pdf/2305.09241v4)

**Authors**: Wan Jiang, Yunfeng Diao, He Wang, Jianxin Sun, Meng Wang, Richang Hong

**Abstract**: Safeguarding data from unauthorized exploitation is vital for privacy and security, especially in recent rampant research in security breach such as adversarial/membership attacks. To this end, \textit{unlearnable examples} (UEs) have been recently proposed as a compelling protection, by adding imperceptible perturbation to data so that models trained on them cannot classify them accurately on original clean distribution. Unfortunately, we find UEs provide a false sense of security, because they cannot stop unauthorized users from utilizing other unprotected data to remove the protection, by turning unlearnable data into learnable again. Motivated by this observation, we formally define a new threat by introducing \textit{learnable unauthorized examples} (LEs) which are UEs with their protection removed. The core of this approach is a novel purification process that projects UEs onto the manifold of LEs. This is realized by a new joint-conditional diffusion model which denoises UEs conditioned on the pixel and perceptual similarity between UEs and LEs. Extensive experiments demonstrate that LE delivers state-of-the-art countering performance against both supervised UEs and unsupervised UEs in various scenarios, which is the first generalizable countermeasure to UEs across supervised learning and unsupervised learning. Our code is available at \url{https://github.com/jiangw-0/LE_JCDP}.



## **14. Password-Stealing without Hacking: Wi-Fi Enabled Practical Keystroke Eavesdropping**

cs.CR

14 pages, 27 figures, in ACM Conference on Computer and  Communications Security 2023

**SubmitDate**: 2023-09-07    [abs](http://arxiv.org/abs/2309.03492v1) [paper-pdf](http://arxiv.org/pdf/2309.03492v1)

**Authors**: Jingyang Hu, Hongbo Wang, Tianyue Zheng, Jingzhi Hu, Zhe Chen, Hongbo Jiang, Jun Luo

**Abstract**: The contact-free sensing nature of Wi-Fi has been leveraged to achieve privacy breaches, yet existing attacks relying on Wi-Fi CSI (channel state information) demand hacking Wi-Fi hardware to obtain desired CSIs. Since such hacking has proven prohibitively hard due to compact hardware, its feasibility in keeping up with fast-developing Wi-Fi technology becomes very questionable. To this end, we propose WiKI-Eve to eavesdrop keystrokes on smartphones without the need for hacking. WiKI-Eve exploits a new feature, BFI (beamforming feedback information), offered by latest Wi-Fi hardware: since BFI is transmitted from a smartphone to an AP in clear-text, it can be overheard (hence eavesdropped) by any other Wi-Fi devices switching to monitor mode. As existing keystroke inference methods offer very limited generalizability, WiKI-Eve further innovates in an adversarial learning scheme to enable its inference generalizable towards unseen scenarios. We implement WiKI-Eve and conduct extensive evaluation on it; the results demonstrate that WiKI-Eve achieves 88.9% inference accuracy for individual keystrokes and up to 65.8% top-10 accuracy for stealing passwords of mobile applications (e.g., WeChat).



## **15. Cyber Recovery from Dynamic Load Altering Attacks: Linking Electricity, Transportation, and Cyber Networks**

eess.SY

**SubmitDate**: 2023-09-06    [abs](http://arxiv.org/abs/2309.03380v1) [paper-pdf](http://arxiv.org/pdf/2309.03380v1)

**Authors**: Mengxiang Liu, Zhongda Chu, Fei Teng

**Abstract**: To address the increasing vulnerability of power grids, significant attention has been focused on the attack detection and impact mitigation. However, it is still unclear how to effectively and quickly recover the cyber and physical networks from a cyberattack. In this context, this paper presents the first investigation of the Cyber Recovery from Dynamic load altering Attack (CRDA). Considering the interconnection among electricity, transportation, and cyber networks, two essential sub-tasks are formulated for the CRDA: i) Optimal design of repair crew routes to remove installed malware and ii) Adaptive adjustment of system operation to eliminate the mitigation costs while guaranteeing stability. To achieve this, linear stability constraints are obtained by estimating the related eigenvalues under the variation of multiple IBR droop gains based on the sensitivity information of strategically selected sampling points. Moreover, to obtain the robust recovery strategy, the potential counter-measures from the adversary during the recovery process are modeled as maximizing the attack impact of remaining compromised resources in each step. A Mixed-Integer Linear Programming (MILP) problem can be finally formulated for the CRDA with the primary objective to reset involved droop gains and secondarily to repair all compromised loads. Case studies are performed in the modified IEEE 39-bus power system to illustrate the effectiveness of the proposed CRDA compared to the benchmark case.



## **16. The Space of Adversarial Strategies**

cs.CR

Accepted to the 32nd USENIX Security Symposium

**SubmitDate**: 2023-09-06    [abs](http://arxiv.org/abs/2209.04521v2) [paper-pdf](http://arxiv.org/pdf/2209.04521v2)

**Authors**: Ryan Sheatsley, Blaine Hoak, Eric Pauley, Patrick McDaniel

**Abstract**: Adversarial examples, inputs designed to induce worst-case behavior in machine learning models, have been extensively studied over the past decade. Yet, our understanding of this phenomenon stems from a rather fragmented pool of knowledge; at present, there are a handful of attacks, each with disparate assumptions in threat models and incomparable definitions of optimality. In this paper, we propose a systematic approach to characterize worst-case (i.e., optimal) adversaries. We first introduce an extensible decomposition of attacks in adversarial machine learning by atomizing attack components into surfaces and travelers. With our decomposition, we enumerate over components to create 576 attacks (568 of which were previously unexplored). Next, we propose the Pareto Ensemble Attack (PEA): a theoretical attack that upper-bounds attack performance. With our new attacks, we measure performance relative to the PEA on: both robust and non-robust models, seven datasets, and three extended lp-based threat models incorporating compute costs, formalizing the Space of Adversarial Strategies. From our evaluation we find that attack performance to be highly contextual: the domain, model robustness, and threat model can have a profound influence on attack efficacy. Our investigation suggests that future studies measuring the security of machine learning should: (1) be contextualized to the domain & threat models, and (2) go beyond the handful of known attacks used today.



## **17. My Art My Choice: Adversarial Protection Against Unruly AI**

cs.CV

**SubmitDate**: 2023-09-06    [abs](http://arxiv.org/abs/2309.03198v1) [paper-pdf](http://arxiv.org/pdf/2309.03198v1)

**Authors**: Anthony Rhodes, Ram Bhagat, Umur Aybars Ciftci, Ilke Demir

**Abstract**: Generative AI is on the rise, enabling everyone to produce realistic content via publicly available interfaces. Especially for guided image generation, diffusion models are changing the creator economy by producing high quality low cost content. In parallel, artists are rising against unruly AI, since their artwork are leveraged, distributed, and dissimulated by large generative models. Our approach, My Art My Choice (MAMC), aims to empower content owners by protecting their copyrighted materials from being utilized by diffusion models in an adversarial fashion. MAMC learns to generate adversarially perturbed "protected" versions of images which can in turn "break" diffusion models. The perturbation amount is decided by the artist to balance distortion vs. protection of the content. MAMC is designed with a simple UNet-based generator, attacking black box diffusion models, combining several losses to create adversarial twins of the original artwork. We experiment on three datasets for various image-to-image tasks, with different user control values. Both protected image and diffusion output results are evaluated in visual, noise, structure, pixel, and generative spaces to validate our claims. We believe that MAMC is a crucial step for preserving ownership information for AI generated content in a flawless, based-on-need, and human-centric way.



## **18. J-Guard: Journalism Guided Adversarially Robust Detection of AI-generated News**

cs.CL

This Paper is Accepted to The 13th International Joint Conference on  Natural Language Processing and the 3rd Conference of the Asia-Pacific  Chapter of the Association for Computational Linguistics (IJCNLP-AACL 2023)

**SubmitDate**: 2023-09-06    [abs](http://arxiv.org/abs/2309.03164v1) [paper-pdf](http://arxiv.org/pdf/2309.03164v1)

**Authors**: Tharindu Kumarage, Amrita Bhattacharjee, Djordje Padejski, Kristy Roschke, Dan Gillmor, Scott Ruston, Huan Liu, Joshua Garland

**Abstract**: The rapid proliferation of AI-generated text online is profoundly reshaping the information landscape. Among various types of AI-generated text, AI-generated news presents a significant threat as it can be a prominent source of misinformation online. While several recent efforts have focused on detecting AI-generated text in general, these methods require enhanced reliability, given concerns about their vulnerability to simple adversarial attacks. Furthermore, due to the eccentricities of news writing, applying these detection methods for AI-generated news can produce false positives, potentially damaging the reputation of news organizations. To address these challenges, we leverage the expertise of an interdisciplinary team to develop a framework, J-Guard, capable of steering existing supervised AI text detectors for detecting AI-generated news while boosting adversarial robustness. By incorporating stylistic cues inspired by the unique journalistic attributes, J-Guard effectively distinguishes between real-world journalism and AI-generated news articles. Our experiments on news articles generated by a vast array of AI models, including ChatGPT (GPT3.5), demonstrate the effectiveness of J-Guard in enhancing detection capabilities while maintaining an average performance decrease of as low as 7% when faced with adversarial attacks.



## **19. Defense-Prefix for Preventing Typographic Attacks on CLIP**

cs.CV

ICCV2023 Workshop

**SubmitDate**: 2023-09-06    [abs](http://arxiv.org/abs/2304.04512v3) [paper-pdf](http://arxiv.org/pdf/2304.04512v3)

**Authors**: Hiroki Azuma, Yusuke Matsui

**Abstract**: Vision-language pre-training models (VLPs) have exhibited revolutionary improvements in various vision-language tasks. In VLP, some adversarial attacks fool a model into false or absurd classifications. Previous studies addressed these attacks by fine-tuning the model or changing its architecture. However, these methods risk losing the original model's performance and are difficult to apply to downstream tasks. In particular, their applicability to other tasks has not been considered. In this study, we addressed the reduction of the impact of typographic attacks on CLIP without changing the model parameters. To achieve this, we expand the idea of "prefix learning" and introduce our simple yet effective method: Defense-Prefix (DP), which inserts the DP token before a class name to make words "robust" against typographic attacks. Our method can be easily applied to downstream tasks, such as object detection, because the proposed method is independent of the model parameters. Our method significantly improves the accuracy of classification tasks for typographic attack datasets, while maintaining the zero-shot capabilities of the model. In addition, we leverage our proposed method for object detection, demonstrating its high applicability and effectiveness. The codes and datasets are available at https://github.com/azuma164/Defense-Prefix.



## **20. Enhancing Adversarial Attacks: The Similar Target Method**

cs.CV

**SubmitDate**: 2023-09-06    [abs](http://arxiv.org/abs/2308.10743v2) [paper-pdf](http://arxiv.org/pdf/2308.10743v2)

**Authors**: Shuo Zhang, Ziruo Wang, Zikai Zhou, Huanran Chen

**Abstract**: Deep neural networks are vulnerable to adversarial examples, posing a threat to the models' applications and raising security concerns. An intriguing property of adversarial examples is their strong transferability. Several methods have been proposed to enhance transferability, including ensemble attacks which have demonstrated their efficacy. However, prior approaches simply average logits, probabilities, or losses for model ensembling, lacking a comprehensive analysis of how and why model ensembling significantly improves transferability. In this paper, we propose a similar targeted attack method named Similar Target~(ST). By promoting cosine similarity between the gradients of each model, our method regularizes the optimization direction to simultaneously attack all surrogate models. This strategy has been proven to enhance generalization ability. Experimental results on ImageNet validate the effectiveness of our approach in improving adversarial transferability. Our method outperforms state-of-the-art attackers on 18 discriminative classifiers and adversarially trained models.



## **21. Efficient Query-Based Attack against ML-Based Android Malware Detection under Zero Knowledge Setting**

cs.CR

To Appear in the ACM Conference on Computer and Communications  Security, November, 2023

**SubmitDate**: 2023-09-06    [abs](http://arxiv.org/abs/2309.01866v2) [paper-pdf](http://arxiv.org/pdf/2309.01866v2)

**Authors**: Ping He, Yifan Xia, Xuhong Zhang, Shouling Ji

**Abstract**: The widespread adoption of the Android operating system has made malicious Android applications an appealing target for attackers. Machine learning-based (ML-based) Android malware detection (AMD) methods are crucial in addressing this problem; however, their vulnerability to adversarial examples raises concerns. Current attacks against ML-based AMD methods demonstrate remarkable performance but rely on strong assumptions that may not be realistic in real-world scenarios, e.g., the knowledge requirements about feature space, model parameters, and training dataset. To address this limitation, we introduce AdvDroidZero, an efficient query-based attack framework against ML-based AMD methods that operates under the zero knowledge setting. Our extensive evaluation shows that AdvDroidZero is effective against various mainstream ML-based AMD methods, in particular, state-of-the-art such methods and real-world antivirus solutions.



## **22. SWAP: Exploiting Second-Ranked Logits for Adversarial Attacks on Time Series**

cs.LG

10 pages, 8 figures

**SubmitDate**: 2023-09-06    [abs](http://arxiv.org/abs/2309.02752v1) [paper-pdf](http://arxiv.org/pdf/2309.02752v1)

**Authors**: Chang George Dong, Liangwei Nathan Zheng, Weitong Chen, Wei Emma Zhang, Lin Yue

**Abstract**: Time series classification (TSC) has emerged as a critical task in various domains, and deep neural models have shown superior performance in TSC tasks. However, these models are vulnerable to adversarial attacks, where subtle perturbations can significantly impact the prediction results. Existing adversarial methods often suffer from over-parameterization or random logit perturbation, hindering their effectiveness. Additionally, increasing the attack success rate (ASR) typically involves generating more noise, making the attack more easily detectable. To address these limitations, we propose SWAP, a novel attacking method for TSC models. SWAP focuses on enhancing the confidence of the second-ranked logits while minimizing the manipulation of other logits. This is achieved by minimizing the Kullback-Leibler divergence between the target logit distribution and the predictive logit distribution. Experimental results demonstrate that SWAP achieves state-of-the-art performance, with an ASR exceeding 50% and an 18% increase compared to existing methods.



## **23. Certifying LLM Safety against Adversarial Prompting**

cs.CL

**SubmitDate**: 2023-09-06    [abs](http://arxiv.org/abs/2309.02705v1) [paper-pdf](http://arxiv.org/pdf/2309.02705v1)

**Authors**: Aounon Kumar, Chirag Agarwal, Suraj Srinivas, Soheil Feizi, Hima Lakkaraju

**Abstract**: Large language models (LLMs) released for public use incorporate guardrails to ensure their output is safe, often referred to as "model alignment." An aligned language model should decline a user's request to produce harmful content. However, such safety measures are vulnerable to adversarial prompts, which contain maliciously designed token sequences to circumvent the model's safety guards and cause it to produce harmful content. In this work, we introduce erase-and-check, the first framework to defend against adversarial prompts with verifiable safety guarantees. We erase tokens individually and inspect the resulting subsequences using a safety filter. Our procedure labels the input prompt as harmful if any subsequences or the input prompt are detected as harmful by the filter. This guarantees that any adversarial modification of a harmful prompt up to a certain size is also labeled harmful. We defend against three attack modes: i) adversarial suffix, which appends an adversarial sequence at the end of the prompt; ii) adversarial insertion, where the adversarial sequence is inserted anywhere in the middle of the prompt; and iii) adversarial infusion, where adversarial tokens are inserted at arbitrary positions in the prompt, not necessarily as a contiguous block. Empirical results demonstrate that our technique obtains strong certified safety guarantees on harmful prompts while maintaining good performance on safe prompts. For example, against adversarial suffixes of length 20, it certifiably detects 93% of the harmful prompts and labels 94% of the safe prompts as safe using the open source language model Llama 2 as the safety filter.



## **24. Experimental quantum key distribution certified by Bell's theorem**

quant-ph

5+1 pages in main text and methods with 4 figures and 1 table; 42  pages of supplementary material (replaced with revision accepted for  publication in Nature; original title: "Device-Independent Quantum Key  Distribution")

**SubmitDate**: 2023-09-05    [abs](http://arxiv.org/abs/2109.14600v2) [paper-pdf](http://arxiv.org/pdf/2109.14600v2)

**Authors**: D. P. Nadlinger, P. Drmota, B. C. Nichol, G. Araneda, D. Main, R. Srinivas, D. M. Lucas, C. J. Ballance, K. Ivanov, E. Y-Z. Tan, P. Sekatski, R. L. Urbanke, R. Renner, N. Sangouard, J-D. Bancal

**Abstract**: Cryptographic key exchange protocols traditionally rely on computational conjectures such as the hardness of prime factorisation to provide security against eavesdropping attacks. Remarkably, quantum key distribution protocols like the one proposed by Bennett and Brassard provide information-theoretic security against such attacks, a much stronger form of security unreachable by classical means. However, quantum protocols realised so far are subject to a new class of attacks exploiting implementation defects in the physical devices involved, as demonstrated in numerous ingenious experiments. Following the pioneering work of Ekert proposing the use of entanglement to bound an adversary's information from Bell's theorem, we present here the experimental realisation of a complete quantum key distribution protocol immune to these vulnerabilities. We achieve this by combining theoretical developments on finite-statistics analysis, error correction, and privacy amplification, with an event-ready scheme enabling the rapid generation of high-fidelity entanglement between two trapped-ion qubits connected by an optical fibre link. The secrecy of our key is guaranteed device-independently: it is based on the validity of quantum theory, and certified by measurement statistics observed during the experiment. Our result shows that provably secure cryptography with real-world devices is possible, and paves the way for further quantum information applications based on the device-independence principle.



## **25. Revisiting Adversarial Attacks on Graph Neural Networks for Graph Classification**

cs.SI

13 pages, 7 figures

**SubmitDate**: 2023-09-05    [abs](http://arxiv.org/abs/2208.06651v2) [paper-pdf](http://arxiv.org/pdf/2208.06651v2)

**Authors**: Xin Wang, Heng Chang, Beini Xie, Tian Bian, Shiji Zhou, Daixin Wang, Zhiqiang Zhang, Wenwu Zhu

**Abstract**: Graph neural networks (GNNs) have achieved tremendous success in the task of graph classification and its diverse downstream real-world applications. Despite the huge success in learning graph representations, current GNN models have demonstrated their vulnerability to potentially existent adversarial examples on graph-structured data. Existing approaches are either limited to structure attacks or restricted to local information, urging for the design of a more general attack framework on graph classification, which faces significant challenges due to the complexity of generating local-node-level adversarial examples using the global-graph-level information. To address this "global-to-local" attack challenge, we present a novel and general framework to generate adversarial examples via manipulating graph structure and node features. Specifically, we make use of Graph Class Activation Mapping and its variant to produce node-level importance corresponding to the graph classification task. Then through a heuristic design of algorithms, we can perform both feature and structure attacks under unnoticeable perturbation budgets with the help of both node-level and subgraph-level importance. Experiments towards attacking four state-of-the-art graph classification models on six real-world benchmarks verify the flexibility and effectiveness of our framework.



## **26. E-DPNCT: An Enhanced Attack Resilient Differential Privacy Model For Smart Grids Using Split Noise Cancellation**

cs.CR

13 pages, 8 figues, 1 tables

**SubmitDate**: 2023-09-05    [abs](http://arxiv.org/abs/2110.11091v4) [paper-pdf](http://arxiv.org/pdf/2110.11091v4)

**Authors**: Khadija Hafeez, Donna OShea, Thomas Newe, Mubashir Husain Rehmani

**Abstract**: High frequency reporting of energy consumption data in smart grids can be used to infer sensitive information regarding the consumer's life style and poses serious security and privacy threats. Differential privacy (DP) based privacy models for smart grids ensure privacy when analysing energy consumption data for billing and load monitoring. However, DP models for smart grids are vulnerable to collusion attack where an adversary colludes with malicious smart meters and un-trusted aggregator in order to get private information from other smart meters. We first show the vulnerability of DP based privacy model for smart grids against collusion attacks to establish the need of a collusion resistant model privacy model. Then, we propose an Enhanced Differential Private Noise Cancellation Model for Load Monitoring and Billing for Smart Meters (E-DPNCT) which not only provides resistance against collusion attacks but also protects the privacy of the smart grid data while providing accurate billing and load monitoring. We use differential privacy with a split noise cancellation protocol with multiple master smart meters (MSMs) to achieve colluison resistance. We did extensive comparison of our E-DPNCT model with state of the art attack resistant privacy preserving models such as EPIC for collusion attack. We simulate our E-DPNCT model with real time data which shows significant improvement in privacy attack scenarios. Further, we analyze the impact of selecting different sensitivity parameters for calibrating DP noise over the privacy of customer electricity profile and accuracy of electricity data aggregation such as load monitoring and billing.



## **27. Probabilistic and Semantic Descriptions of Image Manifolds and Their Applications**

cs.CV

24 pages, 17 figures, 1 table

**SubmitDate**: 2023-09-05    [abs](http://arxiv.org/abs/2307.02881v3) [paper-pdf](http://arxiv.org/pdf/2307.02881v3)

**Authors**: Peter Tu, Zhaoyuan Yang, Richard Hartley, Zhiwei Xu, Jing Zhang, Yiwei Fu, Dylan Campbell, Jaskirat Singh, Tianyu Wang

**Abstract**: This paper begins with a description of methods for estimating probability density functions for images that reflects the observation that such data is usually constrained to lie in restricted regions of the high-dimensional image space - not every pattern of pixels is an image. It is common to say that images lie on a lower-dimensional manifold in the high-dimensional space. However, although images may lie on such lower-dimensional manifolds, it is not the case that all points on the manifold have an equal probability of being images. Images are unevenly distributed on the manifold, and our task is to devise ways to model this distribution as a probability distribution. In pursuing this goal, we consider generative models that are popular in AI and computer vision community. For our purposes, generative/probabilistic models should have the properties of 1) sample generation: it should be possible to sample from this distribution according to the modelled density function, and 2) probability computation: given a previously unseen sample from the dataset of interest, one should be able to compute the probability of the sample, at least up to a normalising constant. To this end, we investigate the use of methods such as normalising flow and diffusion models. We then show that such probabilistic descriptions can be used to construct defences against adversarial attacks. In addition to describing the manifold in terms of density, we also consider how semantic interpretations can be used to describe points on the manifold. To this end, we consider an emergent language framework which makes use of variational encoders to produce a disentangled representation of points that reside on a given manifold. Trajectories between points on a manifold can then be described in terms of evolving semantic descriptions.



## **28. Mayhem: Targeted Corruption of Register and Stack Variables**

cs.CR

**SubmitDate**: 2023-09-05    [abs](http://arxiv.org/abs/2309.02545v1) [paper-pdf](http://arxiv.org/pdf/2309.02545v1)

**Authors**: Andrew J. Adiletta, M. Caner Tol, Yarkın Doröz, Berk Sunar

**Abstract**: In the past decade, many vulnerabilities were discovered in microarchitectures which yielded attack vectors and motivated the study of countermeasures. Further, architectural and physical imperfections in DRAMs led to the discovery of Rowhammer attacks which give an adversary power to introduce bit flips in a victim's memory space. Numerous studies analyzed Rowhammer and proposed techniques to prevent it altogether or to mitigate its effects.   In this work, we push the boundary and show how Rowhammer can be further exploited to inject faults into stack variables and even register values in a victim's process. We achieve this by targeting the register value that is stored in the process's stack, which subsequently is flushed out into the memory, where it becomes vulnerable to Rowhammer. When the faulty value is restored into the register, it will end up used in subsequent iterations. The register value can be stored in the stack via latent function calls in the source or by actively triggering signal handlers. We demonstrate the power of the findings by applying the techniques to bypass SUDO and SSH authentication. We further outline how MySQL and other cryptographic libraries can be targeted with the new attack vector. There are a number of challenges this work overcomes with extensive experimentation before coming together to yield an end-to-end attack on an OpenSSL digital signature: achieving co-location with stack and register variables, with synchronization provided via a blocking window. We show that stack and registers are no longer safe from the Rowhammer attack.



## **29. Adaptive Adversarial Training Does Not Increase Recourse Costs**

cs.LG

**SubmitDate**: 2023-09-05    [abs](http://arxiv.org/abs/2309.02528v1) [paper-pdf](http://arxiv.org/pdf/2309.02528v1)

**Authors**: Ian Hardy, Jayanth Yetukuri, Yang Liu

**Abstract**: Recent work has connected adversarial attack methods and algorithmic recourse methods: both seek minimal changes to an input instance which alter a model's classification decision. It has been shown that traditional adversarial training, which seeks to minimize a classifier's susceptibility to malicious perturbations, increases the cost of generated recourse; with larger adversarial training radii correlating with higher recourse costs. From the perspective of algorithmic recourse, however, the appropriate adversarial training radius has always been unknown. Another recent line of work has motivated adversarial training with adaptive training radii to address the issue of instance-wise variable adversarial vulnerability, showing success in domains with unknown attack radii. This work studies the effects of adaptive adversarial training on algorithmic recourse costs. We establish that the improvements in model robustness induced by adaptive adversarial training show little effect on algorithmic recourse costs, providing a potential avenue for affordable robustness in domains where recoursability is critical.



## **30. Black-Box Attacks against Signed Graph Analysis via Balance Poisoning**

cs.CR

**SubmitDate**: 2023-09-05    [abs](http://arxiv.org/abs/2309.02396v1) [paper-pdf](http://arxiv.org/pdf/2309.02396v1)

**Authors**: Jialong Zhou, Yuni Lai, Jian Ren, Kai Zhou

**Abstract**: Signed graphs are well-suited for modeling social networks as they capture both positive and negative relationships. Signed graph neural networks (SGNNs) are commonly employed to predict link signs (i.e., positive and negative) in such graphs due to their ability to handle the unique structure of signed graphs. However, real-world signed graphs are vulnerable to malicious attacks by manipulating edge relationships, and existing adversarial graph attack methods do not consider the specific structure of signed graphs. SGNNs often incorporate balance theory to effectively model the positive and negative links. Surprisingly, we find that the balance theory that they rely on can ironically be exploited as a black-box attack. In this paper, we propose a novel black-box attack called balance-attack that aims to decrease the balance degree of the signed graphs. We present an efficient heuristic algorithm to solve this NP-hard optimization problem. We conduct extensive experiments on five popular SGNN models and four real-world datasets to demonstrate the effectiveness and wide applicability of our proposed attack method. By addressing these challenges, our research contributes to a better understanding of the limitations and resilience of robust models when facing attacks on SGNNs. This work contributes to enhancing the security and reliability of signed graph analysis in social network modeling. Our PyTorch implementation of the attack is publicly available on GitHub: https://github.com/JialongZhou666/Balance-Attack.git.



## **31. Adaversarial Issue of Machine Learning Approaches Applied in Smart Grid: A Survey**

cs.CR

**SubmitDate**: 2023-09-05    [abs](http://arxiv.org/abs/2308.15736v2) [paper-pdf](http://arxiv.org/pdf/2308.15736v2)

**Authors**: Zhenyong Zhang, Mengxiang Liu

**Abstract**: The machine learning (ML) sees an increasing prevalence of being used in the internet-of-things enabled smart grid. However, the trustworthiness of ML is a severe issue that must be addressed to accommodate the trend of ML-based smart grid applications (MLsgAPPs). The adversarial distortion injected into the power signal will greatly affect the system's normal control and operation. Therefore, it is imperative to conduct vulnerability assessment for MLsgAPPs applied in the context of safety-critical power systems. In this paper, we provide a comprehensive review of the recent progress in designing attack and defense methods for MLsgAPPs. Unlike the traditional survey about ML security, this is the first review work about the security of MLsgAPPs that focuses on the characteristics of power systems. The survey is organized from the aspects of adversarial assumptions, targeted applications, evaluation metrics, defending approaches, physics-related constraints, and applied datasets. We also highlight future directions on this topic to encourage more researchers to conduct further research on adversarial attacks and defending approaches for MLsgAPPs.



## **32. When Measures are Unreliable: Imperceptible Adversarial Perturbations toward Top-$k$ Multi-Label Learning**

cs.CV

22 pages, 7 figures, accepted by ACM MM 2023

**SubmitDate**: 2023-09-05    [abs](http://arxiv.org/abs/2309.00007v2) [paper-pdf](http://arxiv.org/pdf/2309.00007v2)

**Authors**: Yuchen Sun, Qianqian Xu, Zitai Wang, Qingming Huang

**Abstract**: With the great success of deep neural networks, adversarial learning has received widespread attention in various studies, ranging from multi-class learning to multi-label learning. However, existing adversarial attacks toward multi-label learning only pursue the traditional visual imperceptibility but ignore the new perceptible problem coming from measures such as Precision@$k$ and mAP@$k$. Specifically, when a well-trained multi-label classifier performs far below the expectation on some samples, the victim can easily realize that this performance degeneration stems from attack, rather than the model itself. Therefore, an ideal multi-labeling adversarial attack should manage to not only deceive visual perception but also evade monitoring of measures. To this end, this paper first proposes the concept of measure imperceptibility. Then, a novel loss function is devised to generate such adversarial perturbations that could achieve both visual and measure imperceptibility. Furthermore, an efficient algorithm, which enjoys a convex objective, is established to optimize this objective. Finally, extensive experiments on large-scale benchmark datasets, such as PASCAL VOC 2012, MS COCO, and NUS WIDE, demonstrate the superiority of our proposed method in attacking the top-$k$ multi-label systems.



## **33. The Adversarial Implications of Variable-Time Inference**

cs.CR

**SubmitDate**: 2023-09-05    [abs](http://arxiv.org/abs/2309.02159v1) [paper-pdf](http://arxiv.org/pdf/2309.02159v1)

**Authors**: Dudi Biton, Aditi Misra, Efrat Levy, Jaidip Kotak, Ron Bitton, Roei Schuster, Nicolas Papernot, Yuval Elovici, Ben Nassi

**Abstract**: Machine learning (ML) models are known to be vulnerable to a number of attacks that target the integrity of their predictions or the privacy of their training data. To carry out these attacks, a black-box adversary must typically possess the ability to query the model and observe its outputs (e.g., labels). In this work, we demonstrate, for the first time, the ability to enhance such decision-based attacks. To accomplish this, we present an approach that exploits a novel side channel in which the adversary simply measures the execution time of the algorithm used to post-process the predictions of the ML model under attack. The leakage of inference-state elements into algorithmic timing side channels has never been studied before, and we have found that it can contain rich information that facilitates superior timing attacks that significantly outperform attacks based solely on label outputs. In a case study, we investigate leakage from the non-maximum suppression (NMS) algorithm, which plays a crucial role in the operation of object detectors. In our examination of the timing side-channel vulnerabilities associated with this algorithm, we identified the potential to enhance decision-based attacks. We demonstrate attacks against the YOLOv3 detector, leveraging the timing leakage to successfully evade object detection using adversarial examples, and perform dataset inference. Our experiments show that our adversarial examples exhibit superior perturbation quality compared to a decision-based attack. In addition, we present a new threat model in which dataset inference based solely on timing leakage is performed. To address the timing leakage vulnerability inherent in the NMS algorithm, we explore the potential and limitations of implementing constant-time inference passes as a mitigation strategy.



## **34. Robust Recommender System: A Survey and Future Directions**

cs.IR

**SubmitDate**: 2023-09-05    [abs](http://arxiv.org/abs/2309.02057v1) [paper-pdf](http://arxiv.org/pdf/2309.02057v1)

**Authors**: Kaike Zhang, Qi Cao, Fei Sun, Yunfan Wu, Shuchang Tao, Huawei Shen, Xueqi Cheng

**Abstract**: With the rapid growth of information, recommender systems have become integral for providing personalized suggestions and overcoming information overload. However, their practical deployment often encounters "dirty" data, where noise or malicious information can lead to abnormal recommendations. Research on improving recommender systems' robustness against such dirty data has thus gained significant attention. This survey provides a comprehensive review of recent work on recommender systems' robustness. We first present a taxonomy to organize current techniques for withstanding malicious attacks and natural noise. We then explore state-of-the-art methods in each category, including fraudster detection, adversarial training, certifiable robust training against malicious attacks, and regularization, purification, self-supervised learning against natural noise. Additionally, we summarize evaluation metrics and common datasets used to assess robustness. We discuss robustness across varying recommendation scenarios and its interplay with other properties like accuracy, interpretability, privacy, and fairness. Finally, we delve into open issues and future research directions in this emerging field. Our goal is to equip readers with a holistic understanding of robust recommender systems and spotlight pathways for future research and development.



## **35. F3B: A Low-Overhead Blockchain Architecture with Per-Transaction Front-Running Protection**

cs.CR

**SubmitDate**: 2023-09-05    [abs](http://arxiv.org/abs/2205.08529v3) [paper-pdf](http://arxiv.org/pdf/2205.08529v3)

**Authors**: Haoqian Zhang, Louis-Henri Merino, Ziyan Qu, Mahsa Bastankhah, Vero Estrada-Galinanes, Bryan Ford

**Abstract**: Front-running attacks, which benefit from advanced knowledge of pending transactions, have proliferated in the blockchain space since the emergence of decentralized finance. Front-running causes devastating losses to honest participants and continues to endanger the fairness of the ecosystem. We present Flash Freezing Flash Boys (F3B), a blockchain architecture that addresses front-running attacks by using threshold cryptography. In F3B, a user generates a symmetric key to encrypt their transaction, and once the underlying consensus layer has finalized the transaction, a decentralized secret-management committee reveals this key. F3B mitigates front-running attacks because, before the consensus group finalizes it, an adversary can no longer read the content of a transaction, thus preventing the adversary from benefiting from advanced knowledge of pending transactions. Unlike other mitigation systems, F3B properly ensures that all unfinalized transactions, even with significant delays, remain private by adopting per-transaction protection. Furthermore, F3B addresses front-running at the execution layer; thus, our solution is agnostic to the underlying consensus algorithm and compatible with existing smart contracts. We evaluated F3B on Ethereum with a modified execution layer and found only a negligible (0.026%) increase in transaction latency, specifically due to running threshold decryption with a 128-member secret-management committee after a transaction is finalized; this indicates that F3B is both practical and low-cost.



## **36. Boosting the Adversarial Transferability of Surrogate Models with Dark Knowledge**

cs.LG

Accepted at 2023 International Conference on Tools with Artificial  Intelligence (ICTAI)

**SubmitDate**: 2023-09-05    [abs](http://arxiv.org/abs/2206.08316v2) [paper-pdf](http://arxiv.org/pdf/2206.08316v2)

**Authors**: Dingcheng Yang, Zihao Xiao, Wenjian Yu

**Abstract**: Deep neural networks (DNNs) are vulnerable to adversarial examples. And, the adversarial examples have transferability, which means that an adversarial example for a DNN model can fool another model with a non-trivial probability. This gave birth to the transfer-based attack where the adversarial examples generated by a surrogate model are used to conduct black-box attacks. There are some work on generating the adversarial examples from a given surrogate model with better transferability. However, training a special surrogate model to generate adversarial examples with better transferability is relatively under-explored. This paper proposes a method for training a surrogate model with dark knowledge to boost the transferability of the adversarial examples generated by the surrogate model. This trained surrogate model is named dark surrogate model (DSM). The proposed method for training a DSM consists of two key components: a teacher model extracting dark knowledge, and the mixing augmentation skill enhancing dark knowledge of training data. We conducted extensive experiments to show that the proposed method can substantially improve the adversarial transferability of surrogate models across different architectures of surrogate models and optimizers for generating adversarial examples, and it can be applied to other scenarios of transfer-based attack that contain dark knowledge, like face verification. Our code is publicly available at \url{https://github.com/ydc123/Dark_Surrogate_Model}.



## **37. Efficient Defense Against Model Stealing Attacks on Convolutional Neural Networks**

cs.LG

Accepted for publication at 2023 International Conference on Machine  Learning and Applications (ICMLA)

**SubmitDate**: 2023-09-04    [abs](http://arxiv.org/abs/2309.01838v1) [paper-pdf](http://arxiv.org/pdf/2309.01838v1)

**Authors**: Kacem Khaled, Mouna Dhaouadi, Felipe Gohring de Magalhães, Gabriela Nicolescu

**Abstract**: Model stealing attacks have become a serious concern for deep learning models, where an attacker can steal a trained model by querying its black-box API. This can lead to intellectual property theft and other security and privacy risks. The current state-of-the-art defenses against model stealing attacks suggest adding perturbations to the prediction probabilities. However, they suffer from heavy computations and make impracticable assumptions about the adversary. They often require the training of auxiliary models. This can be time-consuming and resource-intensive which hinders the deployment of these defenses in real-world applications. In this paper, we propose a simple yet effective and efficient defense alternative. We introduce a heuristic approach to perturb the output probabilities. The proposed defense can be easily integrated into models without additional training. We show that our defense is effective in defending against three state-of-the-art stealing attacks. We evaluate our approach on large and quantized (i.e., compressed) Convolutional Neural Networks (CNNs) trained on several vision datasets. Our technique outperforms the state-of-the-art defenses with a $\times37$ faster inference latency without requiring any additional model and with a low impact on the model's performance. We validate that our defense is also effective for quantized CNNs targeting edge devices.



## **38. Baseline Defenses for Adversarial Attacks Against Aligned Language Models**

cs.LG

12 pages

**SubmitDate**: 2023-09-04    [abs](http://arxiv.org/abs/2309.00614v2) [paper-pdf](http://arxiv.org/pdf/2309.00614v2)

**Authors**: Neel Jain, Avi Schwarzschild, Yuxin Wen, Gowthami Somepalli, John Kirchenbauer, Ping-yeh Chiang, Micah Goldblum, Aniruddha Saha, Jonas Geiping, Tom Goldstein

**Abstract**: As Large Language Models quickly become ubiquitous, it becomes critical to understand their security vulnerabilities. Recent work shows that text optimizers can produce jailbreaking prompts that bypass moderation and alignment. Drawing from the rich body of work on adversarial machine learning, we approach these attacks with three questions: What threat models are practically useful in this domain? How do baseline defense techniques perform in this new domain? How does LLM security differ from computer vision?   We evaluate several baseline defense strategies against leading adversarial attacks on LLMs, discussing the various settings in which each is feasible and effective. Particularly, we look at three types of defenses: detection (perplexity based), input preprocessing (paraphrase and retokenization), and adversarial training. We discuss white-box and gray-box settings and discuss the robustness-performance trade-off for each of the defenses considered. We find that the weakness of existing discrete optimizers for text, combined with the relatively high costs of optimization, makes standard adaptive attacks more challenging for LLMs. Future research will be needed to uncover whether more powerful optimizers can be developed, or whether the strength of filtering and preprocessing defenses is greater in the LLMs domain than it has been in computer vision.



## **39. Machine Learning (In) Security: A Stream of Problems**

cs.CR

**SubmitDate**: 2023-09-04    [abs](http://arxiv.org/abs/2010.16045v2) [paper-pdf](http://arxiv.org/pdf/2010.16045v2)

**Authors**: Fabrício Ceschin, Marcus Botacin, Albert Bifet, Bernhard Pfahringer, Luiz S. Oliveira, Heitor Murilo Gomes, André Grégio

**Abstract**: Machine Learning (ML) has been widely applied to cybersecurity and is considered state-of-the-art for solving many of the open issues in that field. However, it is very difficult to evaluate how good the produced solutions are, since the challenges faced in security may not appear in other areas. One of these challenges is the concept drift, which increases the existing arms race between attackers and defenders: malicious actors can always create novel threats to overcome the defense solutions, which may not consider them in some approaches. Due to this, it is essential to know how to properly build and evaluate an ML-based security solution. In this paper, we identify, detail, and discuss the main challenges in the correct application of ML techniques to cybersecurity data. We evaluate how concept drift, evolution, delayed labels, and adversarial ML impact the existing solutions. Moreover, we address how issues related to data collection affect the quality of the results presented in the security literature, showing that new strategies are needed to improve current solutions. Finally, we present how existing solutions may fail under certain circumstances, and propose mitigations to them, presenting a novel checklist to help the development of future ML solutions for cybersecurity.



## **40. MathAttack: Attacking Large Language Models Towards Math Solving Ability**

cs.CL

11 pages, 6 figures

**SubmitDate**: 2023-09-04    [abs](http://arxiv.org/abs/2309.01686v1) [paper-pdf](http://arxiv.org/pdf/2309.01686v1)

**Authors**: Zihao Zhou, Qiufeng Wang, Mingyu Jin, Jie Yao, Jianan Ye, Wei Liu, Wei Wang, Xiaowei Huang, Kaizhu Huang

**Abstract**: With the boom of Large Language Models (LLMs), the research of solving Math Word Problem (MWP) has recently made great progress. However, there are few studies to examine the security of LLMs in math solving ability. Instead of attacking prompts in the use of LLMs, we propose a MathAttack model to attack MWP samples which are closer to the essence of security in solving math problems. Compared to traditional text adversarial attack, it is essential to preserve the mathematical logic of original MWPs during the attacking. To this end, we propose logical entity recognition to identify logical entries which are then frozen. Subsequently, the remaining text are attacked by adopting a word-level attacker. Furthermore, we propose a new dataset RobustMath to evaluate the robustness of LLMs in math solving ability. Extensive experiments on our RobustMath and two another math benchmark datasets GSM8K and MultiAirth show that MathAttack could effectively attack the math solving ability of LLMs. In the experiments, we observe that (1) Our adversarial samples from higher-accuracy LLMs are also effective for attacking LLMs with lower accuracy (e.g., transfer from larger to smaller-size LLMs, or from few-shot to zero-shot prompts); (2) Complex MWPs (such as more solving steps, longer text, more numbers) are more vulnerable to attack; (3) We can improve the robustness of LLMs by using our adversarial samples in few-shot prompts. Finally, we hope our practice and observation can serve as an important attempt towards enhancing the robustness of LLMs in math solving ability. We will release our code and dataset.



## **41. Hindering Adversarial Attacks with Multiple Encrypted Patch Embeddings**

cs.CV

To appear in APSIPA ASC 2023

**SubmitDate**: 2023-09-04    [abs](http://arxiv.org/abs/2309.01620v1) [paper-pdf](http://arxiv.org/pdf/2309.01620v1)

**Authors**: AprilPyone MaungMaung, Isao Echizen, Hitoshi Kiya

**Abstract**: In this paper, we propose a new key-based defense focusing on both efficiency and robustness. Although the previous key-based defense seems effective in defending against adversarial examples, carefully designed adaptive attacks can bypass the previous defense, and it is difficult to train the previous defense on large datasets like ImageNet. We build upon the previous defense with two major improvements: (1) efficient training and (2) optional randomization. The proposed defense utilizes one or more secret patch embeddings and classifier heads with a pre-trained isotropic network. When more than one secret embeddings are used, the proposed defense enables randomization on inference. Experiments were carried out on the ImageNet dataset, and the proposed defense was evaluated against an arsenal of state-of-the-art attacks, including adaptive ones. The results show that the proposed defense achieves a high robust accuracy and a comparable clean accuracy compared to the previous key-based defense.



## **42. Robustness of SAM: Segment Anything Under Corruptions and Beyond**

cs.CV

The first work evaluates the robustness of SAM under various  corruptions such as style transfer, local occlusion, and adversarial patch  attack

**SubmitDate**: 2023-09-04    [abs](http://arxiv.org/abs/2306.07713v3) [paper-pdf](http://arxiv.org/pdf/2306.07713v3)

**Authors**: Yu Qiao, Chaoning Zhang, Taegoo Kang, Donghun Kim, Chenshuang Zhang, Choong Seon Hong

**Abstract**: Segment anything model (SAM), as the name suggests, is claimed to be capable of cutting out any object and demonstrates impressive zero-shot transfer performance with the guidance of prompts. However, there is currently a lack of comprehensive evaluation regarding its robustness under various corruptions. Understanding the robustness of SAM across different corruption scenarios is crucial for its real-world deployment. Prior works show that SAM is biased towards texture (style) rather than shape, motivated by which we start by investigating its robustness against style transfer, which is synthetic corruption. Following by interpreting the effects of synthetic corruption as style changes, we proceed to conduct a comprehensive evaluation for its robustness against 15 types of common corruption. These corruptions mainly fall into categories such as digital, noise, weather, and blur, and within each corruption category, we explore 5 severity levels to simulate real-world corruption scenarios. Beyond the corruptions, we further assess the robustness of SAM against local occlusion and local adversarial patch attacks. To the best of our knowledge, our work is the first of its kind to evaluate the robustness of SAM under style change, local occlusion, and local adversarial patch attacks. Given that patch attacks visible to human eyes are easily detectable, we further assess its robustness against global adversarial attacks that are imperceptible to human eyes. Overall, this work provides a comprehensive empirical study of the robustness of SAM, evaluating its performance under various corruptions and extending the assessment to critical aspects such as local occlusion, local adversarial patch attacks, and global adversarial attacks. These evaluations yield valuable insights into the practical applicability and effectiveness of SAM in addressing real-world challenges.



## **43. OUTFOX: LLM-generated Essay Detection through In-context Learning with Adversarially Generated Examples**

cs.CL

**SubmitDate**: 2023-09-04    [abs](http://arxiv.org/abs/2307.11729v2) [paper-pdf](http://arxiv.org/pdf/2307.11729v2)

**Authors**: Ryuto Koike, Masahiro Kaneko, Naoaki Okazaki

**Abstract**: Large Language Models (LLMs) have achieved human-level fluency in text generation, making it difficult to distinguish between human-written and LLM-generated texts. This poses a growing risk of misuse of LLMs and demands the development of detectors to identify LLM-generated texts. However, existing detectors lack robustness against attacks: they degrade detection accuracy by simply paraphrasing LLM-generated texts. Furthermore, a malicious user might attempt to deliberately evade the detectors based on detection results, but this has not been assumed in previous studies. In this paper, we propose OUTFOX, a framework that improves the robustness of LLM-generated-text detectors by allowing both the detector and the attacker to consider each other's output. In this framework, the attacker uses the detector's prediction labels as examples for in-context learning and adversarially generates essays that are harder to detect, while the detector uses the adversarially generated essays as examples for in-context learning to learn to detect essays from a strong attacker. Experiments in the domain of student essays show that the proposed detector improves the detection performance on the attacker-generated texts by up to +41.3 points in F1-score. Furthermore, the proposed detector shows a state-of-the-art detection performance: up to 96.9 points in F1-score, beating existing detectors on non-attacked texts. Finally, the proposed attacker drastically degrades the performance of detectors by up to -57.0 points F1-score, massively outperforming the baseline paraphrasing method for evading detection.



## **44. Communication Lower Bounds for Cryptographic Broadcast Protocols**

cs.CR

A preliminary version of this work appeared in DISC 2023

**SubmitDate**: 2023-09-04    [abs](http://arxiv.org/abs/2309.01466v1) [paper-pdf](http://arxiv.org/pdf/2309.01466v1)

**Authors**: Erica Blum, Elette Boyle, Ran Cohen, Chen-Da Liu-Zhang

**Abstract**: Broadcast protocols enable a set of $n$ parties to agree on the input of a designated sender, even facing attacks by malicious parties. In the honest-majority setting, randomization and cryptography were harnessed to achieve low-communication broadcast with sub-quadratic total communication and balanced sub-linear cost per party. However, comparatively little is known in the dishonest-majority setting. Here, the most communication-efficient constructions are based on Dolev and Strong (SICOMP '83), and sub-quadratic broadcast has not been achieved. On the other hand, the only nontrivial $\omega(n)$ communication lower bounds are restricted to deterministic protocols, or against strong adaptive adversaries that can perform "after the fact" removal of messages.   We provide new communication lower bounds in this space, which hold against arbitrary cryptography and setup assumptions, as well as a simple protocol showing near tightness of our first bound.   1) We demonstrate a tradeoff between resiliency and communication for protocols secure against $n-o(n)$ static corruptions. For example, $\Omega(n\cdot {\sf polylog}(n))$ messages are needed when the number of honest parties is $n/{\sf polylog}(n)$; $\Omega(n\sqrt{n})$ messages are needed for $O(\sqrt{n})$ honest parties; and $\Omega(n^2)$ messages are needed for $O(1)$ honest parties.   Complementarily, we demonstrate broadcast with $O(n\cdot{\sf polylog}(n))$ total communication facing any constant fraction of static corruptions.   2) Our second bound considers $n/2 + k$ corruptions and a weakly adaptive adversary that cannot remove messages "after the fact." We show that any broadcast protocol within this setting can be attacked to force an arbitrary party to send messages to $k$ other parties. This rules out, for example, broadcast facing 51% corruptions in which all non-sender parties have sublinear communication locality.



## **45. Toward Defensive Letter Design**

cs.CV

14 pages, 8 figures, accepted at ACPR 2023

**SubmitDate**: 2023-09-04    [abs](http://arxiv.org/abs/2309.01452v1) [paper-pdf](http://arxiv.org/pdf/2309.01452v1)

**Authors**: Rentaro Kataoka, Akisato Kimura, Seiichi Uchida

**Abstract**: A major approach for defending against adversarial attacks aims at controlling only image classifiers to be more resilient, and it does not care about visual objects, such as pandas and cars, in images. This means that visual objects themselves cannot take any defensive actions, and they are still vulnerable to adversarial attacks. In contrast, letters are artificial symbols, and we can freely control their appearance unless losing their readability. In other words, we can make the letters more defensive to the attacks. This paper poses three research questions related to the adversarial vulnerability of letter images: (1) How defensive are the letters against adversarial attacks? (2) Can we estimate how defensive a given letter image is before attacks? (3) Can we control the letter images to be more defensive against adversarial attacks? For answering the first and second questions, we measure the defensibility of letters by employing Iterative Fast Gradient Sign Method (I-FGSM) and then build a deep regression model for estimating the defensibility of each letter image. We also propose a two-step method based on a generative adversarial network (GAN) for generating character images with higher defensibility, which solves the third research question.



## **46. Open Sesame! Universal Black Box Jailbreaking of Large Language Models**

cs.CL

**SubmitDate**: 2023-09-04    [abs](http://arxiv.org/abs/2309.01446v1) [paper-pdf](http://arxiv.org/pdf/2309.01446v1)

**Authors**: Raz Lapid, Ron Langberg, Moshe Sipper

**Abstract**: Large language models (LLMs), designed to provide helpful and safe responses, often rely on alignment techniques to align with user intent and social guidelines. Unfortunately, this alignment can be exploited by malicious actors seeking to manipulate an LLM's outputs for unintended purposes. In this paper we introduce a novel approach that employs a genetic algorithm (GA) to manipulate LLMs when model architecture and parameters are inaccessible. The GA attack works by optimizing a universal adversarial prompt that -- when combined with a user's query -- disrupts the attacked model's alignment, resulting in unintended and potentially harmful outputs. Our novel approach systematically reveals a model's limitations and vulnerabilities by uncovering instances where its responses deviate from expected behavior. Through extensive experiments we demonstrate the efficacy of our technique, thus contributing to the ongoing discussion on responsible AI development by providing a diagnostic tool for evaluating and enhancing alignment of LLMs with human intent. To our knowledge this is the first automated universal black box jailbreak attack.



## **47. Adv3D: Generating 3D Adversarial Examples in Driving Scenarios with NeRF**

cs.CV

**SubmitDate**: 2023-09-04    [abs](http://arxiv.org/abs/2309.01351v1) [paper-pdf](http://arxiv.org/pdf/2309.01351v1)

**Authors**: Leheng Li, Qing Lian, Ying-Cong Chen

**Abstract**: Deep neural networks (DNNs) have been proven extremely susceptible to adversarial examples, which raises special safety-critical concerns for DNN-based autonomous driving stacks (i.e., 3D object detection). Although there are extensive works on image-level attacks, most are restricted to 2D pixel spaces, and such attacks are not always physically realistic in our 3D world. Here we present Adv3D, the first exploration of modeling adversarial examples as Neural Radiance Fields (NeRFs). Advances in NeRF provide photorealistic appearances and 3D accurate generation, yielding a more realistic and realizable adversarial example. We train our adversarial NeRF by minimizing the surrounding objects' confidence predicted by 3D detectors on the training set. Then we evaluate Adv3D on the unseen validation set and show that it can cause a large performance reduction when rendering NeRF in any sampled pose. To generate physically realizable adversarial examples, we propose primitive-aware sampling and semantic-guided regularization that enable 3D patch attacks with camouflage adversarial texture. Experimental results demonstrate that the trained adversarial NeRF generalizes well to different poses, scenes, and 3D detectors. Finally, we provide a defense method to our attacks that involves adversarial training through data augmentation. Project page: https://len-li.github.io/adv3d-web



## **48. Everyone Can Attack: Repurpose Lossy Compression as a Natural Backdoor Attack**

cs.CR

14 pages. This paper shows everyone can mount a powerful and stealthy  backdoor attack with the widely-used lossy image compression

**SubmitDate**: 2023-09-03    [abs](http://arxiv.org/abs/2308.16684v2) [paper-pdf](http://arxiv.org/pdf/2308.16684v2)

**Authors**: Sze Jue Yang, Quang Nguyen, Chee Seng Chan, Khoa D. Doan

**Abstract**: The vulnerabilities to backdoor attacks have recently threatened the trustworthiness of machine learning models in practical applications. Conventional wisdom suggests that not everyone can be an attacker since the process of designing the trigger generation algorithm often involves significant effort and extensive experimentation to ensure the attack's stealthiness and effectiveness. Alternatively, this paper shows that there exists a more severe backdoor threat: anyone can exploit an easily-accessible algorithm for silent backdoor attacks. Specifically, this attacker can employ the widely-used lossy image compression from a plethora of compression tools to effortlessly inject a trigger pattern into an image without leaving any noticeable trace; i.e., the generated triggers are natural artifacts. One does not require extensive knowledge to click on the "convert" or "save as" button while using tools for lossy image compression. Via this attack, the adversary does not need to design a trigger generator as seen in prior works and only requires poisoning the data. Empirically, the proposed attack consistently achieves 100% attack success rate in several benchmark datasets such as MNIST, CIFAR-10, GTSRB and CelebA. More significantly, the proposed attack can still achieve almost 100% attack success rate with very small (approximately 10%) poisoning rates in the clean label setting. The generated trigger of the proposed attack using one lossy compression algorithm is also transferable across other related compression algorithms, exacerbating the severity of this backdoor threat. This work takes another crucial step toward understanding the extensive risks of backdoor attacks in practice, urging practitioners to investigate similar attacks and relevant backdoor mitigation methods.



## **49. A Comprehensive Study and Comparison of the Robustness of 3D Object Detectors Against Adversarial Attacks**

cs.CV

30 pages, 14 figures

**SubmitDate**: 2023-09-03    [abs](http://arxiv.org/abs/2212.10230v2) [paper-pdf](http://arxiv.org/pdf/2212.10230v2)

**Authors**: Yifan Zhang, Junhui Hou, Yixuan Yuan

**Abstract**: Recent years have witnessed significant advancements in deep learning-based 3D object detection, leading to its widespread adoption in numerous applications. As 3D object detectors become increasingly crucial for security-critical tasks, it is imperative to understand their robustness against adversarial attacks. This paper presents the first comprehensive evaluation and analysis of the robustness of LiDAR-based 3D detectors under adversarial attacks. Specifically, we extend three distinct adversarial attacks to the 3D object detection task, benchmarking the robustness of state-of-the-art LiDAR-based 3D object detectors against attacks on the KITTI and Waymo datasets. We further analyze the relationship between robustness and detector properties. Additionally, we explore the transferability of cross-model, cross-task, and cross-data attacks. Thorough experiments on defensive strategies for 3D detectors are conducted, demonstrating that simple transformations like flipping provide little help in improving robustness when the applied transformation strategy is exposed to attackers. Finally, we propose balanced adversarial focal training, based on conventional adversarial training, to strike a balance between accuracy and robustness. Our findings will facilitate investigations into understanding and defending against adversarial attacks on LiDAR-based 3D object detectors, thus advancing the field. The source code is publicly available at \url{https://github.com/Eaphan/Robust3DOD}.



## **50. AdvMono3D: Advanced Monocular 3D Object Detection with Depth-Aware Robust Adversarial Training**

cs.CV

**SubmitDate**: 2023-09-03    [abs](http://arxiv.org/abs/2309.01106v1) [paper-pdf](http://arxiv.org/pdf/2309.01106v1)

**Authors**: Xingyuan Li, Jinyuan Liu, Long Ma, Xin Fan, Risheng Liu

**Abstract**: Monocular 3D object detection plays a pivotal role in the field of autonomous driving and numerous deep learning-based methods have made significant breakthroughs in this area. Despite the advancements in detection accuracy and efficiency, these models tend to fail when faced with such attacks, rendering them ineffective. Therefore, bolstering the adversarial robustness of 3D detection models has become a crucial issue that demands immediate attention and innovative solutions. To mitigate this issue, we propose a depth-aware robust adversarial training method for monocular 3D object detection, dubbed DART3D. Specifically, we first design an adversarial attack that iteratively degrades the 2D and 3D perception capabilities of 3D object detection models(IDP), serves as the foundation for our subsequent defense mechanism. In response to this attack, we propose an uncertainty-based residual learning method for adversarial training. Our adversarial training approach capitalizes on the inherent uncertainty, enabling the model to significantly improve its robustness against adversarial attacks. We conducted extensive experiments on the KITTI 3D datasets, demonstrating that DART3D surpasses direct adversarial training (the most popular approach) under attacks in 3D object detection $AP_{R40}$ of car category for the Easy, Moderate, and Hard settings, with improvements of 4.415%, 4.112%, and 3.195%, respectively.



