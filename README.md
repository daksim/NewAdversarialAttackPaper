# Latest Adversarial Attack Papers
**update at 2024-03-04 16:55:53**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. Adversarial Examples are Misaligned in Diffusion Model Manifolds**

cs.CV

under review

**SubmitDate**: 2024-03-01    [abs](http://arxiv.org/abs/2401.06637v4) [paper-pdf](http://arxiv.org/pdf/2401.06637v4)

**Authors**: Peter Lorenz, Ricard Durall, Janis Keuper

**Abstract**: In recent years, diffusion models (DMs) have drawn significant attention for their success in approximating data distributions, yielding state-of-the-art generative results. Nevertheless, the versatility of these models extends beyond their generative capabilities to encompass various vision applications, such as image inpainting, segmentation, adversarial robustness, among others. This study is dedicated to the investigation of adversarial attacks through the lens of diffusion models. However, our objective does not involve enhancing the adversarial robustness of image classifiers. Instead, our focus lies in utilizing the diffusion model to detect and analyze the anomalies introduced by these attacks on images. To that end, we systematically examine the alignment of the distributions of adversarial examples when subjected to the process of transformation using diffusion models. The efficacy of this approach is assessed across CIFAR-10 and ImageNet datasets, including varying image sizes in the latter. The results demonstrate a notable capacity to discriminate effectively between benign and attacked images, providing compelling evidence that adversarial instances do not align with the learned manifold of the DMs.



## **2. Unfolding Local Growth Rate Estimates for (Almost) Perfect Adversarial Detection**

cs.CV

accepted at VISAPP23

**SubmitDate**: 2024-03-01    [abs](http://arxiv.org/abs/2212.06776v5) [paper-pdf](http://arxiv.org/pdf/2212.06776v5)

**Authors**: Peter Lorenz, Margret Keuper, Janis Keuper

**Abstract**: Convolutional neural networks (CNN) define the state-of-the-art solution on many perceptual tasks. However, current CNN approaches largely remain vulnerable against adversarial perturbations of the input that have been crafted specifically to fool the system while being quasi-imperceptible to the human eye. In recent years, various approaches have been proposed to defend CNNs against such attacks, for example by model hardening or by adding explicit defence mechanisms. Thereby, a small "detector" is included in the network and trained on the binary classification task of distinguishing genuine data from data containing adversarial perturbations. In this work, we propose a simple and light-weight detector, which leverages recent findings on the relation between networks' local intrinsic dimensionality (LID) and adversarial attacks. Based on a re-interpretation of the LID measure and several simple adaptations, we surpass the state-of-the-art on adversarial detection by a significant margin and reach almost perfect results in terms of F1-score for several networks and datasets. Sources available at: https://github.com/adverML/multiLID



## **3. Protect and Extend -- Using GANs for Synthetic Data Generation of Time-Series Medical Records**

cs.LG

**SubmitDate**: 2024-03-01    [abs](http://arxiv.org/abs/2402.14042v2) [paper-pdf](http://arxiv.org/pdf/2402.14042v2)

**Authors**: Navid Ashrafi, Vera Schmitt, Robert P. Spang, Sebastian Möller, Jan-Niklas Voigt-Antons

**Abstract**: Preservation of private user data is of paramount importance for high Quality of Experience (QoE) and acceptability, particularly with services treating sensitive data, such as IT-based health services. Whereas anonymization techniques were shown to be prone to data re-identification, synthetic data generation has gradually replaced anonymization since it is relatively less time and resource-consuming and more robust to data leakage. Generative Adversarial Networks (GANs) have been used for generating synthetic datasets, especially GAN frameworks adhering to the differential privacy phenomena. This research compares state-of-the-art GAN-based models for synthetic data generation to generate time-series synthetic medical records of dementia patients which can be distributed without privacy concerns. Predictive modeling, autocorrelation, and distribution analysis are used to assess the Quality of Generating (QoG) of the generated data. The privacy preservation of the respective models is assessed by applying membership inference attacks to determine potential data leakage risks. Our experiments indicate the superiority of the privacy-preserving GAN (PPGAN) model over other models regarding privacy preservation while maintaining an acceptable level of QoG. The presented results can support better data protection for medical use cases in the future.



## **4. Attacking Delay-based PUFs with Minimal Adversary Model**

cs.CR

13 pages, 6 figures, journal

**SubmitDate**: 2024-03-01    [abs](http://arxiv.org/abs/2403.00464v1) [paper-pdf](http://arxiv.org/pdf/2403.00464v1)

**Authors**: Hongming Fei, Owen Millwood, Prosanta Gope, Jack Miskelly, Biplab Sikdar

**Abstract**: Physically Unclonable Functions (PUFs) provide a streamlined solution for lightweight device authentication. Delay-based Arbiter PUFs, with their ease of implementation and vast challenge space, have received significant attention; however, they are not immune to modelling attacks that exploit correlations between their inputs and outputs. Research is therefore polarized between developing modelling-resistant PUFs and devising machine learning attacks against them. This dichotomy often results in exaggerated concerns and overconfidence in PUF security, primarily because there lacks a universal tool to gauge a PUF's security. In many scenarios, attacks require additional information, such as PUF type or configuration parameters. Alarmingly, new PUFs are often branded `secure' if they lack a specific attack model upon introduction. To impartially assess the security of delay-based PUFs, we present a generic framework featuring a Mixture-of-PUF-Experts (MoPE) structure for mounting attacks on various PUFs with minimal adversarial knowledge, which provides a way to compare their performance fairly and impartially. We demonstrate the capability of our model to attack different PUF types, including the first successful attack on Heterogeneous Feed-Forward PUFs using only a reasonable amount of challenges and responses. We propose an extension version of our model, a Multi-gate Mixture-of-PUF-Experts (MMoPE) structure, facilitating multi-task learning across diverse PUFs to recognise commonalities across PUF designs. This allows a streamlining of training periods for attacking multiple PUFs simultaneously. We conclude by showcasing the potent performance of MoPE and MMoPE across a spectrum of PUF types, employing simulated, real-world unbiased, and biased data sets for analysis.



## **5. Robust Deep Reinforcement Learning Through Adversarial Attacks and Training : A Survey**

cs.LG

57 pages, 16 figues, 2 tables

**SubmitDate**: 2024-03-01    [abs](http://arxiv.org/abs/2403.00420v1) [paper-pdf](http://arxiv.org/pdf/2403.00420v1)

**Authors**: Lucas Schott, Josephine Delas, Hatem Hajri, Elies Gherbi, Reda Yaich, Nora Boulahia-Cuppens, Frederic Cuppens, Sylvain Lamprier

**Abstract**: Deep Reinforcement Learning (DRL) is an approach for training autonomous agents across various complex environments. Despite its significant performance in well known environments, it remains susceptible to minor conditions variations, raising concerns about its reliability in real-world applications. To improve usability, DRL must demonstrate trustworthiness and robustness. A way to improve robustness of DRL to unknown changes in the conditions is through Adversarial Training, by training the agent against well suited adversarial attacks on the dynamics of the environment. Addressing this critical issue, our work presents an in-depth analysis of contemporary adversarial attack methodologies, systematically categorizing them and comparing their objectives and operational mechanisms. This classification offers a detailed insight into how adversarial attacks effectively act for evaluating the resilience of DRL agents, thereby paving the way for enhancing their robustness.



## **6. DrAttack: Prompt Decomposition and Reconstruction Makes Powerful LLM Jailbreakers**

cs.CR

**SubmitDate**: 2024-03-01    [abs](http://arxiv.org/abs/2402.16914v2) [paper-pdf](http://arxiv.org/pdf/2402.16914v2)

**Authors**: Xirui Li, Ruochen Wang, Minhao Cheng, Tianyi Zhou, Cho-Jui Hsieh

**Abstract**: The safety alignment of Large Language Models (LLMs) is vulnerable to both manual and automated jailbreak attacks, which adversarially trigger LLMs to output harmful content. However, current methods for jailbreaking LLMs, which nest entire harmful prompts, are not effective at concealing malicious intent and can be easily identified and rejected by well-aligned LLMs. This paper discovers that decomposing a malicious prompt into separated sub-prompts can effectively obscure its underlying malicious intent by presenting it in a fragmented, less detectable form, thereby addressing these limitations. We introduce an automatic prompt \textbf{D}ecomposition and \textbf{R}econstruction framework for jailbreak \textbf{Attack} (DrAttack). DrAttack includes three key components: (a) `Decomposition' of the original prompt into sub-prompts, (b) `Reconstruction' of these sub-prompts implicitly by in-context learning with semantically similar but harmless reassembling demo, and (c) a `Synonym Search' of sub-prompts, aiming to find sub-prompts' synonyms that maintain the original intent while jailbreaking LLMs. An extensive empirical study across multiple open-source and closed-source LLMs demonstrates that, with a significantly reduced number of queries, DrAttack obtains a substantial gain of success rate over prior SOTA prompt-only attackers. Notably, the success rate of 78.0\% on GPT-4 with merely 15 queries surpassed previous art by 33.1\%. The project is available at https://github.com/xirui-li/DrAttack.



## **7. SoK: Security of Programmable Logic Controllers**

cs.CR

25 pages, 13 figures, Extended version February 2024, A shortened  version is to be published in the 33rd USENIX Security Symposium, for more  information, see https://efrenlopez.org/

**SubmitDate**: 2024-03-01    [abs](http://arxiv.org/abs/2403.00280v1) [paper-pdf](http://arxiv.org/pdf/2403.00280v1)

**Authors**: Efrén López-Morales, Ulysse Planta, Carlos Rubio-Medrano, Ali Abbasi, Alvaro A. Cardenas

**Abstract**: Billions of people rely on essential utility and manufacturing infrastructures such as water treatment plants, energy management, and food production. Our dependence on reliable infrastructures makes them valuable targets for cyberattacks. One of the prime targets for adversaries attacking physical infrastructures are Programmable Logic Controllers (PLCs) because they connect the cyber and physical worlds. In this study, we conduct the first comprehensive systematization of knowledge that explores the security of PLCs: We present an in-depth analysis of PLC attacks and defenses and discover trends in the security of PLCs from the last 17 years of research. We introduce a novel threat taxonomy for PLCs and Industrial Control Systems (ICS). Finally, we identify and point out research gaps that, if left ignored, could lead to new catastrophic attacks against critical infrastructures.



## **8. LoRA-as-an-Attack! Piercing LLM Safety Under The Share-and-Play Scenario**

cs.CR

**SubmitDate**: 2024-02-29    [abs](http://arxiv.org/abs/2403.00108v1) [paper-pdf](http://arxiv.org/pdf/2403.00108v1)

**Authors**: Hongyi Liu, Zirui Liu, Ruixiang Tang, Jiayi Yuan, Shaochen Zhong, Yu-Neng Chuang, Li Li, Rui Chen, Xia Hu

**Abstract**: Fine-tuning LLMs is crucial to enhancing their task-specific performance and ensuring model behaviors are aligned with human preferences. Among various fine-tuning methods, LoRA is popular for its efficiency and ease to use, allowing end-users to easily post and adopt lightweight LoRA modules on open-source platforms to tailor their model for different customization. However, such a handy share-and-play setting opens up new attack surfaces, that the attacker can render LoRA as an attacker, such as backdoor injection, and widely distribute the adversarial LoRA to the community easily. This can result in detrimental outcomes. Despite the huge potential risks of sharing LoRA modules, this aspect however has not been fully explored. To fill the gap, in this study we thoroughly investigate the attack opportunities enabled in the growing share-and-play scenario. Specifically, we study how to inject backdoor into the LoRA module and dive deeper into LoRA's infection mechanisms. We found that training-free mechanism is possible in LoRA backdoor injection. We also discover the impact of backdoor attacks with the presence of multiple LoRA adaptions concurrently as well as LoRA based backdoor transferability. Our aim is to raise awareness of the potential risks under the emerging share-and-play scenario, so as to proactively prevent potential consequences caused by LoRA-as-an-Attack. Warning: the paper contains potential offensive content generated by models.



## **9. Unraveling Adversarial Examples against Speaker Identification -- Techniques for Attack Detection and Victim Model Classification**

cs.SD

**SubmitDate**: 2024-02-29    [abs](http://arxiv.org/abs/2402.19355v1) [paper-pdf](http://arxiv.org/pdf/2402.19355v1)

**Authors**: Sonal Joshi, Thomas Thebaud, Jesús Villalba, Najim Dehak

**Abstract**: Adversarial examples have proven to threaten speaker identification systems, and several countermeasures against them have been proposed. In this paper, we propose a method to detect the presence of adversarial examples, i.e., a binary classifier distinguishing between benign and adversarial examples. We build upon and extend previous work on attack type classification by exploring new architectures. Additionally, we introduce a method for identifying the victim model on which the adversarial attack is carried out. To achieve this, we generate a new dataset containing multiple attacks performed against various victim models. We achieve an AUC of 0.982 for attack detection, with no more than a 0.03 drop in performance for unknown attacks. Our attack classification accuracy (excluding benign) reaches 86.48% across eight attack types using our LightResNet34 architecture, while our victim model classification accuracy reaches 72.28% across four victim models.



## **10. Verification of Neural Networks' Global Robustness**

cs.LG

**SubmitDate**: 2024-02-29    [abs](http://arxiv.org/abs/2402.19322v1) [paper-pdf](http://arxiv.org/pdf/2402.19322v1)

**Authors**: Anan Kabaha, Dana Drachsler-Cohen

**Abstract**: Neural networks are successful in various applications but are also susceptible to adversarial attacks. To show the safety of network classifiers, many verifiers have been introduced to reason about the local robustness of a given input to a given perturbation. While successful, local robustness cannot generalize to unseen inputs. Several works analyze global robustness properties, however, neither can provide a precise guarantee about the cases where a network classifier does not change its classification. In this work, we propose a new global robustness property for classifiers aiming at finding the minimal globally robust bound, which naturally extends the popular local robustness property for classifiers. We introduce VHAGaR, an anytime verifier for computing this bound. VHAGaR relies on three main ideas: encoding the problem as a mixed-integer programming and pruning the search space by identifying dependencies stemming from the perturbation or network computation and generalizing adversarial attacks to unknown inputs. We evaluate VHAGaR on several datasets and classifiers and show that, given a three hour timeout, the average gap between the lower and upper bound on the minimal globally robust bound computed by VHAGaR is 1.9, while the gap of an existing global robustness verifier is 154.7. Moreover, VHAGaR is 130.6x faster than this verifier. Our results further indicate that leveraging dependencies and adversarial attacks makes VHAGaR 78.6x faster.



## **11. Attacks Against Mobility Prediction in 5G Networks**

cs.CR

This is the preprint version of a paper which appears in 22th IEEE  International Conference on Trust, Security and Privacy in Computing and  Communications (TrustCom 2023)

**SubmitDate**: 2024-02-29    [abs](http://arxiv.org/abs/2402.19319v1) [paper-pdf](http://arxiv.org/pdf/2402.19319v1)

**Authors**: Syafiq Al Atiiq, Yachao Yuan, Christian Gehrmann, Jakob Sternby, Luis Barriga

**Abstract**: The $5^{th}$ generation of mobile networks introduces a new Network Function (NF) that was not present in previous generations, namely the Network Data Analytics Function (NWDAF). Its primary objective is to provide advanced analytics services to various entities within the network and also towards external application services in the 5G ecosystem. One of the key use cases of NWDAF is mobility trajectory prediction, which aims to accurately support efficient mobility management of User Equipment (UE) in the network by allocating ``just in time'' necessary network resources. In this paper, we show that there are potential mobility attacks that can compromise the accuracy of these predictions. In a semi-realistic scenario with 10,000 subscribers, we demonstrate that an adversary equipped with the ability to hijack cellular mobile devices and clone them can significantly reduce the prediction accuracy from 75\% to 40\% using just 100 adversarial UEs. While a defense mechanism largely depends on the attack and the mobility types in a particular area, we prove that a basic KMeans clustering is effective in distinguishing legitimate and adversarial UEs.



## **12. Topology-Based Reconstruction Prevention for Decentralised Learning**

cs.CR

13 pages, 8 figures, submitted to PETS 2024, for associated  experiment source code see doi:10.4121/21572601

**SubmitDate**: 2024-02-29    [abs](http://arxiv.org/abs/2312.05248v2) [paper-pdf](http://arxiv.org/pdf/2312.05248v2)

**Authors**: Florine W. Dekker, Zekeriya Erkin, Mauro Conti

**Abstract**: Decentralised learning has recently gained traction as an alternative to federated learning in which both data and coordination are distributed over its users. To preserve data confidentiality, decentralised learning relies on differential privacy, multi-party computation, or a combination thereof. However, running multiple privacy-preserving summations in sequence may allow adversaries to perform reconstruction attacks. Unfortunately, current reconstruction countermeasures either cannot trivially be adapted to the distributed setting, or add excessive amounts of noise.   In this work, we first show that passive honest-but-curious adversaries can infer other users' private data after several privacy-preserving summations. For example, in subgraphs with 18 users, we show that only three passive honest-but-curious adversaries succeed at reconstructing private data 11.0% of the time, requiring an average of 8.8 summations per adversary. The success rate depends only on the adversaries' direct neighbourhood, independent of the size of the full network. We consider weak adversaries, who do not control the graph topology and can exploit neither the inner workings of the summation protocol nor the specifics of users' data.   We develop a mathematical understanding of how reconstruction relates to topology and propose the first topology-based decentralised defence against reconstruction attacks. Specifically, we show that reconstruction requires a number of adversaries linear in the length of the network's shortest cycle. Consequently, reconstructing private data from privacy-preserving summations is impossible in acyclic networks.   Our work is a stepping stone for a formal theory of topology-based reconstruction defences. Such a theory would generalise our countermeasure beyond summation, define confidentiality in terms of entropy, and describe the effects of differential privacy.



## **13. How to Train your Antivirus: RL-based Hardening through the Problem-Space**

cs.CR

20 pages,4 figures

**SubmitDate**: 2024-02-29    [abs](http://arxiv.org/abs/2402.19027v1) [paper-pdf](http://arxiv.org/pdf/2402.19027v1)

**Authors**: Jacopo Cortellazzi, Ilias Tsingenopoulos, Branislav Bošanský, Simone Aonzo, Davy Preuveneers, Wouter Joosen, Fabio Pierazzi, Lorenzo Cavallaro

**Abstract**: ML-based malware detection on dynamic analysis reports is vulnerable to both evasion and spurious correlations. In this work, we investigate a specific ML architecture employed in the pipeline of a widely-known commercial antivirus company, with the goal to harden it against adversarial malware. Adversarial training, the sole defensive technique that can confer empirical robustness, is not applicable out of the box in this domain, for the principal reason that gradient-based perturbations rarely map back to feasible problem-space programs. We introduce a novel Reinforcement Learning approach for constructing adversarial examples, a constituent part of adversarially training a model against evasion. Our approach comes with multiple advantages. It performs modifications that are feasible in the problem-space, and only those; thus it circumvents the inverse mapping problem. It also makes possible to provide theoretical guarantees on the robustness of the model against a particular set of adversarial capabilities. Our empirical exploration validates our theoretical insights, where we can consistently reach 0\% Attack Success Rate after a few adversarial retraining iterations.



## **14. Invariant Aggregator for Defending against Federated Backdoor Attacks**

cs.LG

**SubmitDate**: 2024-02-29    [abs](http://arxiv.org/abs/2210.01834v3) [paper-pdf](http://arxiv.org/pdf/2210.01834v3)

**Authors**: Xiaoyang Wang, Dimitrios Dimitriadis, Sanmi Koyejo, Shruti Tople

**Abstract**: Federated learning enables training high-utility models across several clients without directly sharing their private data. As a downside, the federated setting makes the model vulnerable to various adversarial attacks in the presence of malicious clients. Despite the theoretical and empirical success in defending against attacks that aim to degrade models' utility, defense against backdoor attacks that increase model accuracy on backdoor samples exclusively without hurting the utility on other samples remains challenging. To this end, we first analyze the failure modes of existing defenses over a flat loss landscape, which is common for well-designed neural networks such as Resnet [He et al., 2015] but is often overlooked by previous works. Then, we propose an invariant aggregator that redirects the aggregated update to invariant directions that are generally useful via selectively masking out the update elements that favor few and possibly malicious clients. Theoretical results suggest that our approach provably mitigates backdoor attacks and remains effective over flat loss landscapes. Empirical results on three datasets with different modalities and varying numbers of clients further demonstrate that our approach mitigates a broad class of backdoor attacks with a negligible cost on the model utility.



## **15. MPAT: Building Robust Deep Neural Networks against Textual Adversarial Attacks**

cs.LG

**SubmitDate**: 2024-02-29    [abs](http://arxiv.org/abs/2402.18792v1) [paper-pdf](http://arxiv.org/pdf/2402.18792v1)

**Authors**: Fangyuan Zhang, Huichi Zhou, Shuangjiao Li, Hongtao Wang

**Abstract**: Deep neural networks have been proven to be vulnerable to adversarial examples and various methods have been proposed to defend against adversarial attacks for natural language processing tasks. However, previous defense methods have limitations in maintaining effective defense while ensuring the performance of the original task. In this paper, we propose a malicious perturbation based adversarial training method (MPAT) for building robust deep neural networks against textual adversarial attacks. Specifically, we construct a multi-level malicious example generation strategy to generate adversarial examples with malicious perturbations, which are used instead of original inputs for model training. Additionally, we employ a novel training objective function to ensure achieving the defense goal without compromising the performance on the original task. We conduct comprehensive experiments to evaluate our defense method by attacking five victim models on three benchmark datasets. The result demonstrates that our method is more effective against malicious adversarial attacks compared with previous defense methods while maintaining or further improving the performance on the original task.



## **16. Enhancing the "Immunity" of Mixture-of-Experts Networks for Adversarial Defense**

cs.LG

**SubmitDate**: 2024-02-29    [abs](http://arxiv.org/abs/2402.18787v1) [paper-pdf](http://arxiv.org/pdf/2402.18787v1)

**Authors**: Qiao Han, yong huang, xinling Guo, Yiteng Zhai, Yu Qin, Yao Yang

**Abstract**: Recent studies have revealed the vulnerability of Deep Neural Networks (DNNs) to adversarial examples, which can easily fool DNNs into making incorrect predictions. To mitigate this deficiency, we propose a novel adversarial defense method called "Immunity" (Innovative MoE with MUtual information \& positioN stabilITY) based on a modified Mixture-of-Experts (MoE) architecture in this work. The key enhancements to the standard MoE are two-fold: 1) integrating of Random Switch Gates (RSGs) to obtain diverse network structures via random permutation of RSG parameters at evaluation time, despite of RSGs being determined after one-time training; 2) devising innovative Mutual Information (MI)-based and Position Stability-based loss functions by capitalizing on Grad-CAM's explanatory power to increase the diversity and the causality of expert networks. Notably, our MI-based loss operates directly on the heatmaps, thereby inducing subtler negative impacts on the classification performance when compared to other losses of the same type, theoretically. Extensive evaluation validates the efficacy of the proposed approach in improving adversarial robustness against a wide range of attacks.



## **17. On Defeating Graph Analysis of Anonymous Transactions**

cs.CR

**SubmitDate**: 2024-02-28    [abs](http://arxiv.org/abs/2402.18755v1) [paper-pdf](http://arxiv.org/pdf/2402.18755v1)

**Authors**: Christoph Egger, Russell W. F. Lai, Viktoria Ronge, Ivy K. Y. Woo, Hoover H. F. Yin

**Abstract**: In a ring-signature-based anonymous cryptocurrency, signers of a transaction are hidden among a set of potential signers, called a ring, whose size is much smaller than the number of all users. The ring-membership relations specified by the sets of transactions thus induce bipartite transaction graphs, whose distribution is in turn induced by the ring sampler underlying the cryptocurrency.   Since efficient graph analysis could be performed on transaction graphs to potentially deanonymise signers, it is crucial to understand the resistance of (the transaction graphs induced by) a ring sampler against graph analysis. Of particular interest is the class of partitioning ring samplers. Although previous works showed that they provide almost optimal local anonymity, their resistance against global, e.g. graph-based, attacks were unclear.   In this work, we analyse transaction graphs induced by partitioning ring samplers. Specifically, we show (partly analytically and partly empirically) that, somewhat surprisingly, by setting the ring size to be at least logarithmic in the number of users, a graph-analysing adversary is no better than the one that performs random guessing in deanonymisation up to constant factor of 2.



## **18. A New Era in LLM Security: Exploring Security Concerns in Real-World LLM-based Systems**

cs.CR

**SubmitDate**: 2024-02-28    [abs](http://arxiv.org/abs/2402.18649v1) [paper-pdf](http://arxiv.org/pdf/2402.18649v1)

**Authors**: Fangzhou Wu, Ning Zhang, Somesh Jha, Patrick McDaniel, Chaowei Xiao

**Abstract**: Large Language Model (LLM) systems are inherently compositional, with individual LLM serving as the core foundation with additional layers of objects such as plugins, sandbox, and so on. Along with the great potential, there are also increasing concerns over the security of such probabilistic intelligent systems. However, existing studies on LLM security often focus on individual LLM, but without examining the ecosystem through the lens of LLM systems with other objects (e.g., Frontend, Webtool, Sandbox, and so on). In this paper, we systematically analyze the security of LLM systems, instead of focusing on the individual LLMs. To do so, we build on top of the information flow and formulate the security of LLM systems as constraints on the alignment of the information flow within LLM and between LLM and other objects. Based on this construction and the unique probabilistic nature of LLM, the attack surface of the LLM system can be decomposed into three key components: (1) multi-layer security analysis, (2) analysis of the existence of constraints, and (3) analysis of the robustness of these constraints. To ground this new attack surface, we propose a multi-layer and multi-step approach and apply it to the state-of-art LLM system, OpenAI GPT4. Our investigation exposes several security issues, not just within the LLM model itself but also in its integration with other components. We found that although the OpenAI GPT4 has designed numerous safety constraints to improve its safety features, these safety constraints are still vulnerable to attackers. To further demonstrate the real-world threats of our discovered vulnerabilities, we construct an end-to-end attack where an adversary can illicitly acquire the user's chat history, all without the need to manipulate the user's input or gain direct access to OpenAI GPT4. Our demo is in the link: https://fzwark.github.io/LLM-System-Attack-Demo/



## **19. Model Predictive Control with adaptive resilience for Denial-of-Service Attacks mitigation on a Regulated Dam**

eess.SY

**SubmitDate**: 2024-02-28    [abs](http://arxiv.org/abs/2402.18516v1) [paper-pdf](http://arxiv.org/pdf/2402.18516v1)

**Authors**: Raffaele Giuseppe Cestari, Stefano Longari, Stefano Zanero, Simone Formentin

**Abstract**: In recent years, SCADA (Supervisory Control and Data Acquisition) systems have increasingly become the target of cyber attacks. SCADAs are no longer isolated, as web-based applications expose strategic infrastructures to the outside world connection. In a cyber-warfare context, we propose a Model Predictive Control (MPC) architecture with adaptive resilience, capable of guaranteeing control performance in normal operating conditions and driving towards resilience against DoS (controller-actuator) attacks when needed. Since the attackers' goal is typically to maximize the system damage, we assume they solve an adversarial optimal control problem. An adaptive resilience factor is then designed as a function of the intensity function of a Hawkes process, a point process model estimating the occurrence of random events in time, trained on a moving window to estimate the return time of the next attack. We demonstrate the resulting MPC strategy's effectiveness in 2 attack scenarios on a real system with actual data, the regulated Olginate dam of Lake Como.



## **20. DevPhish: Exploring Social Engineering in Software Supply Chain Attacks on Developers**

cs.SE

7 pages, 2 figures

**SubmitDate**: 2024-02-28    [abs](http://arxiv.org/abs/2402.18401v1) [paper-pdf](http://arxiv.org/pdf/2402.18401v1)

**Authors**: Hossein Siadati, Sima Jafarikhah, Elif Sahin, Terrence Brent Hernandez, Elijah Lorenzo Tripp, Denis Khryashchev

**Abstract**: The Software Supply Chain (SSC) has captured considerable attention from attackers seeking to infiltrate systems and undermine organizations. There is evidence indicating that adversaries utilize Social Engineering (SocE) techniques specifically aimed at software developers. That is, they interact with developers at critical steps in the Software Development Life Cycle (SDLC), such as accessing Github repositories, incorporating code dependencies, and obtaining approval for Pull Requests (PR) to introduce malicious code. This paper aims to comprehensively explore the existing and emerging SocE tactics employed by adversaries to trick Software Engineers (SWEs) into delivering malicious software. By analyzing a diverse range of resources, which encompass established academic literature and real-world incidents, the paper systematically presents an overview of these manipulative strategies within the realm of the SSC. Such insights prove highly beneficial for threat modeling and security gap analysis.



## **21. Neuromorphic Event-Driven Semantic Communication in Microgrids**

cs.ET

The manuscript has been accepted for publication in IEEE Transactions  on Smart Grid

**SubmitDate**: 2024-02-28    [abs](http://arxiv.org/abs/2402.18390v1) [paper-pdf](http://arxiv.org/pdf/2402.18390v1)

**Authors**: Xiaoguang Diao, Yubo Song, Subham Sahoo, Yuan Li

**Abstract**: Synergies between advanced communications, computing and artificial intelligence are unraveling new directions of coordinated operation and resiliency in microgrids. On one hand, coordination among sources is facilitated by distributed, privacy-minded processing at multiple locations, whereas on the other hand, it also creates exogenous data arrival paths for adversaries that can lead to cyber-physical attacks amongst other reliability issues in the communication layer. This long-standing problem necessitates new intrinsic ways of exchanging information between converters through power lines to optimize the system's control performance. Going beyond the existing power and data co-transfer technologies that are limited by efficiency and scalability concerns, this paper proposes neuromorphic learning to implant communicative features using spiking neural networks (SNNs) at each node, which is trained collaboratively in an online manner simply using the power exchanges between the nodes. As opposed to the conventional neuromorphic sensors that operate with spiking signals, we employ an event-driven selective process to collect sparse data for training of SNNs. Finally, its multi-fold effectiveness and reliable performance is validated under simulation conditions with different microgrid topologies and components to establish a new direction in the sense-actuate-compute cycle for power electronic dominated grids and microgrids.



## **22. A Game-theoretic Framework for Privacy-preserving Federated Learning**

cs.LG

**SubmitDate**: 2024-02-28    [abs](http://arxiv.org/abs/2304.05836v3) [paper-pdf](http://arxiv.org/pdf/2304.05836v3)

**Authors**: Xiaojin Zhang, Lixin Fan, Siwei Wang, Wenjie Li, Kai Chen, Qiang Yang

**Abstract**: In federated learning, benign participants aim to optimize a global model collaboratively. However, the risk of \textit{privacy leakage} cannot be ignored in the presence of \textit{semi-honest} adversaries. Existing research has focused either on designing protection mechanisms or on inventing attacking mechanisms. While the battle between defenders and attackers seems never-ending, we are concerned with one critical question: is it possible to prevent potential attacks in advance? To address this, we propose the first game-theoretic framework that considers both FL defenders and attackers in terms of their respective payoffs, which include computational costs, FL model utilities, and privacy leakage risks. We name this game the federated learning privacy game (FLPG), in which neither defenders nor attackers are aware of all participants' payoffs.   To handle the \textit{incomplete information} inherent in this situation, we propose associating the FLPG with an \textit{oracle} that has two primary responsibilities. First, the oracle provides lower and upper bounds of the payoffs for the players. Second, the oracle acts as a correlation device, privately providing suggested actions to each player. With this novel framework, we analyze the optimal strategies of defenders and attackers. Furthermore, we derive and demonstrate conditions under which the attacker, as a rational decision-maker, should always follow the oracle's suggestion \textit{not to attack}.



## **23. Living-off-The-Land Reverse-Shell Detection by Informed Data Augmentation**

cs.CR

**SubmitDate**: 2024-02-28    [abs](http://arxiv.org/abs/2402.18329v1) [paper-pdf](http://arxiv.org/pdf/2402.18329v1)

**Authors**: Dmitrijs Trizna, Luca Demetrio, Battista Biggio, Fabio Roli

**Abstract**: The living-off-the-land (LOTL) offensive methodologies rely on the perpetration of malicious actions through chains of commands executed by legitimate applications, identifiable exclusively by analysis of system logs. LOTL techniques are well hidden inside the stream of events generated by common legitimate activities, moreover threat actors often camouflage activity through obfuscation, making them particularly difficult to detect without incurring in plenty of false alarms, even using machine learning. To improve the performance of models in such an harsh environment, we propose an augmentation framework to enhance and diversify the presence of LOTL malicious activity inside legitimate logs. Guided by threat intelligence, we generate a dataset by injecting attack templates known to be employed in the wild, further enriched by malleable patterns of legitimate activities to replicate the behavior of evasive threat actors. We conduct an extensive ablation study to understand which models better handle our augmented dataset, also manipulated to mimic the presence of model-agnostic evasion and poisoning attacks. Our results suggest that augmentation is needed to maintain high-predictive capabilities, robustness to attack is achieved through specific hardening techniques like adversarial training, and it is possible to deploy near-real-time models with almost-zero false alarms.



## **24. Black-box Adversarial Attacks Against Image Quality Assessment Models**

cs.CV

**SubmitDate**: 2024-02-28    [abs](http://arxiv.org/abs/2402.17533v2) [paper-pdf](http://arxiv.org/pdf/2402.17533v2)

**Authors**: Yu Ran, Ao-Xiang Zhang, Mingjie Li, Weixuan Tang, Yuan-Gen Wang

**Abstract**: The goal of No-Reference Image Quality Assessment (NR-IQA) is to predict the perceptual quality of an image in line with its subjective evaluation. To put the NR-IQA models into practice, it is essential to study their potential loopholes for model refinement. This paper makes the first attempt to explore the black-box adversarial attacks on NR-IQA models. Specifically, we first formulate the attack problem as maximizing the deviation between the estimated quality scores of original and perturbed images, while restricting the perturbed image distortions for visual quality preservation. Under such formulation, we then design a Bi-directional loss function to mislead the estimated quality scores of adversarial examples towards an opposite direction with maximum deviation. On this basis, we finally develop an efficient and effective black-box attack method against NR-IQA models. Extensive experiments reveal that all the evaluated NR-IQA models are vulnerable to the proposed attack method. And the generated perturbations are not transferable, enabling them to serve the investigation of specialities of disparate IQA models.



## **25. Embodied Adversarial Attack: A Dynamic Robust Physical Attack in Autonomous Driving**

cs.CV

10 pages, 7 figures

**SubmitDate**: 2024-02-28    [abs](http://arxiv.org/abs/2312.09554v2) [paper-pdf](http://arxiv.org/pdf/2312.09554v2)

**Authors**: Yitong Sun, Yao Huang, Xingxing Wei

**Abstract**: As physical adversarial attacks become extensively applied in unearthing the potential risk of security-critical scenarios, especially in autonomous driving, their vulnerability to environmental changes has also been brought to light. The non-robust nature of physical adversarial attack methods brings less-than-stable performance consequently. To enhance the robustness of physical adversarial attacks in the real world, instead of statically optimizing a robust adversarial example via an off-line training manner like the existing methods, this paper proposes a brand new robust adversarial attack framework: Embodied Adversarial Attack (EAA) from the perspective of dynamic adaptation, which aims to employ the paradigm of embodied intelligence: Perception-Decision-Control to dynamically adjust the optimal attack strategy according to the current situations in real time. For the perception module, given the challenge of needing simulation for the victim's viewpoint, EAA innovatively devises a Perspective Transformation Network to estimate the target's transformation from the attacker's perspective. For the decision and control module, EAA adopts the laser-a highly manipulable medium to implement physical attacks, and further trains an attack agent with reinforcement learning to make it capable of instantaneously determining the best attack strategy based on the perceived information. Finally, we apply our framework to the autonomous driving scenario. A variety of experiments verify the high effectiveness of our method under complex scenes.



## **26. Towards Transferable Targeted 3D Adversarial Attack in the Physical World**

cs.CV

Accepted by CVPR 2024

**SubmitDate**: 2024-02-28    [abs](http://arxiv.org/abs/2312.09558v2) [paper-pdf](http://arxiv.org/pdf/2312.09558v2)

**Authors**: Yao Huang, Yinpeng Dong, Shouwei Ruan, Xiao Yang, Hang Su, Xingxing Wei

**Abstract**: Compared with transferable untargeted attacks, transferable targeted adversarial attacks could specify the misclassification categories of adversarial samples, posing a greater threat to security-critical tasks. In the meanwhile, 3D adversarial samples, due to their potential of multi-view robustness, can more comprehensively identify weaknesses in existing deep learning systems, possessing great application value. However, the field of transferable targeted 3D adversarial attacks remains vacant. The goal of this work is to develop a more effective technique that could generate transferable targeted 3D adversarial examples, filling the gap in this field. To achieve this goal, we design a novel framework named TT3D that could rapidly reconstruct from few multi-view images into Transferable Targeted 3D textured meshes. While existing mesh-based texture optimization methods compute gradients in the high-dimensional mesh space and easily fall into local optima, leading to unsatisfactory transferability and distinct distortions, TT3D innovatively performs dual optimization towards both feature grid and Multi-layer Perceptron (MLP) parameters in the grid-based NeRF space, which significantly enhances black-box transferability while enjoying naturalness. Experimental results show that TT3D not only exhibits superior cross-model transferability but also maintains considerable adaptability across different renders and vision tasks. More importantly, we produce 3D adversarial examples with 3D printing techniques in the real world and verify their robust performance under various scenarios.



## **27. Exploring Privacy and Fairness Risks in Sharing Diffusion Models: An Adversarial Perspective**

cs.LG

**SubmitDate**: 2024-02-28    [abs](http://arxiv.org/abs/2402.18607v1) [paper-pdf](http://arxiv.org/pdf/2402.18607v1)

**Authors**: Xinjian Luo, Yangfan Jiang, Fei Wei, Yuncheng Wu, Xiaokui Xiao, Beng Chin Ooi

**Abstract**: Diffusion models have recently gained significant attention in both academia and industry due to their impressive generative performance in terms of both sampling quality and distribution coverage. Accordingly, proposals are made for sharing pre-trained diffusion models across different organizations, as a way of improving data utilization while enhancing privacy protection by avoiding sharing private data directly. However, the potential risks associated with such an approach have not been comprehensively examined.   In this paper, we take an adversarial perspective to investigate the potential privacy and fairness risks associated with the sharing of diffusion models. Specifically, we investigate the circumstances in which one party (the sharer) trains a diffusion model using private data and provides another party (the receiver) black-box access to the pre-trained model for downstream tasks. We demonstrate that the sharer can execute fairness poisoning attacks to undermine the receiver's downstream models by manipulating the training data distribution of the diffusion model. Meanwhile, the receiver can perform property inference attacks to reveal the distribution of sensitive features in the sharer's dataset. Our experiments conducted on real-world datasets demonstrate remarkable attack performance on different types of diffusion models, which highlights the critical importance of robust data auditing and privacy protection protocols in pertinent applications.



## **28. The NISQ Complexity of Collision Finding**

quant-ph

40 pages; v2: title changed, major extension to other complexity  models

**SubmitDate**: 2024-02-28    [abs](http://arxiv.org/abs/2211.12954v2) [paper-pdf](http://arxiv.org/pdf/2211.12954v2)

**Authors**: Yassine Hamoudi, Qipeng Liu, Makrand Sinha

**Abstract**: Collision-resistant hashing, a fundamental primitive in modern cryptography, ensures that there is no efficient way to find distinct inputs that produce the same hash value. This property underpins the security of various cryptographic applications, making it crucial to understand its complexity. The complexity of this problem is well-understood in the classical setting and $\Theta(N^{1/2})$ queries are needed to find a collision. However, the advent of quantum computing has introduced new challenges since quantum adversaries $\unicode{x2013}$ equipped with the power of quantum queries $\unicode{x2013}$ can find collisions much more efficiently. Brassard, H\"oyer and Tapp and Aaronson and Shi established that full-scale quantum adversaries require $\Theta(N^{1/3})$ queries to find a collision, prompting a need for longer hash outputs, which impacts efficiency in terms of the key lengths needed for security.   This paper explores the implications of quantum attacks in the Noisy-Intermediate Scale Quantum (NISQ) era. In this work, we investigate three different models for NISQ algorithms and achieve tight bounds for all of them:   (1) A hybrid algorithm making adaptive quantum or classical queries but with a limited quantum query budget, or   (2) A quantum algorithm with access to a noisy oracle, subject to a dephasing or depolarizing channel, or   (3) A hybrid algorithm with an upper bound on its maximum quantum depth; i.e., a classical algorithm aided by low-depth quantum circuits.   In fact, our results handle all regimes between NISQ and full-scale quantum computers. Previously, only results for the pre-image search problem were known for these models by Sun and Zheng, Rosmanis, Chen, Cotler, Huang and Li while nothing was known about the collision finding problem.



## **29. Catastrophic Overfitting: A Potential Blessing in Disguise**

cs.LG

**SubmitDate**: 2024-02-28    [abs](http://arxiv.org/abs/2402.18211v1) [paper-pdf](http://arxiv.org/pdf/2402.18211v1)

**Authors**: Mengnan Zhao, Lihe Zhang, Yuqiu Kong, Baocai Yin

**Abstract**: Fast Adversarial Training (FAT) has gained increasing attention within the research community owing to its efficacy in improving adversarial robustness. Particularly noteworthy is the challenge posed by catastrophic overfitting (CO) in this field. Although existing FAT approaches have made strides in mitigating CO, the ascent of adversarial robustness occurs with a non-negligible decline in classification accuracy on clean samples. To tackle this issue, we initially employ the feature activation differences between clean and adversarial examples to analyze the underlying causes of CO. Intriguingly, our findings reveal that CO can be attributed to the feature coverage induced by a few specific pathways. By intentionally manipulating feature activation differences in these pathways with well-designed regularization terms, we can effectively mitigate and induce CO, providing further evidence for this observation. Notably, models trained stably with these terms exhibit superior performance compared to prior FAT work. On this basis, we harness CO to achieve `attack obfuscation', aiming to bolster model performance. Consequently, the models suffering from CO can attain optimal classification accuracy on both clean and adversarial data when adding random noise to inputs during evaluation. We also validate their robustness against transferred adversarial examples and the necessity of inducing CO to improve robustness. Hence, CO may not be a problem that has to be solved.



## **30. On the Robustness of Bayesian Neural Networks to Adversarial Attacks**

cs.LG

arXiv admin note: text overlap with arXiv:2002.04359

**SubmitDate**: 2024-02-28    [abs](http://arxiv.org/abs/2207.06154v3) [paper-pdf](http://arxiv.org/pdf/2207.06154v3)

**Authors**: Luca Bortolussi, Ginevra Carbone, Luca Laurenti, Andrea Patane, Guido Sanguinetti, Matthew Wicker

**Abstract**: Vulnerability to adversarial attacks is one of the principal hurdles to the adoption of deep learning in safety-critical applications. Despite significant efforts, both practical and theoretical, training deep learning models robust to adversarial attacks is still an open problem. In this paper, we analyse the geometry of adversarial attacks in the large-data, overparameterized limit for Bayesian Neural Networks (BNNs). We show that, in the limit, vulnerability to gradient-based attacks arises as a result of degeneracy in the data distribution, i.e., when the data lies on a lower-dimensional submanifold of the ambient space. As a direct consequence, we demonstrate that in this limit BNN posteriors are robust to gradient-based adversarial attacks. Crucially, we prove that the expected gradient of the loss with respect to the BNN posterior distribution is vanishing, even when each neural network sampled from the posterior is vulnerable to gradient-based attacks. Experimental results on the MNIST, Fashion MNIST, and half moons datasets, representing the finite data regime, with BNNs trained with Hamiltonian Monte Carlo and Variational Inference, support this line of arguments, showing that BNNs can display both high accuracy on clean data and robustness to both gradient-based and gradient-free based adversarial attacks.



## **31. Understanding the Role of Pathways in a Deep Neural Network**

cs.CV

**SubmitDate**: 2024-02-28    [abs](http://arxiv.org/abs/2402.18132v1) [paper-pdf](http://arxiv.org/pdf/2402.18132v1)

**Authors**: Lei Lyu, Chen Pang, Jihua Wang

**Abstract**: Deep neural networks have demonstrated superior performance in artificial intelligence applications, but the opaqueness of their inner working mechanism is one major drawback in their application. The prevailing unit-based interpretation is a statistical observation of stimulus-response data, which fails to show a detailed internal process of inherent mechanisms of neural networks. In this work, we analyze a convolutional neural network (CNN) trained in the classification task and present an algorithm to extract the diffusion pathways of individual pixels to identify the locations of pixels in an input image associated with object classes. The pathways allow us to test the causal components which are important for classification and the pathway-based representations are clearly distinguishable between categories. We find that the few largest pathways of an individual pixel from an image tend to cross the feature maps in each layer that is important for classification. And the large pathways of images of the same category are more consistent in their trends than those of different categories. We also apply the pathways to understanding adversarial attacks, object completion, and movement perception. Further, the total number of pathways on feature maps in all layers can clearly discriminate the original, deformed, and target samples.



## **32. Making Them Ask and Answer: Jailbreaking Large Language Models in Few Queries via Disguise and Reconstruction**

cs.CR

**SubmitDate**: 2024-02-28    [abs](http://arxiv.org/abs/2402.18104v1) [paper-pdf](http://arxiv.org/pdf/2402.18104v1)

**Authors**: Tong Liu, Yingjie Zhang, Zhe Zhao, Yinpeng Dong, Guozhu Meng, Kai Chen

**Abstract**: In recent years, large language models (LLMs) have demonstrated notable success across various tasks, but the trustworthiness of LLMs is still an open problem. One specific threat is the potential to generate toxic or harmful responses. Attackers can craft adversarial prompts that induce harmful responses from LLMs. In this work, we pioneer a theoretical foundation in LLMs security by identifying bias vulnerabilities within the safety fine-tuning and design a black-box jailbreak method named DRA (Disguise and Reconstruction Attack), which conceals harmful instructions through disguise and prompts the model to reconstruct the original harmful instruction within its completion. We evaluate DRA across various open-source and close-source models, showcasing state-of-the-art jailbreak success rates and attack efficiency. Notably, DRA boasts a 90\% attack success rate on LLM chatbots GPT-4.



## **33. Black-box Targeted Adversarial Attack on Segment Anything (SAM)**

cs.CV

**SubmitDate**: 2024-02-28    [abs](http://arxiv.org/abs/2310.10010v2) [paper-pdf](http://arxiv.org/pdf/2310.10010v2)

**Authors**: Sheng Zheng, Chaoning Zhang, Xinhong Hao

**Abstract**: Deep recognition models are widely vulnerable to adversarial examples, which change the model output by adding quasi-imperceptible perturbation to the image input. Recently, Segment Anything Model (SAM) has emerged to become a popular foundation model in computer vision due to its impressive generalization to unseen data and tasks. Realizing flexible attacks on SAM is beneficial for understanding the robustness of SAM in the adversarial context. To this end, this work aims to achieve a targeted adversarial attack (TAA) on SAM. Specifically, under a certain prompt, the goal is to make the predicted mask of an adversarial example resemble that of a given target image. The task of TAA on SAM has been realized in a recent arXiv work in the white-box setup by assuming access to prompt and model, which is thus less practical. To address the issue of prompt dependence, we propose a simple yet effective approach by only attacking the image encoder. Moreover, we propose a novel regularization loss to enhance the cross-model transferability by increasing the feature dominance of adversarial images over random natural images. Extensive experiments verify the effectiveness of our proposed simple techniques to conduct a successful black-box TAA on SAM.



## **34. False Claims against Model Ownership Resolution**

cs.CR

13pages,3 figures. To appear in the 33rd USENIX Security Symposium  (USENIX Security '24)

**SubmitDate**: 2024-02-28    [abs](http://arxiv.org/abs/2304.06607v4) [paper-pdf](http://arxiv.org/pdf/2304.06607v4)

**Authors**: Jian Liu, Rui Zhang, Sebastian Szyller, Kui Ren, N. Asokan

**Abstract**: Deep neural network (DNN) models are valuable intellectual property of model owners, constituting a competitive advantage. Therefore, it is crucial to develop techniques to protect against model theft. Model ownership resolution (MOR) is a class of techniques that can deter model theft. A MOR scheme enables an accuser to assert an ownership claim for a suspect model by presenting evidence, such as a watermark or fingerprint, to show that the suspect model was stolen or derived from a source model owned by the accuser. Most of the existing MOR schemes prioritize robustness against malicious suspects, ensuring that the accuser will win if the suspect model is indeed a stolen model.   In this paper, we show that common MOR schemes in the literature are vulnerable to a different, equally important but insufficiently explored, robustness concern: a malicious accuser. We show how malicious accusers can successfully make false claims against independent suspect models that were not stolen. Our core idea is that a malicious accuser can deviate (without detection) from the specified MOR process by finding (transferable) adversarial examples that successfully serve as evidence against independent suspect models. To this end, we first generalize the procedures of common MOR schemes and show that, under this generalization, defending against false claims is as challenging as preventing (transferable) adversarial examples. Via systematic empirical evaluation, we demonstrate that our false claim attacks always succeed in the MOR schemes that follow our generalization, including against a real-world model: Amazon's Rekognition API.



## **35. Breaking the Black-Box: Confidence-Guided Model Inversion Attack for Distribution Shift**

cs.CV

8pages,5 figures

**SubmitDate**: 2024-02-28    [abs](http://arxiv.org/abs/2402.18027v1) [paper-pdf](http://arxiv.org/pdf/2402.18027v1)

**Authors**: Xinhao Liu, Yingzhao Jiang, Zetao Lin

**Abstract**: Model inversion attacks (MIAs) seek to infer the private training data of a target classifier by generating synthetic images that reflect the characteristics of the target class through querying the model. However, prior studies have relied on full access to the target model, which is not practical in real-world scenarios. Additionally, existing black-box MIAs assume that the image prior and target model follow the same distribution. However, when confronted with diverse data distribution settings, these methods may result in suboptimal performance in conducting attacks. To address these limitations, this paper proposes a \textbf{C}onfidence-\textbf{G}uided \textbf{M}odel \textbf{I}nversion attack method called CG-MI, which utilizes the latent space of a pre-trained publicly available generative adversarial network (GAN) as prior information and gradient-free optimizer, enabling high-resolution MIAs across different data distributions in a black-box setting. Our experiments demonstrate that our method significantly \textbf{outperforms the SOTA black-box MIA by more than 49\% for Celeba and 58\% for Facescrub in different distribution settings}. Furthermore, our method exhibits the ability to generate high-quality images \textbf{comparable to those produced by white-box attacks}. Our method provides a practical and effective solution for black-box model inversion attacks.



## **36. Enhancing Tracking Robustness with Auxiliary Adversarial Defense Networks**

cs.CV

**SubmitDate**: 2024-02-28    [abs](http://arxiv.org/abs/2402.17976v1) [paper-pdf](http://arxiv.org/pdf/2402.17976v1)

**Authors**: Zhewei Wu, Ruilong Yu, Qihe Liu, Shuying Cheng, Shilin Qiu, Shijie Zhou

**Abstract**: Adversarial attacks in visual object tracking have significantly degraded the performance of advanced trackers by introducing imperceptible perturbations into images. These attack methods have garnered considerable attention from researchers in recent years. However, there is still a lack of research on designing adversarial defense methods specifically for visual object tracking. To address these issues, we propose an effective additional pre-processing network called DuaLossDef that eliminates adversarial perturbations during the tracking process. DuaLossDef is deployed ahead of the search branche or template branche of the tracker to apply defensive transformations to the input images. Moreover, it can be seamlessly integrated with other visual trackers as a plug-and-play module without requiring any parameter adjustments. We train DuaLossDef using adversarial training, specifically employing Dua-Loss to generate adversarial samples that simultaneously attack the classification and regression branches of the tracker. Extensive experiments conducted on the OTB100, LaSOT, and VOT2018 benchmarks demonstrate that DuaLossDef maintains excellent defense robustness against adversarial attack methods in both adaptive and non-adaptive attack scenarios. Moreover, when transferring the defense network to other trackers, it exhibits reliable transferability. Finally, DuaLossDef achieves a processing time of up to 5ms/frame, allowing seamless integration with existing high-speed trackers without introducing significant computational overhead. We will make our code publicly available soon.



## **37. LLM-Resistant Math Word Problem Generation via Adversarial Attacks**

cs.CL

Code is available at  https://github.com/ruoyuxie/adversarial_mwps_generation

**SubmitDate**: 2024-02-27    [abs](http://arxiv.org/abs/2402.17916v1) [paper-pdf](http://arxiv.org/pdf/2402.17916v1)

**Authors**: Roy Xie, Chengxuan Huang, Junlin Wang, Bhuwan Dhingra

**Abstract**: Large language models (LLMs) have significantly transformed the educational landscape. As current plagiarism detection tools struggle to keep pace with LLMs' rapid advancements, the educational community faces the challenge of assessing students' true problem-solving abilities in the presence of LLMs. In this work, we explore a new paradigm for ensuring fair evaluation -- generating adversarial examples which preserve the structure and difficulty of the original questions aimed for assessment, but are unsolvable by LLMs. Focusing on the domain of math word problems, we leverage abstract syntax trees to structurally generate adversarial examples that cause LLMs to produce incorrect answers by simply editing the numeric values in the problems. We conduct experiments on various open- and closed-source LLMs, quantitatively and qualitatively demonstrating that our method significantly degrades their math problem-solving ability. We identify shared vulnerabilities among LLMs and propose a cost-effective approach to attack high-cost models. Additionally, we conduct automatic analysis on math problems and investigate the cause of failure to guide future research on LLM's mathematical capability.



## **38. Optimal Zero-Shot Detector for Multi-Armed Attacks**

cs.LG

Accepted to appear in the 27th International Conference on Artificial  Intelligence and Statistics (AISTATS), May 2nd - May 4th, 2024 This article  supersedes arXiv:2302.02216

**SubmitDate**: 2024-02-27    [abs](http://arxiv.org/abs/2402.15808v2) [paper-pdf](http://arxiv.org/pdf/2402.15808v2)

**Authors**: Federica Granese, Marco Romanelli, Pablo Piantanida

**Abstract**: This paper explores a scenario in which a malicious actor employs a multi-armed attack strategy to manipulate data samples, offering them various avenues to introduce noise into the dataset. Our central objective is to protect the data by detecting any alterations to the input. We approach this defensive strategy with utmost caution, operating in an environment where the defender possesses significantly less information compared to the attacker. Specifically, the defender is unable to utilize any data samples for training a defense model or verifying the integrity of the channel. Instead, the defender relies exclusively on a set of pre-existing detectors readily available "off the shelf". To tackle this challenge, we derive an innovative information-theoretic defense approach that optimally aggregates the decisions made by these detectors, eliminating the need for any training data. We further explore a practical use-case scenario for empirical evaluation, where the attacker possesses a pre-trained classifier and launches well-known adversarial attacks against it. Our experiments highlight the effectiveness of our proposed solution, even in scenarios that deviate from the optimal setup.



## **39. Attention-GAN for Anomaly Detection: A Cutting-Edge Approach to Cybersecurity Threat Management**

cs.CR

**SubmitDate**: 2024-02-27    [abs](http://arxiv.org/abs/2402.15945v2) [paper-pdf](http://arxiv.org/pdf/2402.15945v2)

**Authors**: Mohammed Abo Sen

**Abstract**: This paper proposes an innovative Attention-GAN framework for enhancing cybersecurity, focusing on anomaly detection. In response to the challenges posed by the constantly evolving nature of cyber threats, the proposed approach aims to generate diverse and realistic synthetic attack scenarios, thereby enriching the dataset and improving threat identification. Integrating attention mechanisms with Generative Adversarial Networks (GANs) is a key feature of the proposed method. The attention mechanism enhances the model's ability to focus on relevant features, essential for detecting subtle and complex attack patterns. In addition, GANs address the issue of data scarcity by generating additional varied attack data, encompassing known and emerging threats. This dual approach ensures that the system remains relevant and effective against the continuously evolving cyberattacks. The KDD Cup and CICIDS2017 datasets were used to validate this model, which exhibited significant improvements in anomaly detection. It achieved an accuracy of 99.69% on the KDD dataset and 97.93% on the CICIDS2017 dataset, with precision, recall, and F1-scores above 97%, demonstrating its effectiveness in recognizing complex attack patterns. This study contributes significantly to cybersecurity by providing a scalable and adaptable solution for anomaly detection in the face of sophisticated and dynamic cyber threats. The exploration of GANs for data augmentation highlights a promising direction for future research, particularly in situations where data limitations restrict the development of cybersecurity systems. The attention-GAN framework has emerged as a pioneering approach, setting a new benchmark for advanced cyber-defense strategies.



## **40. Follow My Instruction and Spill the Beans: Scalable Data Extraction from Retrieval-Augmented Generation Systems**

cs.CL

**SubmitDate**: 2024-02-27    [abs](http://arxiv.org/abs/2402.17840v1) [paper-pdf](http://arxiv.org/pdf/2402.17840v1)

**Authors**: Zhenting Qi, Hanlin Zhang, Eric Xing, Sham Kakade, Himabindu Lakkaraju

**Abstract**: Retrieval-Augmented Generation (RAG) improves pre-trained models by incorporating external knowledge at test time to enable customized adaptation. We study the risk of datastore leakage in Retrieval-In-Context RAG Language Models (LMs). We show that an adversary can exploit LMs' instruction-following capabilities to easily extract text data verbatim from the datastore of RAG systems built with instruction-tuned LMs via prompt injection. The vulnerability exists for a wide range of modern LMs that span Llama2, Mistral/Mixtral, Vicuna, SOLAR, WizardLM, Qwen1.5, and Platypus2, and the exploitability exacerbates as the model size scales up. Extending our study to production RAG models GPTs, we design an attack that can cause datastore leakage with a 100% success rate on 25 randomly selected customized GPTs with at most 2 queries, and we extract text data verbatim at a rate of 41% from a book of 77,000 words and 3% from a corpus of 1,569,000 words by prompting the GPTs with only 100 queries generated by themselves.



## **41. BarraCUDA: GPUs do Leak DNN Weights**

cs.CR

**SubmitDate**: 2024-02-27    [abs](http://arxiv.org/abs/2312.07783v2) [paper-pdf](http://arxiv.org/pdf/2312.07783v2)

**Authors**: Peter Horvath, Lukasz Chmielewski, Leo Weissbart, Lejla Batina, Yuval Yarom

**Abstract**: Over the last decade, applications of neural networks (NNs) have spread to various aspects of our lives. A large number of companies base their businesses on building products that use neural networks for tasks such as face recognition, machine translation, and self-driving cars. Much of the intellectual property underpinning these products is encoded in the exact parameters of the neural networks. Consequently, protecting these is of utmost priority to businesses. At the same time, many of these products need to operate under a strong threat model, in which the adversary has unfettered physical control of the product. In this work, we present BarraCUDA, a novel attack on general purpose Graphic Processing Units (GPUs) that can extract parameters of neural networks running on the popular Nvidia Jetson Nano device. BarraCUDA uses correlation electromagnetic analysis to recover parameters of real-world convolutional neural networks.



## **42. Extreme Miscalibration and the Illusion of Adversarial Robustness**

cs.CL

**SubmitDate**: 2024-02-27    [abs](http://arxiv.org/abs/2402.17509v1) [paper-pdf](http://arxiv.org/pdf/2402.17509v1)

**Authors**: Vyas Raina, Samson Tan, Volkan Cevher, Aditya Rawal, Sheng Zha, George Karypis

**Abstract**: Deep learning-based Natural Language Processing (NLP) models are vulnerable to adversarial attacks, where small perturbations can cause a model to misclassify. Adversarial Training (AT) is often used to increase model robustness. However, we have discovered an intriguing phenomenon: deliberately or accidentally miscalibrating models masks gradients in a way that interferes with adversarial attack search methods, giving rise to an apparent increase in robustness. We show that this observed gain in robustness is an illusion of robustness (IOR), and demonstrate how an adversary can perform various forms of test-time temperature calibration to nullify the aforementioned interference and allow the adversarial attack to find adversarial examples. Hence, we urge the NLP community to incorporate test-time temperature scaling into their robustness evaluations to ensure that any observed gains are genuine. Finally, we show how the temperature can be scaled during \textit{training} to improve genuine robustness.



## **43. AdvDiff: Generating Unrestricted Adversarial Examples using Diffusion Models**

cs.LG

**SubmitDate**: 2024-02-27    [abs](http://arxiv.org/abs/2307.12499v3) [paper-pdf](http://arxiv.org/pdf/2307.12499v3)

**Authors**: Xuelong Dai, Kaisheng Liang, Bin Xiao

**Abstract**: Unrestricted adversarial attacks present a serious threat to deep learning models and adversarial defense techniques. They pose severe security problems for deep learning applications because they can effectively bypass defense mechanisms. However, previous attack methods often utilize Generative Adversarial Networks (GANs), which are not theoretically provable and thus generate unrealistic examples by incorporating adversarial objectives, especially for large-scale datasets like ImageNet. In this paper, we propose a new method, called AdvDiff, to generate unrestricted adversarial examples with diffusion models. We design two novel adversarial guidance techniques to conduct adversarial sampling in the reverse generation process of diffusion models. These two techniques are effective and stable to generate high-quality, realistic adversarial examples by integrating gradients of the target classifier interpretably. Experimental results on MNIST and ImageNet datasets demonstrate that AdvDiff is effective to generate unrestricted adversarial examples, which outperforms GAN-based methods in terms of attack performance and generation quality.



## **44. Adaptive Perturbation for Adversarial Attack**

cs.CV

Accepted by IEEE Transactions on Pattern Analysis and Machine  Intelligence (TPAMI). 18 pages, 7 figures, 14 tables

**SubmitDate**: 2024-02-27    [abs](http://arxiv.org/abs/2111.13841v3) [paper-pdf](http://arxiv.org/pdf/2111.13841v3)

**Authors**: Zheng Yuan, Jie Zhang, Zhaoyan Jiang, Liangliang Li, Shiguang Shan

**Abstract**: In recent years, the security of deep learning models achieves more and more attentions with the rapid development of neural networks, which are vulnerable to adversarial examples. Almost all existing gradient-based attack methods use the sign function in the generation to meet the requirement of perturbation budget on $L_\infty$ norm. However, we find that the sign function may be improper for generating adversarial examples since it modifies the exact gradient direction. Instead of using the sign function, we propose to directly utilize the exact gradient direction with a scaling factor for generating adversarial perturbations, which improves the attack success rates of adversarial examples even with fewer perturbations. At the same time, we also theoretically prove that this method can achieve better black-box transferability. Moreover, considering that the best scaling factor varies across different images, we propose an adaptive scaling factor generator to seek an appropriate scaling factor for each image, which avoids the computational cost for manually searching the scaling factor. Our method can be integrated with almost all existing gradient-based attack methods to further improve their attack success rates. Extensive experiments on the CIFAR10 and ImageNet datasets show that our method exhibits higher transferability and outperforms the state-of-the-art methods.



## **45. Conformal Shield: A Novel Adversarial Attack Detection Framework for Automatic Modulation Classification**

eess.SP

**SubmitDate**: 2024-02-27    [abs](http://arxiv.org/abs/2402.17450v1) [paper-pdf](http://arxiv.org/pdf/2402.17450v1)

**Authors**: Tailai Wen, Da Ke, Xiang Wang, Zhitao Huang

**Abstract**: Deep learning algorithms have become an essential component in the field of cognitive radio, especially playing a pivotal role in automatic modulation classification. However, Deep learning also present risks and vulnerabilities. Despite their outstanding classification performance, they exhibit fragility when confronted with meticulously crafted adversarial examples, posing potential risks to the reliability of modulation recognition results. Addressing this issue, this letter pioneers the development of an intelligent modulation classification framework based on conformal theory, named the Conformal Shield, aimed at detecting the presence of adversarial examples in unknown signals and assessing the reliability of recognition results. Utilizing conformal mapping from statistical learning theory, introduces a custom-designed Inconsistency Soft-solution Set, enabling multiple validity assessments of the recognition outcomes. Experimental results demonstrate that the Conformal Shield maintains robust detection performance against a variety of typical adversarial sample attacks in the received signals under different perturbation-to-signal power ratio conditions.



## **46. Comparing the Robustness of Modern No-Reference Image- and Video-Quality Metrics to Adversarial Attacks**

cs.CV

**SubmitDate**: 2024-02-27    [abs](http://arxiv.org/abs/2310.06958v4) [paper-pdf](http://arxiv.org/pdf/2310.06958v4)

**Authors**: Anastasia Antsiferova, Khaled Abud, Aleksandr Gushchin, Ekaterina Shumitskaya, Sergey Lavrushkin, Dmitriy Vatolin

**Abstract**: Nowadays, neural-network-based image- and video-quality metrics perform better than traditional methods. However, they also became more vulnerable to adversarial attacks that increase metrics' scores without improving visual quality. The existing benchmarks of quality metrics compare their performance in terms of correlation with subjective quality and calculation time. Nonetheless, the adversarial robustness of image-quality metrics is also an area worth researching. This paper analyses modern metrics' robustness to different adversarial attacks. We adapted adversarial attacks from computer vision tasks and compared attacks' efficiency against 15 no-reference image- and video-quality metrics. Some metrics showed high resistance to adversarial attacks, which makes their usage in benchmarks safer than vulnerable metrics. The benchmark accepts submissions of new metrics for researchers who want to make their metrics more robust to attacks or to find such metrics for their needs. The latest results can be found online: https://videoprocessing.ai/benchmarks/metrics-robustness.html.



## **47. HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal**

cs.LG

Website: https://www.harmbench.org

**SubmitDate**: 2024-02-27    [abs](http://arxiv.org/abs/2402.04249v2) [paper-pdf](http://arxiv.org/pdf/2402.04249v2)

**Authors**: Mantas Mazeika, Long Phan, Xuwang Yin, Andy Zou, Zifan Wang, Norman Mu, Elham Sakhaee, Nathaniel Li, Steven Basart, Bo Li, David Forsyth, Dan Hendrycks

**Abstract**: Automated red teaming holds substantial promise for uncovering and mitigating the risks associated with the malicious use of large language models (LLMs), yet the field lacks a standardized evaluation framework to rigorously assess new methods. To address this issue, we introduce HarmBench, a standardized evaluation framework for automated red teaming. We identify several desirable properties previously unaccounted for in red teaming evaluations and systematically design HarmBench to meet these criteria. Using HarmBench, we conduct a large-scale comparison of 18 red teaming methods and 33 target LLMs and defenses, yielding novel insights. We also introduce a highly efficient adversarial training method that greatly enhances LLM robustness across a wide range of attacks, demonstrating how HarmBench enables codevelopment of attacks and defenses. We open source HarmBench at https://github.com/centerforaisafety/HarmBench.



## **48. Break the Breakout: Reinventing LM Defense Against Jailbreak Attacks with Self-Refinement**

cs.LG

under review

**SubmitDate**: 2024-02-27    [abs](http://arxiv.org/abs/2402.15180v2) [paper-pdf](http://arxiv.org/pdf/2402.15180v2)

**Authors**: Heegyu Kim, Sehyun Yuk, Hyunsouk Cho

**Abstract**: Caution: This paper includes offensive words that could potentially cause unpleasantness. Language models (LMs) are vulnerable to exploitation for adversarial misuse. Training LMs for safety alignment is extensive and makes it hard to respond to fast-developing attacks immediately, such as jailbreaks. We propose self-refine with formatting that achieves outstanding safety even in non-safety-aligned LMs and evaluate our method alongside several defense baselines, demonstrating that it is the safest training-free method against jailbreak attacks. Additionally, we proposed a formatting method that improves the efficiency of the self-refine process while reducing attack success rates in fewer iterations. We've also observed that non-safety-aligned LMs outperform safety-aligned LMs in safety tasks by giving more helpful and safe responses. In conclusion, our findings can achieve less safety risk with fewer computational costs, allowing non-safety LM to be easily utilized in real-world service.



## **49. Adversarial example soups: averaging multiple adversarial examples improves transferability without increasing additional generation time**

cs.CV

16 pages, 8 figures, 12 tables

**SubmitDate**: 2024-02-27    [abs](http://arxiv.org/abs/2402.18370v1) [paper-pdf](http://arxiv.org/pdf/2402.18370v1)

**Authors**: Bo Yang, Hengwei Zhang, Chenwei Li, Jindong Wang

**Abstract**: For transfer-based attacks, the adversarial examples are crafted on the surrogate model, which can be implemented to mislead the target model effectively. The conventional method for maximizing adversarial transferability involves: (1) fine-tuning hyperparameters to generate multiple batches of adversarial examples on the substitute model; (2) conserving the batch of adversarial examples that have the best comprehensive performance on substitute model and target model, and discarding the others. In this work, we revisit the second step of this process in the context of fine-tuning hyperparameters to craft adversarial examples, where multiple batches of fine-tuned adversarial examples often appear in a single high error hilltop. We demonstrate that averaging multiple batches of adversarial examples under different hyperparameter configurations, which refers to as "adversarial example soups", can often enhance adversarial transferability. Compared with traditional methods, the proposed method incurs no additional generation time and computational cost. Besides, our method is orthogonal to existing transfer-based methods and can be combined with them seamlessly to generate more transferable adversarial examples. Extensive experiments on the ImageNet dataset show that our methods achieve a higher attack success rate than the state-of-the-art attacks.



## **50. Dempster-Shafer P-values: Thoughts on an Alternative Approach for Multinomial Inference**

stat.ME

**SubmitDate**: 2024-02-26    [abs](http://arxiv.org/abs/2402.17070v1) [paper-pdf](http://arxiv.org/pdf/2402.17070v1)

**Authors**: Kentaro Hoffman, Kai Zhang, Tyler McCormick, Jan Hannig

**Abstract**: In this paper, we demonstrate that a new measure of evidence we developed called the Dempster-Shafer p-value which allow for insights and interpretations which retain most of the structure of the p-value while covering for some of the disadvantages that traditional p- values face. Moreover, we show through classical large-sample bounds and simulations that there exists a close connection between our form of DS hypothesis testing and the classical frequentist testing paradigm. We also demonstrate how our approach gives unique insights into the dimensionality of a hypothesis test, as well as models the effects of adversarial attacks on multinomial data. Finally, we demonstrate how these insights can be used to analyze text data for public health through an analysis of the Population Health Metrics Research Consortium dataset for verbal autopsies.



