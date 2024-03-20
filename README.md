# Latest Adversarial Attack Papers
**update at 2024-03-20 15:25:50**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. Review of Generative AI Methods in Cybersecurity**

cs.CR

40 pages

**SubmitDate**: 2024-03-19    [abs](http://arxiv.org/abs/2403.08701v2) [paper-pdf](http://arxiv.org/pdf/2403.08701v2)

**Authors**: Yagmur Yigit, William J Buchanan, Madjid G Tehrani, Leandros Maglaras

**Abstract**: Over the last decade, Artificial Intelligence (AI) has become increasingly popular, especially with the use of chatbots such as ChatGPT, Gemini, and DALL-E. With this rise, large language models (LLMs) and Generative AI (GenAI) have also become more prevalent in everyday use. These advancements strengthen cybersecurity's defensive posture and open up new attack avenues for adversaries as well. This paper provides a comprehensive overview of the current state-of-the-art deployments of GenAI, covering assaults, jailbreaking, and applications of prompt injection and reverse psychology. This paper also provides the various applications of GenAI in cybercrimes, such as automated hacking, phishing emails, social engineering, reverse cryptography, creating attack payloads, and creating malware. GenAI can significantly improve the automation of defensive cyber security processes through strategies such as dataset construction, safe code development, threat intelligence, defensive measures, reporting, and cyberattack detection. In this study, we suggest that future research should focus on developing robust ethical norms and innovative defense mechanisms to address the current issues that GenAI creates and to also further encourage an impartial approach to its future application in cybersecurity. Moreover, we underscore the importance of interdisciplinary approaches further to bridge the gap between scientific developments and ethical considerations.



## **2. As Firm As Their Foundations: Can open-sourced foundation models be used to create adversarial examples for downstream tasks?**

cs.CV

**SubmitDate**: 2024-03-19    [abs](http://arxiv.org/abs/2403.12693v1) [paper-pdf](http://arxiv.org/pdf/2403.12693v1)

**Authors**: Anjun Hu, Jindong Gu, Francesco Pinto, Konstantinos Kamnitsas, Philip Torr

**Abstract**: Foundation models pre-trained on web-scale vision-language data, such as CLIP, are widely used as cornerstones of powerful machine learning systems. While pre-training offers clear advantages for downstream learning, it also endows downstream models with shared adversarial vulnerabilities that can be easily identified through the open-sourced foundation model. In this work, we expose such vulnerabilities in CLIP's downstream models and show that foundation models can serve as a basis for attacking their downstream systems. In particular, we propose a simple yet effective adversarial attack strategy termed Patch Representation Misalignment (PRM). Solely based on open-sourced CLIP vision encoders, this method produces adversaries that simultaneously fool more than 20 downstream models spanning 4 common vision-language tasks (semantic segmentation, object detection, image captioning and visual question-answering). Our findings highlight the concerning safety risks introduced by the extensive usage of public foundational models in the development of downstream systems, calling for extra caution in these scenarios.



## **3. Securing Large Language Models: Threats, Vulnerabilities and Responsible Practices**

cs.CR

**SubmitDate**: 2024-03-19    [abs](http://arxiv.org/abs/2403.12503v1) [paper-pdf](http://arxiv.org/pdf/2403.12503v1)

**Authors**: Sara Abdali, Richard Anarfi, CJ Barberan, Jia He

**Abstract**: Large language models (LLMs) have significantly transformed the landscape of Natural Language Processing (NLP). Their impact extends across a diverse spectrum of tasks, revolutionizing how we approach language understanding and generations. Nevertheless, alongside their remarkable utility, LLMs introduce critical security and risk considerations. These challenges warrant careful examination to ensure responsible deployment and safeguard against potential vulnerabilities. This research paper thoroughly investigates security and privacy concerns related to LLMs from five thematic perspectives: security and privacy concerns, vulnerabilities against adversarial attacks, potential harms caused by misuses of LLMs, mitigation strategies to address these challenges while identifying limitations of current strategies. Lastly, the paper recommends promising avenues for future research to enhance the security and risk management of LLMs.



## **4. Boosting Transferability in Vision-Language Attacks via Diversification along the Intersection Region of Adversarial Trajectory**

cs.CV

**SubmitDate**: 2024-03-19    [abs](http://arxiv.org/abs/2403.12445v1) [paper-pdf](http://arxiv.org/pdf/2403.12445v1)

**Authors**: Sensen Gao, Xiaojun Jia, Xuhong Ren, Ivor Tsang, Qing Guo

**Abstract**: Vision-language pre-training (VLP) models exhibit remarkable capabilities in comprehending both images and text, yet they remain susceptible to multimodal adversarial examples (AEs). Strengthening adversarial attacks and uncovering vulnerabilities, especially common issues in VLP models (e.g., high transferable AEs), can stimulate further research on constructing reliable and practical VLP models. A recent work (i.e., Set-level guidance attack) indicates that augmenting image-text pairs to increase AE diversity along the optimization path enhances the transferability of adversarial examples significantly. However, this approach predominantly emphasizes diversity around the online adversarial examples (i.e., AEs in the optimization period), leading to the risk of overfitting the victim model and affecting the transferability. In this study, we posit that the diversity of adversarial examples towards the clean input and online AEs are both pivotal for enhancing transferability across VLP models. Consequently, we propose using diversification along the intersection region of adversarial trajectory to expand the diversity of AEs. To fully leverage the interaction between modalities, we introduce text-guided adversarial example selection during optimization. Furthermore, to further mitigate the potential overfitting, we direct the adversarial text deviating from the last intersection region along the optimization path, rather than adversarial images as in existing methods. Extensive experiments affirm the effectiveness of our method in improving transferability across various VLP models and downstream vision-and-language tasks (e.g., Image-Text Retrieval(ITR), Visual Grounding(VG), Image Captioning(IC)).



## **5. Algorithmic Complexity Attacks on Dynamic Learned Indexes**

cs.DB

VLDB 2024

**SubmitDate**: 2024-03-19    [abs](http://arxiv.org/abs/2403.12433v1) [paper-pdf](http://arxiv.org/pdf/2403.12433v1)

**Authors**: Rui Yang, Evgenios M. Kornaropoulos, Yue Cheng

**Abstract**: Learned Index Structures (LIS) view a sorted index as a model that learns the data distribution, takes a data element key as input, and outputs the predicted position of the key. The original LIS can only handle lookup operations with no support for updates, rendering it impractical to use for typical workloads. To address this limitation, recent studies have focused on designing efficient dynamic learned indexes. ALEX, as the pioneering dynamic learned index structures, enables dynamism by incorporating a series of design choices, including adaptive key space partitioning, dynamic model retraining, and sophisticated engineering and policies that prioritize read/write performance. While these design choices offer improved average-case performance, the emphasis on flexibility and performance increases the attack surface by allowing adversarial behaviors that maximize ALEX's memory space and time complexity in worst-case scenarios. In this work, we present the first systematic investigation of algorithmic complexity attacks (ACAs) targeting the worst-case scenarios of ALEX. We introduce new ACAs that fall into two categories, space ACAs and time ACAs, which target the memory space and time complexity, respectively. First, our space ACA on data nodes exploits ALEX's gapped array layout and uses Multiple-Choice Knapsack (MCK) to generate an optimal adversarial insertion plan for maximizing the memory consumption at the data node level. Second, our space ACA on internal nodes exploits ALEX's catastrophic cost mitigation mechanism, causing an out-of-memory error with only a few hundred adversarial insertions. Third, our time ACA generates pathological insertions to increase the disparity between the actual key distribution and the linear models of data nodes, deteriorating the runtime performance by up to 1,641X compared to ALEX operating under legitimate workloads.



## **6. Electioneering the Network: Dynamic Multi-Step Adversarial Attacks for Community Canvassing**

cs.LG

**SubmitDate**: 2024-03-19    [abs](http://arxiv.org/abs/2403.12399v1) [paper-pdf](http://arxiv.org/pdf/2403.12399v1)

**Authors**: Saurabh Sharma, Ambuj SIngh

**Abstract**: The problem of online social network manipulation for community canvassing is of real concern in today's world. Motivated by the study of voter models, opinion and polarization dynamics on networks, we model community canvassing as a dynamic process over a network enabled via gradient-based attacks on GNNs. Existing attacks on GNNs are all single-step and do not account for the dynamic cascading nature of information diffusion in networks. We consider the realistic scenario where an adversary uses a GNN as a proxy to predict and manipulate voter preferences, especially uncertain voters. Gradient-based attacks on the GNN inform the adversary of strategic manipulations that can be made to proselytize targeted voters. In particular, we explore $\textit{minimum budget attacks for community canvassing}$ (MBACC). We show that the MBACC problem is NP-Hard and propose Dynamic Multi-Step Adversarial Community Canvassing (MAC) to address it. MAC makes dynamic local decisions based on the heuristic of low budget and high second-order influence to convert and perturb target voters. MAC is a dynamic multi-step attack that discovers low-budget and high-influence targets from which efficient cascading attacks can happen. We evaluate MAC against single-step baselines on the MBACC problem with multiple underlying networks and GNN models. Our experiments show the superiority of MAC which is able to discover efficient multi-hop attacks for adversarial community canvassing. Our code implementation and data is available at https://github.com/saurabhsharma1993/mac.



## **7. Securely Fine-tuning Pre-trained Encoders Against Adversarial Examples**

cs.CV

**SubmitDate**: 2024-03-19    [abs](http://arxiv.org/abs/2403.10801v2) [paper-pdf](http://arxiv.org/pdf/2403.10801v2)

**Authors**: Ziqi Zhou, Minghui Li, Wei Liu, Shengshan Hu, Yechao Zhang, Wei Wan, Lulu Xue, Leo Yu Zhang, Dezhong Yao, Hai Jin

**Abstract**: With the evolution of self-supervised learning, the pre-training paradigm has emerged as a predominant solution within the deep learning landscape. Model providers furnish pre-trained encoders designed to function as versatile feature extractors, enabling downstream users to harness the benefits of expansive models with minimal effort through fine-tuning. Nevertheless, recent works have exposed a vulnerability in pre-trained encoders, highlighting their susceptibility to downstream-agnostic adversarial examples (DAEs) meticulously crafted by attackers. The lingering question pertains to the feasibility of fortifying the robustness of downstream models against DAEs, particularly in scenarios where the pre-trained encoders are publicly accessible to the attackers.   In this paper, we initially delve into existing defensive mechanisms against adversarial examples within the pre-training paradigm. Our findings reveal that the failure of current defenses stems from the domain shift between pre-training data and downstream tasks, as well as the sensitivity of encoder parameters. In response to these challenges, we propose Genetic Evolution-Nurtured Adversarial Fine-tuning (Gen-AF), a two-stage adversarial fine-tuning approach aimed at enhancing the robustness of downstream models. Our extensive experiments, conducted across ten self-supervised training methods and six datasets, demonstrate that Gen-AF attains high testing accuracy and robust testing accuracy against state-of-the-art DAEs.



## **8. Improving Visual Quality and Transferability of Adversarial Attacks on Face Recognition Simultaneously with Adversarial Restoration**

cs.CV

\copyright 2023 IEEE. Personal use of this material is permitted.  Permission from IEEE must be obtained for all other uses, in any current or  future media, including reprinting/republishing this material for advertising  or promotional purposes, creating new collective works, for resale or  redistribution to servers or lists, or reuse of any copyrighted component of  this work in other works

**SubmitDate**: 2024-03-19    [abs](http://arxiv.org/abs/2309.01582v4) [paper-pdf](http://arxiv.org/pdf/2309.01582v4)

**Authors**: Fengfan Zhou, Hefei Ling, Yuxuan Shi, Jiazhong Chen, Ping Li

**Abstract**: Adversarial face examples possess two critical properties: Visual Quality and Transferability. However, existing approaches rarely address these properties simultaneously, leading to subpar results. To address this issue, we propose a novel adversarial attack technique known as Adversarial Restoration (AdvRestore), which enhances both visual quality and transferability of adversarial face examples by leveraging a face restoration prior. In our approach, we initially train a Restoration Latent Diffusion Model (RLDM) designed for face restoration. Subsequently, we employ the inference process of RLDM to generate adversarial face examples. The adversarial perturbations are applied to the intermediate features of RLDM. Additionally, by treating RLDM face restoration as a sibling task, the transferability of the generated adversarial face examples is further improved. Our experimental results validate the effectiveness of the proposed attack method.



## **9. Large language models in 6G security: challenges and opportunities**

cs.CR

29 pages, 2 figures

**SubmitDate**: 2024-03-18    [abs](http://arxiv.org/abs/2403.12239v1) [paper-pdf](http://arxiv.org/pdf/2403.12239v1)

**Authors**: Tri Nguyen, Huong Nguyen, Ahmad Ijaz, Saeid Sheikhi, Athanasios V. Vasilakos, Panos Kostakos

**Abstract**: The rapid integration of Generative AI (GenAI) and Large Language Models (LLMs) in sectors such as education and healthcare have marked a significant advancement in technology. However, this growth has also led to a largely unexplored aspect: their security vulnerabilities. As the ecosystem that includes both offline and online models, various tools, browser plugins, and third-party applications continues to expand, it significantly widens the attack surface, thereby escalating the potential for security breaches. These expansions in the 6G and beyond landscape provide new avenues for adversaries to manipulate LLMs for malicious purposes. We focus on the security aspects of LLMs from the viewpoint of potential adversaries. We aim to dissect their objectives and methodologies, providing an in-depth analysis of known security weaknesses. This will include the development of a comprehensive threat taxonomy, categorizing various adversary behaviors. Also, our research will concentrate on how LLMs can be integrated into cybersecurity efforts by defense teams, also known as blue teams. We will explore the potential synergy between LLMs and blockchain technology, and how this combination could lead to the development of next-generation, fully autonomous security solutions. This approach aims to establish a unified cybersecurity strategy across the entire computing continuum, enhancing overall digital security infrastructure.



## **10. Adversarial Training Should Be Cast as a Non-Zero-Sum Game**

cs.LG

**SubmitDate**: 2024-03-18    [abs](http://arxiv.org/abs/2306.11035v2) [paper-pdf](http://arxiv.org/pdf/2306.11035v2)

**Authors**: Alexander Robey, Fabian Latorre, George J. Pappas, Hamed Hassani, Volkan Cevher

**Abstract**: One prominent approach toward resolving the adversarial vulnerability of deep neural networks is the two-player zero-sum paradigm of adversarial training, in which predictors are trained against adversarially chosen perturbations of data. Despite the promise of this approach, algorithms based on this paradigm have not engendered sufficient levels of robustness and suffer from pathological behavior like robust overfitting. To understand this shortcoming, we first show that the commonly used surrogate-based relaxation used in adversarial training algorithms voids all guarantees on the robustness of trained classifiers. The identification of this pitfall informs a novel non-zero-sum bilevel formulation of adversarial training, wherein each player optimizes a different objective function. Our formulation yields a simple algorithmic framework that matches and in some cases outperforms state-of-the-art attacks, attains comparable levels of robustness to standard adversarial training algorithms, and does not suffer from robust overfitting.



## **11. Diffusion Denoising as a Certified Defense against Clean-label Poisoning**

cs.CR

**SubmitDate**: 2024-03-18    [abs](http://arxiv.org/abs/2403.11981v1) [paper-pdf](http://arxiv.org/pdf/2403.11981v1)

**Authors**: Sanghyun Hong, Nicholas Carlini, Alexey Kurakin

**Abstract**: We present a certified defense to clean-label poisoning attacks. These attacks work by injecting a small number of poisoning samples (e.g., 1%) that contain $p$-norm bounded adversarial perturbations into the training data to induce a targeted misclassification of a test-time input. Inspired by the adversarial robustness achieved by $denoised$ $smoothing$, we show how an off-the-shelf diffusion model can sanitize the tampered training data. We extensively test our defense against seven clean-label poisoning attacks and reduce their attack success to 0-16% with only a negligible drop in the test time accuracy. We compare our defense with existing countermeasures against clean-label poisoning, showing that the defense reduces the attack success the most and offers the best model utility. Our results highlight the need for future work on developing stronger clean-label attacks and using our certified yet practical defense as a strong baseline to evaluate these attacks.



## **12. Enhancing the Antidote: Improved Pointwise Certifications against Poisoning Attacks**

cs.LG

**SubmitDate**: 2024-03-18    [abs](http://arxiv.org/abs/2308.07553v2) [paper-pdf](http://arxiv.org/pdf/2308.07553v2)

**Authors**: Shijie Liu, Andrew C. Cullen, Paul Montague, Sarah M. Erfani, Benjamin I. P. Rubinstein

**Abstract**: Poisoning attacks can disproportionately influence model behaviour by making small changes to the training corpus. While defences against specific poisoning attacks do exist, they in general do not provide any guarantees, leaving them potentially countered by novel attacks. In contrast, by examining worst-case behaviours Certified Defences make it possible to provide guarantees of the robustness of a sample against adversarial attacks modifying a finite number of training samples, known as pointwise certification. We achieve this by exploiting both Differential Privacy and the Sampled Gaussian Mechanism to ensure the invariance of prediction for each testing instance against finite numbers of poisoned examples. In doing so, our model provides guarantees of adversarial robustness that are more than twice as large as those provided by prior certifications.



## **13. SSCAE -- Semantic, Syntactic, and Context-aware natural language Adversarial Examples generator**

cs.CL

**SubmitDate**: 2024-03-18    [abs](http://arxiv.org/abs/2403.11833v1) [paper-pdf](http://arxiv.org/pdf/2403.11833v1)

**Authors**: Javad Rafiei Asl, Mohammad H. Rafiei, Manar Alohaly, Daniel Takabi

**Abstract**: Machine learning models are vulnerable to maliciously crafted Adversarial Examples (AEs). Training a machine learning model with AEs improves its robustness and stability against adversarial attacks. It is essential to develop models that produce high-quality AEs. Developing such models has been much slower in natural language processing (NLP) than in areas such as computer vision. This paper introduces a practical and efficient adversarial attack model called SSCAE for \textbf{S}emantic, \textbf{S}yntactic, and \textbf{C}ontext-aware natural language \textbf{AE}s generator. SSCAE identifies important words and uses a masked language model to generate an early set of substitutions. Next, two well-known language models are employed to evaluate the initial set in terms of semantic and syntactic characteristics. We introduce (1) a dynamic threshold to capture more efficient perturbations and (2) a local greedy search to generate high-quality AEs. As a black-box method, SSCAE generates humanly imperceptible and context-aware AEs that preserve semantic consistency and the source language's syntactical and grammatical requirements. The effectiveness and superiority of the proposed SSCAE model are illustrated with fifteen comparative experiments and extensive sensitivity analysis for parameter optimization. SSCAE outperforms the existing models in all experiments while maintaining a higher semantic consistency with a lower query number and a comparable perturbation rate.



## **14. Problem space structural adversarial attacks for Network Intrusion Detection Systems based on Graph Neural Networks**

cs.CR

preprint submitted to IEEE TIFS, under review

**SubmitDate**: 2024-03-18    [abs](http://arxiv.org/abs/2403.11830v1) [paper-pdf](http://arxiv.org/pdf/2403.11830v1)

**Authors**: Andrea Venturi, Dario Stabili, Mirco Marchetti

**Abstract**: Machine Learning (ML) algorithms have become increasingly popular for supporting Network Intrusion Detection Systems (NIDS). Nevertheless, extensive research has shown their vulnerability to adversarial attacks, which involve subtle perturbations to the inputs of the models aimed at compromising their performance. Recent proposals have effectively leveraged Graph Neural Networks (GNN) to produce predictions based also on the structural patterns exhibited by intrusions to enhance the detection robustness. However, the adoption of GNN-based NIDS introduces new types of risks. In this paper, we propose the first formalization of adversarial attacks specifically tailored for GNN in network intrusion detection. Moreover, we outline and model the problem space constraints that attackers need to consider to carry out feasible structural attacks in real-world scenarios. As a final contribution, we conduct an extensive experimental campaign in which we launch the proposed attacks against state-of-the-art GNN-based NIDS. Our findings demonstrate the increased robustness of the models against classical feature-based adversarial attacks, while highlighting their susceptibility to structure-based attacks.



## **15. Expressive Losses for Verified Robustness via Convex Combinations**

cs.LG

ICLR 2024

**SubmitDate**: 2024-03-18    [abs](http://arxiv.org/abs/2305.13991v3) [paper-pdf](http://arxiv.org/pdf/2305.13991v3)

**Authors**: Alessandro De Palma, Rudy Bunel, Krishnamurthy Dvijotham, M. Pawan Kumar, Robert Stanforth, Alessio Lomuscio

**Abstract**: In order to train networks for verified adversarial robustness, it is common to over-approximate the worst-case loss over perturbation regions, resulting in networks that attain verifiability at the expense of standard performance. As shown in recent work, better trade-offs between accuracy and robustness can be obtained by carefully coupling adversarial training with over-approximations. We hypothesize that the expressivity of a loss function, which we formalize as the ability to span a range of trade-offs between lower and upper bounds to the worst-case loss through a single parameter (the over-approximation coefficient), is key to attaining state-of-the-art performance. To support our hypothesis, we show that trivial expressive losses, obtained via convex combinations between adversarial attacks and IBP bounds, yield state-of-the-art results across a variety of settings in spite of their conceptual simplicity. We provide a detailed analysis of the relationship between the over-approximation coefficient and performance profiles across different expressive losses, showing that, while expressivity is essential, better approximations of the worst-case loss are not necessarily linked to superior robustness-accuracy trade-offs.



## **16. Hidden in Plain Sight: Undetectable Adversarial Bias Attacks on Vulnerable Patient Populations**

cs.LG

29 pages, 4 figures

**SubmitDate**: 2024-03-18    [abs](http://arxiv.org/abs/2402.05713v2) [paper-pdf](http://arxiv.org/pdf/2402.05713v2)

**Authors**: Pranav Kulkarni, Andrew Chan, Nithya Navarathna, Skylar Chan, Paul H. Yi, Vishwa S. Parekh

**Abstract**: The proliferation of artificial intelligence (AI) in radiology has shed light on the risk of deep learning (DL) models exacerbating clinical biases towards vulnerable patient populations. While prior literature has focused on quantifying biases exhibited by trained DL models, demographically targeted adversarial bias attacks on DL models and its implication in the clinical environment remains an underexplored field of research in medical imaging. In this work, we demonstrate that demographically targeted label poisoning attacks can introduce undetectable underdiagnosis bias in DL models. Our results across multiple performance metrics and demographic groups like sex, age, and their intersectional subgroups show that adversarial bias attacks demonstrate high-selectivity for bias in the targeted group by degrading group model performance without impacting overall model performance. Furthermore, our results indicate that adversarial bias attacks result in biased DL models that propagate prediction bias even when evaluated with external datasets.



## **17. Stop Reasoning! When Multimodal LLMs with Chain-of-Thought Reasoning Meets Adversarial Images**

cs.CV

**SubmitDate**: 2024-03-18    [abs](http://arxiv.org/abs/2402.14899v2) [paper-pdf](http://arxiv.org/pdf/2402.14899v2)

**Authors**: Zefeng Wang, Zhen Han, Shuo Chen, Fan Xue, Zifeng Ding, Xun Xiao, Volker Tresp, Philip Torr, Jindong Gu

**Abstract**: Recently, Multimodal LLMs (MLLMs) have shown a great ability to understand images. However, like traditional vision models, they are still vulnerable to adversarial images. Meanwhile, Chain-of-Thought (CoT) reasoning has been widely explored on MLLMs, which not only improves model's performance, but also enhances model's explainability by giving intermediate reasoning steps. Nevertheless, there is still a lack of study regarding MLLMs' adversarial robustness with CoT and an understanding of what the rationale looks like when MLLMs infer wrong answers with adversarial images. Our research evaluates the adversarial robustness of MLLMs when employing CoT reasoning, finding that CoT marginally improves adversarial robustness against existing attack methods. Moreover, we introduce a novel stop-reasoning attack technique that effectively bypasses the CoT-induced robustness enhancements. Finally, we demonstrate the alterations in CoT reasoning when MLLMs confront adversarial images, shedding light on their reasoning process under adversarial attacks.



## **18. LocalStyleFool: Regional Video Style Transfer Attack Using Segment Anything Model**

cs.CV

Accepted to 2024 IEEE Security and Privacy Workshops (SPW)

**SubmitDate**: 2024-03-18    [abs](http://arxiv.org/abs/2403.11656v1) [paper-pdf](http://arxiv.org/pdf/2403.11656v1)

**Authors**: Yuxin Cao, Jinghao Li, Xi Xiao, Derui Wang, Minhui Xue, Hao Ge, Wei Liu, Guangwu Hu

**Abstract**: Previous work has shown that well-crafted adversarial perturbations can threaten the security of video recognition systems. Attackers can invade such models with a low query budget when the perturbations are semantic-invariant, such as StyleFool. Despite the query efficiency, the naturalness of the minutia areas still requires amelioration, since StyleFool leverages style transfer to all pixels in each frame. To close the gap, we propose LocalStyleFool, an improved black-box video adversarial attack that superimposes regional style-transfer-based perturbations on videos. Benefiting from the popularity and scalably usability of Segment Anything Model (SAM), we first extract different regions according to semantic information and then track them through the video stream to maintain the temporal consistency. Then, we add style-transfer-based perturbations to several regions selected based on the associative criterion of transfer-based gradient information and regional area. Perturbation fine adjustment is followed to make stylized videos adversarial. We demonstrate that LocalStyleFool can improve both intra-frame and inter-frame naturalness through a human-assessed survey, while maintaining competitive fooling rate and query efficiency. Successful experiments on the high-resolution dataset also showcase that scrupulous segmentation of SAM helps to improve the scalability of adversarial attacks under high-resolution data.



## **19. Zeroth-Order Hard-Thresholding: Gradient Error vs. Expansivity**

cs.LG

Accepted for publication at NeurIPS 2022

**SubmitDate**: 2024-03-18    [abs](http://arxiv.org/abs/2210.05279v2) [paper-pdf](http://arxiv.org/pdf/2210.05279v2)

**Authors**: William de Vazelhes, Hualin Zhang, Huimin Wu, Xiao-Tong Yuan, Bin Gu

**Abstract**: $\ell_0$ constrained optimization is prevalent in machine learning, particularly for high-dimensional problems, because it is a fundamental approach to achieve sparse learning. Hard-thresholding gradient descent is a dominant technique to solve this problem. However, first-order gradients of the objective function may be either unavailable or expensive to calculate in a lot of real-world problems, where zeroth-order (ZO) gradients could be a good surrogate. Unfortunately, whether ZO gradients can work with the hard-thresholding operator is still an unsolved problem. To solve this puzzle, in this paper, we focus on the $\ell_0$ constrained black-box stochastic optimization problems, and propose a new stochastic zeroth-order gradient hard-thresholding (SZOHT) algorithm with a general ZO gradient estimator powered by a novel random support sampling. We provide the convergence analysis of SZOHT under standard assumptions. Importantly, we reveal a conflict between the deviation of ZO estimators and the expansivity of the hard-thresholding operator, and provide a theoretical minimal value of the number of random directions in ZO gradients. In addition, we find that the query complexity of SZOHT is independent or weakly dependent on the dimensionality under different settings. Finally, we illustrate the utility of our method on a portfolio optimization problem as well as black-box adversarial attacks.



## **20. The Lipschitz-Variance-Margin Tradeoff for Enhanced Randomized Smoothing**

cs.LG

**SubmitDate**: 2024-03-18    [abs](http://arxiv.org/abs/2309.16883v4) [paper-pdf](http://arxiv.org/pdf/2309.16883v4)

**Authors**: Blaise Delattre, Alexandre Araujo, Quentin Barthélemy, Alexandre Allauzen

**Abstract**: Real-life applications of deep neural networks are hindered by their unsteady predictions when faced with noisy inputs and adversarial attacks. The certified radius in this context is a crucial indicator of the robustness of models. However how to design an efficient classifier with an associated certified radius? Randomized smoothing provides a promising framework by relying on noise injection into the inputs to obtain a smoothed and robust classifier. In this paper, we first show that the variance introduced by the Monte-Carlo sampling in the randomized smoothing procedure estimate closely interacts with two other important properties of the classifier, \textit{i.e.} its Lipschitz constant and margin. More precisely, our work emphasizes the dual impact of the Lipschitz constant of the base classifier, on both the smoothed classifier and the empirical variance. To increase the certified robust radius, we introduce a different way to convert logits to probability vectors for the base classifier to leverage the variance-margin trade-off. We leverage the use of Bernstein's concentration inequality along with enhanced Lipschitz bounds for randomized smoothing. Experimental results show a significant improvement in certified accuracy compared to current state-of-the-art methods. Our novel certification procedure allows us to use pre-trained models with randomized smoothing, effectively improving the current certification radius in a zero-shot manner.



## **21. SSAP: A Shape-Sensitive Adversarial Patch for Comprehensive Disruption of Monocular Depth Estimation in Autonomous Navigation Applications**

cs.CV

**SubmitDate**: 2024-03-18    [abs](http://arxiv.org/abs/2403.11515v1) [paper-pdf](http://arxiv.org/pdf/2403.11515v1)

**Authors**: Amira Guesmi, Muhammad Abdullah Hanif, Ihsen Alouani, Bassem Ouni, Muhammad Shafique

**Abstract**: Monocular depth estimation (MDE) has advanced significantly, primarily through the integration of convolutional neural networks (CNNs) and more recently, Transformers. However, concerns about their susceptibility to adversarial attacks have emerged, especially in safety-critical domains like autonomous driving and robotic navigation. Existing approaches for assessing CNN-based depth prediction methods have fallen short in inducing comprehensive disruptions to the vision system, often limited to specific local areas. In this paper, we introduce SSAP (Shape-Sensitive Adversarial Patch), a novel approach designed to comprehensively disrupt monocular depth estimation (MDE) in autonomous navigation applications. Our patch is crafted to selectively undermine MDE in two distinct ways: by distorting estimated distances or by creating the illusion of an object disappearing from the system's perspective. Notably, our patch is shape-sensitive, meaning it considers the specific shape and scale of the target object, thereby extending its influence beyond immediate proximity. Furthermore, our patch is trained to effectively address different scales and distances from the camera. Experimental results demonstrate that our approach induces a mean depth estimation error surpassing 0.5, impacting up to 99% of the targeted region for CNN-based MDE models. Additionally, we investigate the vulnerability of Transformer-based MDE models to patch-based attacks, revealing that SSAP yields a significant error of 0.59 and exerts substantial influence over 99% of the target region on these models.



## **22. Robust Overfitting Does Matter: Test-Time Adversarial Purification With FGSM**

cs.CV

CVPR 2024

**SubmitDate**: 2024-03-18    [abs](http://arxiv.org/abs/2403.11448v1) [paper-pdf](http://arxiv.org/pdf/2403.11448v1)

**Authors**: Linyu Tang, Lei Zhang

**Abstract**: Numerous studies have demonstrated the susceptibility of deep neural networks (DNNs) to subtle adversarial perturbations, prompting the development of many advanced adversarial defense methods aimed at mitigating adversarial attacks. Current defense strategies usually train DNNs for a specific adversarial attack method and can achieve good robustness in defense against this type of adversarial attack. Nevertheless, when subjected to evaluations involving unfamiliar attack modalities, empirical evidence reveals a pronounced deterioration in the robustness of DNNs. Meanwhile, there is a trade-off between the classification accuracy of clean examples and adversarial examples. Most defense methods often sacrifice the accuracy of clean examples in order to improve the adversarial robustness of DNNs. To alleviate these problems and enhance the overall robust generalization of DNNs, we propose the Test-Time Pixel-Level Adversarial Purification (TPAP) method. This approach is based on the robust overfitting characteristic of DNNs to the fast gradient sign method (FGSM) on training and test datasets. It utilizes FGSM for adversarial purification, to process images for purifying unknown adversarial perturbations from pixels at testing time in a "counter changes with changelessness" manner, thereby enhancing the defense capability of DNNs against various unknown adversarial attacks. Extensive experimental results show that our method can effectively improve both overall robust generalization of DNNs, notably over previous methods.



## **23. Defense Against Adversarial Attacks on No-Reference Image Quality Models with Gradient Norm Regularization**

cs.CV

accepted by CVPR 2024

**SubmitDate**: 2024-03-18    [abs](http://arxiv.org/abs/2403.11397v1) [paper-pdf](http://arxiv.org/pdf/2403.11397v1)

**Authors**: Yujia Liu, Chenxi Yang, Dingquan Li, Jianhao Ding, Tingting Jiang

**Abstract**: The task of No-Reference Image Quality Assessment (NR-IQA) is to estimate the quality score of an input image without additional information. NR-IQA models play a crucial role in the media industry, aiding in performance evaluation and optimization guidance. However, these models are found to be vulnerable to adversarial attacks, which introduce imperceptible perturbations to input images, resulting in significant changes in predicted scores. In this paper, we propose a defense method to improve the stability in predicted scores when attacked by small perturbations, thus enhancing the adversarial robustness of NR-IQA models. To be specific, we present theoretical evidence showing that the magnitude of score changes is related to the $\ell_1$ norm of the model's gradient with respect to the input image. Building upon this theoretical foundation, we propose a norm regularization training strategy aimed at reducing the $\ell_1$ norm of the gradient, thereby boosting the robustness of NR-IQA models. Experiments conducted on four NR-IQA baseline models demonstrate the effectiveness of our strategy in reducing score changes in the presence of adversarial attacks. To the best of our knowledge, this work marks the first attempt to defend against adversarial attacks on NR-IQA models. Our study offers valuable insights into the adversarial robustness of NR-IQA models and provides a foundation for future research in this area.



## **24. A Modified Word Saliency-Based Adversarial Attack on Text Classification Models**

cs.CL

The paper is a preprint of a version submitted in ICCIDA 2024. It  consists of 10 pages and contains 7 tables

**SubmitDate**: 2024-03-17    [abs](http://arxiv.org/abs/2403.11297v1) [paper-pdf](http://arxiv.org/pdf/2403.11297v1)

**Authors**: Hetvi Waghela, Sneha Rakshit, Jaydip Sen

**Abstract**: This paper introduces a novel adversarial attack method targeting text classification models, termed the Modified Word Saliency-based Adversarial At-tack (MWSAA). The technique builds upon the concept of word saliency to strategically perturb input texts, aiming to mislead classification models while preserving semantic coherence. By refining the traditional adversarial attack approach, MWSAA significantly enhances its efficacy in evading detection by classification systems. The methodology involves first identifying salient words in the input text through a saliency estimation process, which prioritizes words most influential to the model's decision-making process. Subsequently, these salient words are subjected to carefully crafted modifications, guided by semantic similarity metrics to ensure that the altered text remains coherent and retains its original meaning. Empirical evaluations conducted on diverse text classification datasets demonstrate the effectiveness of the proposed method in generating adversarial examples capable of successfully deceiving state-of-the-art classification models. Comparative analyses with existing adversarial attack techniques further indicate the superiority of the proposed approach in terms of both attack success rate and preservation of text coherence.



## **25. Forging the Forger: An Attempt to Improve Authorship Verification via Data Augmentation**

cs.LG

**SubmitDate**: 2024-03-17    [abs](http://arxiv.org/abs/2403.11265v1) [paper-pdf](http://arxiv.org/pdf/2403.11265v1)

**Authors**: Silvia Corbara, Alejandro Moreo

**Abstract**: Authorship Verification (AV) is a text classification task concerned with inferring whether a candidate text has been written by one specific author or by someone else. It has been shown that many AV systems are vulnerable to adversarial attacks, where a malicious author actively tries to fool the classifier by either concealing their writing style, or by imitating the style of another author. In this paper, we investigate the potential benefits of augmenting the classifier training set with (negative) synthetic examples. These synthetic examples are generated to imitate the style of the author of interest. We analyze the improvements in classifier prediction that this augmentation brings to bear in the task of AV in an adversarial setting. In particular, we experiment with three different generator architectures (one based on Recurrent Neural Networks, another based on small-scale transformers, and another based on the popular GPT model) and with two training strategies (one inspired by standard Language Models, and another inspired by Wasserstein Generative Adversarial Networks). We evaluate our hypothesis on five datasets (three of which have been specifically collected to represent an adversarial setting) and using two learning algorithms for the AV classifier (Support Vector Machines and Convolutional Neural Networks). This experimentation has yielded negative results, revealing that, although our methodology proves effective in many adversarial settings, its benefits are too sporadic for a pragmatical application.



## **26. A Tip for IOTA Privacy: IOTA Light Node Deanonymization via Tip Selection**

cs.CR

This paper is accepted to the IEEE International Conference on  Blockchain and Cryptocurrency(ICBC) 2024

**SubmitDate**: 2024-03-17    [abs](http://arxiv.org/abs/2403.11171v1) [paper-pdf](http://arxiv.org/pdf/2403.11171v1)

**Authors**: Hojung Yang, Suhyeon Lee, Seungjoo Kim

**Abstract**: IOTA is a distributed ledger technology that uses a Directed Acyclic Graph (DAG) structure called the Tangle. It is known for its efficiency and is widely used in the Internet of Things (IoT) environment. Tangle can be configured by utilizing the tip selection process. Due to performance issues with light nodes, full nodes are being asked to perform the tip selections of light nodes. However, in this paper, we demonstrate that tip selection can be exploited to compromise users' privacy. An adversary full node can associate a transaction with the identity of a light node by comparing the light node's request with its ledger. We show that these types of attacks are not only viable in the current IOTA environment but also in IOTA 2.0 and the privacy improvement being studied. We also provide solutions to mitigate these attacks and propose ways to enhance anonymity in the IOTA network while maintaining efficiency and scalability.



## **27. PubDef: Defending Against Transfer Attacks From Public Models**

cs.LG

ICLR 2024. Code available at https://github.com/wagner-group/pubdef

**SubmitDate**: 2024-03-17    [abs](http://arxiv.org/abs/2310.17645v2) [paper-pdf](http://arxiv.org/pdf/2310.17645v2)

**Authors**: Chawin Sitawarin, Jaewon Chang, David Huang, Wesson Altoyan, David Wagner

**Abstract**: Adversarial attacks have been a looming and unaddressed threat in the industry. However, through a decade-long history of the robustness evaluation literature, we have learned that mounting a strong or optimal attack is challenging. It requires both machine learning and domain expertise. In other words, the white-box threat model, religiously assumed by a large majority of the past literature, is unrealistic. In this paper, we propose a new practical threat model where the adversary relies on transfer attacks through publicly available surrogate models. We argue that this setting will become the most prevalent for security-sensitive applications in the future. We evaluate the transfer attacks in this setting and propose a specialized defense method based on a game-theoretic perspective. The defenses are evaluated under 24 public models and 11 attack algorithms across three datasets (CIFAR-10, CIFAR-100, and ImageNet). Under this threat model, our defense, PubDef, outperforms the state-of-the-art white-box adversarial training by a large margin with almost no loss in the normal accuracy. For instance, on ImageNet, our defense achieves 62% accuracy under the strongest transfer attack vs only 36% of the best adversarially trained model. Its accuracy when not under attack is only 2% lower than that of an undefended model (78% vs 80%). We release our code at https://github.com/wagner-group/pubdef.



## **28. RobustSentEmbed: Robust Sentence Embeddings Using Adversarial Self-Supervised Contrastive Learning**

cs.CL

Accepted at the Annual Conference of the North American Chapter of  the Association for Computational Linguistics (NAACL Findings) 2024.  [https://openreview.net/forum?id=9dEAg4lJEA]

**SubmitDate**: 2024-03-17    [abs](http://arxiv.org/abs/2403.11082v1) [paper-pdf](http://arxiv.org/pdf/2403.11082v1)

**Authors**: Javad Rafiei Asl, Prajwal Panzade, Eduardo Blanco, Daniel Takabi, Zhipeng Cai

**Abstract**: Pre-trained language models (PLMs) have consistently demonstrated outstanding performance across a diverse spectrum of natural language processing tasks. Nevertheless, despite their success with unseen data, current PLM-based representations often exhibit poor robustness in adversarial settings. In this paper, we introduce RobustSentEmbed, a self-supervised sentence embedding framework designed to improve both generalization and robustness in diverse text representation tasks and against a diverse set of adversarial attacks. Through the generation of high-risk adversarial perturbations and their utilization in a novel objective function, RobustSentEmbed adeptly learns high-quality and robust sentence embeddings. Our experiments confirm the superiority of RobustSentEmbed over state-of-the-art representations. Specifically, Our framework achieves a significant reduction in the success rate of various adversarial attacks, notably reducing the BERTAttack success rate by almost half (from 75.51\% to 38.81\%). The framework also yields improvements of 1.59\% and 0.23\% in semantic textual similarity tasks and various transfer tasks, respectively.



## **29. Instance-Level Trojan Attacks on Visual Question Answering via Adversarial Learning in Neuron Activation Space**

cs.CV

Accepted for IJCNN 2024

**SubmitDate**: 2024-03-17    [abs](http://arxiv.org/abs/2304.00436v2) [paper-pdf](http://arxiv.org/pdf/2304.00436v2)

**Authors**: Yuwei Sun, Hideya Ochiai, Jun Sakuma

**Abstract**: Trojan attacks embed perturbations in input data leading to malicious behavior in neural network models. A combination of various Trojans in different modalities enables an adversary to mount a sophisticated attack on multimodal learning such as Visual Question Answering (VQA). However, multimodal Trojans in conventional methods are susceptible to parameter adjustment during processes such as fine-tuning. To this end, we propose an instance-level multimodal Trojan attack on VQA that efficiently adapts to fine-tuned models through a dual-modality adversarial learning method. This method compromises two specific neurons in a specific perturbation layer in the pretrained model to produce overly large neuron activations. Then, a malicious correlation between these overactive neurons and the malicious output of a fine-tuned model is established through adversarial learning. Extensive experiments are conducted using the VQA-v2 dataset, based on a wide range of metrics including sample efficiency, stealthiness, and robustness. The proposed attack demonstrates enhanced performance with diverse vision and text Trojans tailored for each sample. We demonstrate that the proposed attack can be efficiently adapted to different fine-tuned models, by injecting only a few shots of Trojan samples. Moreover, we investigate the attack performance under conventional defenses, where the defenses cannot effectively mitigate the attack.



## **30. Fast Inference of Removal-Based Node Influence**

cs.LG

To be published in the Web Conference 2024

**SubmitDate**: 2024-03-16    [abs](http://arxiv.org/abs/2403.08333v2) [paper-pdf](http://arxiv.org/pdf/2403.08333v2)

**Authors**: Weikai Li, Zhiping Xiao, Xiao Luo, Yizhou Sun

**Abstract**: Graph neural networks (GNNs) are widely utilized to capture the information spreading patterns in graphs. While remarkable performance has been achieved, there is a new trending topic of evaluating node influence. We propose a new method of evaluating node influence, which measures the prediction change of a trained GNN model caused by removing a node. A real-world application is, "In the task of predicting Twitter accounts' polarity, had a particular account been removed, how would others' polarity change?". We use the GNN as a surrogate model whose prediction could simulate the change of nodes or edges caused by node removal. Our target is to obtain the influence score for every node, and a straightforward way is to alternately remove every node and apply the trained GNN on the modified graph to generate new predictions. It is reliable but time-consuming, so we need an efficient method. The related lines of work, such as graph adversarial attack and counterfactual explanation, cannot directly satisfy our needs, since their problem settings are different. We propose an efficient, intuitive, and effective method, NOde-Removal-based fAst GNN inference (NORA), which uses the gradient information to approximate the node-removal influence. It only costs one forward propagation and one backpropagation to approximate the influence score for all nodes. Extensive experiments on six datasets and six GNN models verify the effectiveness of NORA. Our code is available at https://github.com/weikai-li/NORA.git.



## **31. Understanding Robustness of Visual State Space Models for Image Classification**

cs.CV

27 pages

**SubmitDate**: 2024-03-16    [abs](http://arxiv.org/abs/2403.10935v1) [paper-pdf](http://arxiv.org/pdf/2403.10935v1)

**Authors**: Chengbin Du, Yanxi Li, Chang Xu

**Abstract**: Visual State Space Model (VMamba) has recently emerged as a promising architecture, exhibiting remarkable performance in various computer vision tasks. However, its robustness has not yet been thoroughly studied. In this paper, we delve into the robustness of this architecture through comprehensive investigations from multiple perspectives. Firstly, we investigate its robustness to adversarial attacks, employing both whole-image and patch-specific adversarial attacks. Results demonstrate superior adversarial robustness compared to Transformer architectures while revealing scalability weaknesses. Secondly, the general robustness of VMamba is assessed against diverse scenarios, including natural adversarial examples, out-of-distribution data, and common corruptions. VMamba exhibits exceptional generalizability with out-of-distribution data but shows scalability weaknesses against natural adversarial examples and common corruptions. Additionally, we explore VMamba's gradients and back-propagation during white-box attacks, uncovering unique vulnerabilities and defensive capabilities of its novel components. Lastly, the sensitivity of VMamba to image structure variations is examined, highlighting vulnerabilities associated with the distribution of disturbance areas and spatial information, with increased susceptibility closer to the image center. Through these comprehensive studies, we contribute to a deeper understanding of VMamba's robustness, providing valuable insights for refining and advancing the capabilities of deep neural networks in computer vision applications.



## **32. Adversarial Examples are Misaligned in Diffusion Model Manifolds**

cs.CV

accepted at IJCNN

**SubmitDate**: 2024-03-16    [abs](http://arxiv.org/abs/2401.06637v5) [paper-pdf](http://arxiv.org/pdf/2401.06637v5)

**Authors**: Peter Lorenz, Ricard Durall, Janis Keuper

**Abstract**: In recent years, diffusion models (DMs) have drawn significant attention for their success in approximating data distributions, yielding state-of-the-art generative results. Nevertheless, the versatility of these models extends beyond their generative capabilities to encompass various vision applications, such as image inpainting, segmentation, adversarial robustness, among others. This study is dedicated to the investigation of adversarial attacks through the lens of diffusion models. However, our objective does not involve enhancing the adversarial robustness of image classifiers. Instead, our focus lies in utilizing the diffusion model to detect and analyze the anomalies introduced by these attacks on images. To that end, we systematically examine the alignment of the distributions of adversarial examples when subjected to the process of transformation using diffusion models. The efficacy of this approach is assessed across CIFAR-10 and ImageNet datasets, including varying image sizes in the latter. The results demonstrate a notable capacity to discriminate effectively between benign and attacked images, providing compelling evidence that adversarial instances do not align with the learned manifold of the DMs.



## **33. Improving Adversarial Transferability of Visual-Language Pre-training Models through Collaborative Multimodal Interaction**

cs.CV

**SubmitDate**: 2024-03-16    [abs](http://arxiv.org/abs/2403.10883v1) [paper-pdf](http://arxiv.org/pdf/2403.10883v1)

**Authors**: Jiyuan Fu, Zhaoyu Chen, Kaixun Jiang, Haijing Guo, Jiafeng Wang, Shuyong Gao, Wenqiang Zhang

**Abstract**: Despite the substantial advancements in Vision-Language Pre-training (VLP) models, their susceptibility to adversarial attacks poses a significant challenge. Existing work rarely studies the transferability of attacks on VLP models, resulting in a substantial performance gap from white-box attacks. We observe that prior work overlooks the interaction mechanisms between modalities, which plays a crucial role in understanding the intricacies of VLP models. In response, we propose a novel attack, called Collaborative Multimodal Interaction Attack (CMI-Attack), leveraging modality interaction through embedding guidance and interaction enhancement. Specifically, attacking text at the embedding level while preserving semantics, as well as utilizing interaction image gradients to enhance constraints on perturbations of texts and images. Significantly, in the image-text retrieval task on Flickr30K dataset, CMI-Attack raises the transfer success rates from ALBEF to TCL, $\text{CLIP}_{\text{ViT}}$ and $\text{CLIP}_{\text{CNN}}$ by 8.11%-16.75% over state-of-the-art methods. Moreover, CMI-Attack also demonstrates superior performance in cross-task generalization scenarios. Our work addresses the underexplored realm of transfer attacks on VLP models, shedding light on the importance of modality interaction for enhanced adversarial robustness.



## **34. TTPXHunter: Actionable Threat Intelligence Extraction as TTPs from Finished Cyber Threat Reports**

cs.CR

Submitted to Journal of Information Security and Applications (JISA)

**SubmitDate**: 2024-03-16    [abs](http://arxiv.org/abs/2403.03267v2) [paper-pdf](http://arxiv.org/pdf/2403.03267v2)

**Authors**: Nanda Rani, Bikash Saha, Vikas Maurya, Sandeep Kumar Shukla

**Abstract**: Understanding the modus operandi of adversaries aids organizations in employing efficient defensive strategies and sharing intelligence in the community. This knowledge is often present in unstructured natural language text within threat analysis reports. A translation tool is needed to interpret the modus operandi explained in the sentences of the threat report and translate it into a structured format. This research introduces a methodology named TTPXHunter for the automated extraction of threat intelligence in terms of Tactics, Techniques, and Procedures (TTPs) from finished cyber threat reports. It leverages cyber domain-specific state-of-the-art natural language processing (NLP) to augment sentences for minority class TTPs and refine pinpointing the TTPs in threat analysis reports significantly. The knowledge of threat intelligence in terms of TTPs is essential for comprehensively understanding cyber threats and enhancing detection and mitigation strategies. We create two datasets: an augmented sentence-TTP dataset of 39,296 samples and a 149 real-world cyber threat intelligence report-to-TTP dataset. Further, we evaluate TTPXHunter on the augmented sentence dataset and the cyber threat reports. The TTPXHunter achieves the highest performance of 92.42% f1-score on the augmented dataset, and it also outperforms existing state-of-the-art solutions in TTP extraction by achieving an f1-score of 97.09% when evaluated over the report dataset. TTPXHunter significantly improves cybersecurity threat intelligence by offering quick, actionable insights into attacker behaviors. This advancement automates threat intelligence analysis, providing a crucial tool for cybersecurity professionals fighting cyber threats.



## **35. Enhancing Adversarial Training with Prior Knowledge Distillation for Robust Image Compression**

eess.IV

**SubmitDate**: 2024-03-16    [abs](http://arxiv.org/abs/2403.06700v2) [paper-pdf](http://arxiv.org/pdf/2403.06700v2)

**Authors**: Zhi Cao, Youneng Bao, Fanyang Meng, Chao Li, Wen Tan, Genhong Wang, Yongsheng Liang

**Abstract**: Deep neural network-based image compression (NIC) has achieved excellent performance, but NIC method models have been shown to be susceptible to backdoor attacks. Adversarial training has been validated in image compression models as a common method to enhance model robustness. However, the improvement effect of adversarial training on model robustness is limited. In this paper, we propose a prior knowledge-guided adversarial training framework for image compression models. Specifically, first, we propose a gradient regularization constraint for training robust teacher models. Subsequently, we design a knowledge distillation based strategy to generate a priori knowledge from the teacher model to the student model for guiding adversarial training. Experimental results show that our method improves the reconstruction quality by about 9dB when the Kodak dataset is elected as the backdoor attack object for psnr attack. Compared with Ma2023, our method has a 5dB higher PSNR output at high bitrate points.



## **36. Bergeron: Combating Adversarial Attacks through a Conscience-Based Alignment Framework**

cs.CR

**SubmitDate**: 2024-03-15    [abs](http://arxiv.org/abs/2312.00029v2) [paper-pdf](http://arxiv.org/pdf/2312.00029v2)

**Authors**: Matthew Pisano, Peter Ly, Abraham Sanders, Bingsheng Yao, Dakuo Wang, Tomek Strzalkowski, Mei Si

**Abstract**: Research into AI alignment has grown considerably since the recent introduction of increasingly capable Large Language Models (LLMs). Unfortunately, modern methods of alignment still fail to fully prevent harmful responses when models are deliberately attacked. These attacks can trick seemingly aligned models into giving manufacturing instructions for dangerous materials, inciting violence, or recommending other immoral acts. To help mitigate this issue, we introduce Bergeron: a framework designed to improve the robustness of LLMs against attacks without any additional parameter fine-tuning. Bergeron is organized into two tiers; with a secondary LLM emulating the conscience of a protected, primary LLM. This framework better safeguards the primary model against incoming attacks while monitoring its output for any harmful content. Empirical analysis shows that, by using Bergeron to complement models with existing alignment training, we can improve the robustness and safety of multiple, commonly used commercial and open-source LLMs.



## **37. Not Just Change the Labels, Learn the Features: Watermarking Deep Neural Networks with Multi-View Data**

cs.CR

**SubmitDate**: 2024-03-15    [abs](http://arxiv.org/abs/2403.10663v1) [paper-pdf](http://arxiv.org/pdf/2403.10663v1)

**Authors**: Yuxuan Li, Sarthak Kumar Maharana, Yunhui Guo

**Abstract**: With the increasing prevalence of Machine Learning as a Service (MLaaS) platforms, there is a growing focus on deep neural network (DNN) watermarking techniques. These methods are used to facilitate the verification of ownership for a target DNN model to protect intellectual property. One of the most widely employed watermarking techniques involves embedding a trigger set into the source model. Unfortunately, existing methodologies based on trigger sets are still susceptible to functionality-stealing attacks, potentially enabling adversaries to steal the functionality of the source model without a reliable means of verifying ownership. In this paper, we first introduce a novel perspective on trigger set-based watermarking methods from a feature learning perspective. Specifically, we demonstrate that by selecting data exhibiting multiple features, also referred to as $\textit{multi-view data}$, it becomes feasible to effectively defend functionality stealing attacks. Based on this perspective, we introduce a novel watermarking technique based on Multi-view dATa, called MAT, for efficiently embedding watermarks within DNNs. This approach involves constructing a trigger set with multi-view data and incorporating a simple feature-based regularization method for training the source model. We validate our method across various benchmarks and demonstrate its efficacy in defending against model extraction attacks, surpassing relevant baselines by a significant margin.



## **38. Benchmarking Zero-Shot Robustness of Multimodal Foundation Models: A Pilot Study**

cs.LG

**SubmitDate**: 2024-03-15    [abs](http://arxiv.org/abs/2403.10499v1) [paper-pdf](http://arxiv.org/pdf/2403.10499v1)

**Authors**: Chenguang Wang, Ruoxi Jia, Xin Liu, Dawn Song

**Abstract**: Pre-training image representations from the raw text about images enables zero-shot vision transfer to downstream tasks. Through pre-training on millions of samples collected from the internet, multimodal foundation models, such as CLIP, produce state-of-the-art zero-shot results that often reach competitiveness with fully supervised methods without the need for task-specific training. Besides the encouraging performance on classification accuracy, it is reported that these models close the robustness gap by matching the performance of supervised models trained on ImageNet under natural distribution shift. Because robustness is critical to real-world applications, especially safety-critical ones, in this paper, we present a comprehensive evaluation based on a large-scale robustness benchmark covering 7 natural, 3 synthetic distribution shifts, and 11 adversarial attacks. We use CLIP as a pilot study. We show that CLIP leads to a significant robustness drop compared to supervised ImageNet models on our benchmark, especially under synthetic distribution shift and adversarial attacks. Furthermore, data overlap analysis suggests that the observed robustness under natural distribution shifts could be attributed, at least in part, to data overlap. In summary, our evaluation shows a comprehensive evaluation of robustness is necessary; and there is a significant need to improve the robustness of zero-shot multimodal models.



## **39. Mitigating Dialogue Hallucination for Large Multi-modal Models via Adversarial Instruction Tuning**

cs.CV

**SubmitDate**: 2024-03-15    [abs](http://arxiv.org/abs/2403.10492v1) [paper-pdf](http://arxiv.org/pdf/2403.10492v1)

**Authors**: Dongmin Park, Zhaofang Qian, Guangxing Han, Ser-Nam Lim

**Abstract**: Mitigating hallucinations of Large Multi-modal Models(LMMs) is crucial to enhance their reliability for general-purpose assistants. This paper shows that such hallucinations of LMMs can be significantly exacerbated by preceding user-system dialogues. To precisely measure this, we first present an evaluation benchmark by extending popular multi-modal benchmark datasets with prepended hallucinatory dialogues generated by our novel Adversarial Question Generator, which can automatically generate image-related yet adversarial dialogues by adopting adversarial attacks on LMMs. On our benchmark, the zero-shot performance of state-of-the-art LMMs dropped significantly for both the VQA and Captioning tasks. Next, we further reveal this hallucination is mainly due to the prediction bias toward preceding dialogues rather than visual content. To reduce this bias, we propose Adversarial Instruction Tuning that robustly fine-tunes LMMs on augmented multi-modal instruction-following datasets with hallucinatory dialogues. Extensive experiments show that our proposed approach successfully reduces dialogue hallucination while maintaining or even improving performance.



## **40. Introducing Adaptive Continuous Adversarial Training (ACAT) to Enhance ML Robustness**

cs.LG

**SubmitDate**: 2024-03-15    [abs](http://arxiv.org/abs/2403.10461v1) [paper-pdf](http://arxiv.org/pdf/2403.10461v1)

**Authors**: Mohamed elShehaby, Aditya Kotha, Ashraf Matrawy

**Abstract**: Machine Learning (ML) is susceptible to adversarial attacks that aim to trick ML models, making them produce faulty predictions. Adversarial training was found to increase the robustness of ML models against these attacks. However, in network and cybersecurity, obtaining labeled training and adversarial training data is challenging and costly. Furthermore, concept drift deepens the challenge, particularly in dynamic domains like network and cybersecurity, and requires various models to conduct periodic retraining. This letter introduces Adaptive Continuous Adversarial Training (ACAT) to continuously integrate adversarial training samples into the model during ongoing learning sessions, using real-world detected adversarial data, to enhance model resilience against evolving adversarial threats. ACAT is an adaptive defense mechanism that utilizes periodic retraining to effectively counter adversarial attacks while mitigating catastrophic forgetting. Our approach also reduces the total time required for adversarial sample detection, especially in environments such as network security where the rate of attacks could be very high. Traditional detection processes that involve two stages may result in lengthy procedures. Experimental results using a SPAM detection dataset demonstrate that with ACAT, the accuracy of the SPAM filter increased from 69% to over 88% after just three retraining sessions. Furthermore, ACAT outperforms conventional adversarial sample detectors, providing faster decision times, up to four times faster in some cases.



## **41. DeepZero: Scaling up Zeroth-Order Optimization for Deep Model Training**

cs.LG

Accepted to ICLR'24. Codes are available at  https://github.com/OPTML-Group/DeepZero

**SubmitDate**: 2024-03-15    [abs](http://arxiv.org/abs/2310.02025v4) [paper-pdf](http://arxiv.org/pdf/2310.02025v4)

**Authors**: Aochuan Chen, Yimeng Zhang, Jinghan Jia, James Diffenderfer, Jiancheng Liu, Konstantinos Parasyris, Yihua Zhang, Zheng Zhang, Bhavya Kailkhura, Sijia Liu

**Abstract**: Zeroth-order (ZO) optimization has become a popular technique for solving machine learning (ML) problems when first-order (FO) information is difficult or impossible to obtain. However, the scalability of ZO optimization remains an open problem: Its use has primarily been limited to relatively small-scale ML problems, such as sample-wise adversarial attack generation. To our best knowledge, no prior work has demonstrated the effectiveness of ZO optimization in training deep neural networks (DNNs) without a significant decrease in performance. To overcome this roadblock, we develop DeepZero, a principled ZO deep learning (DL) framework that can scale ZO optimization to DNN training from scratch through three primary innovations. First, we demonstrate the advantages of coordinatewise gradient estimation (CGE) over randomized vector-wise gradient estimation in training accuracy and computational efficiency. Second, we propose a sparsityinduced ZO training protocol that extends the model pruning methodology using only finite differences to explore and exploit the sparse DL prior in CGE. Third, we develop the methods of feature reuse and forward parallelization to advance the practical implementations of ZO training. Our extensive experiments show that DeepZero achieves state-of-the-art (SOTA) accuracy on ResNet-20 trained on CIFAR-10, approaching FO training performance for the first time. Furthermore, we show the practical utility of DeepZero in applications of certified adversarial defense and DL-based partial differential equation error correction, achieving 10-20% improvement over SOTA. We believe our results will inspire future research on scalable ZO optimization and contribute to advancing DL with black box. Codes are available at https://github.com/OPTML-Group/DeepZero.



## **42. Towards Non-Adversarial Algorithmic Recourse**

cs.LG

**SubmitDate**: 2024-03-15    [abs](http://arxiv.org/abs/2403.10330v1) [paper-pdf](http://arxiv.org/pdf/2403.10330v1)

**Authors**: Tobias Leemann, Martin Pawelczyk, Bardh Prenkaj, Gjergji Kasneci

**Abstract**: The streams of research on adversarial examples and counterfactual explanations have largely been growing independently. This has led to several recent works trying to elucidate their similarities and differences. Most prominently, it has been argued that adversarial examples, as opposed to counterfactual explanations, have a unique characteristic in that they lead to a misclassification compared to the ground truth. However, the computational goals and methodologies employed in existing counterfactual explanation and adversarial example generation methods often lack alignment with this requirement. Using formal definitions of adversarial examples and counterfactual explanations, we introduce non-adversarial algorithmic recourse and outline why in high-stakes situations, it is imperative to obtain counterfactual explanations that do not exhibit adversarial characteristics. We subsequently investigate how different components in the objective functions, e.g., the machine learning model or cost function used to measure distance, determine whether the outcome can be considered an adversarial example or not. Our experiments on common datasets highlight that these design choices are often more critical in deciding whether recourse is non-adversarial than whether recourse or attack algorithms are used. Furthermore, we show that choosing a robust and accurate machine learning model results in less adversarial recourse desired in practice.



## **43. Interactive Trimming against Evasive Online Data Manipulation Attacks: A Game-Theoretic Approach**

cs.CR

This manuscript is accepted by ICDE '24

**SubmitDate**: 2024-03-15    [abs](http://arxiv.org/abs/2403.10313v1) [paper-pdf](http://arxiv.org/pdf/2403.10313v1)

**Authors**: Yue Fu, Qingqing Ye, Rong Du, Haibo Hu

**Abstract**: With the exponential growth of data and its crucial impact on our lives and decision-making, the integrity of data has become a significant concern. Malicious data poisoning attacks, where false values are injected into the data, can disrupt machine learning processes and lead to severe consequences. To mitigate these attacks, distance-based defenses, such as trimming, have been proposed, but they can be easily evaded by white-box attackers. The evasiveness and effectiveness of poisoning attack strategies are two sides of the same coin, making game theory a promising approach. However, existing game-theoretical models often overlook the complexities of online data poisoning attacks, where strategies must adapt to the dynamic process of data collection.   In this paper, we present an interactive game-theoretical model to defend online data manipulation attacks using the trimming strategy. Our model accommodates a complete strategy space, making it applicable to strong evasive and colluding adversaries. Leveraging the principle of least action and the Euler-Lagrange equation from theoretical physics, we derive an analytical model for the game-theoretic process. To demonstrate its practical usage, we present a case study in a privacy-preserving data collection system under local differential privacy where a non-deterministic utility function is adopted. Two strategies are devised from this analytical model, namely, Tit-for-tat and Elastic. We conduct extensive experiments on real-world datasets, which showcase the effectiveness and accuracy of these two strategies.



## **44. Chernoff Information as a Privacy Constraint for Adversarial Classification**

cs.IT

**SubmitDate**: 2024-03-15    [abs](http://arxiv.org/abs/2403.10307v1) [paper-pdf](http://arxiv.org/pdf/2403.10307v1)

**Authors**: Ayşe Ünsal, Melek Önen

**Abstract**: This work studies a privacy metric based on Chernoff information, \textit{Chernoff differential privacy}, due to its significance in characterization of classifier performance. Adversarial classification, as any other classification problem is built around minimization of the (average or correct detection) probability of error in deciding on either of the classes in the case of binary classification. Unlike the classical hypothesis testing problem, where the false alarm and mis-detection probabilities are handled separately resulting in an asymmetric behavior of the best error exponent, in this work, we focus on the Bayesian setting and characterize the relationship between the best error exponent of the average error probability and $\varepsilon-$differential privacy. Accordingly, we re-derive Chernoff differential privacy in terms of $\varepsilon-$differential privacy using the Radon-Nikodym derivative and show that it satisfies the composition property. Subsequently, we present numerical evaluation results, which demonstrates that Chernoff information outperforms Kullback-Leibler divergence as a function of the privacy parameter $\varepsilon$, the impact of the adversary's attack and global sensitivity for the problem of adversarial classification in Laplace mechanisms.



## **45. IRAD: Implicit Representation-driven Image Resampling against Adversarial Attacks**

cs.CV

**SubmitDate**: 2024-03-15    [abs](http://arxiv.org/abs/2310.11890v2) [paper-pdf](http://arxiv.org/pdf/2310.11890v2)

**Authors**: Yue Cao, Tianlin Li, Xiaofeng Cao, Ivor Tsang, Yang Liu, Qing Guo

**Abstract**: We introduce a novel approach to counter adversarial attacks, namely, image resampling. Image resampling transforms a discrete image into a new one, simulating the process of scene recapturing or rerendering as specified by a geometrical transformation. The underlying rationale behind our idea is that image resampling can alleviate the influence of adversarial perturbations while preserving essential semantic information, thereby conferring an inherent advantage in defending against adversarial attacks. To validate this concept, we present a comprehensive study on leveraging image resampling to defend against adversarial attacks. We have developed basic resampling methods that employ interpolation strategies and coordinate shifting magnitudes. Our analysis reveals that these basic methods can partially mitigate adversarial attacks. However, they come with apparent limitations: the accuracy of clean images noticeably decreases, while the improvement in accuracy on adversarial examples is not substantial. We propose implicit representation-driven image resampling (IRAD) to overcome these limitations. First, we construct an implicit continuous representation that enables us to represent any input image within a continuous coordinate space. Second, we introduce SampleNet, which automatically generates pixel-wise shifts for resampling in response to different inputs. Furthermore, we can extend our approach to the state-of-the-art diffusion-based method, accelerating it with fewer time steps while preserving its defense capability. Extensive experiments demonstrate that our method significantly enhances the adversarial robustness of diverse deep models against various attacks while maintaining high accuracy on clean images.



## **46. Synthesizing Physical Backdoor Datasets: An Automated Framework Leveraging Deep Generative Models**

cs.CR

**SubmitDate**: 2024-03-15    [abs](http://arxiv.org/abs/2312.03419v3) [paper-pdf](http://arxiv.org/pdf/2312.03419v3)

**Authors**: Sze Jue Yang, Chinh D. La, Quang H. Nguyen, Kok-Seng Wong, Anh Tuan Tran, Chee Seng Chan, Khoa D. Doan

**Abstract**: Backdoor attacks, representing an emerging threat to the integrity of deep neural networks, have garnered significant attention due to their ability to compromise deep learning systems clandestinely. While numerous backdoor attacks occur within the digital realm, their practical implementation in real-world prediction systems remains limited and vulnerable to disturbances in the physical world. Consequently, this limitation has given rise to the development of physical backdoor attacks, where trigger objects manifest as physical entities within the real world. However, creating the requisite dataset to train or evaluate a physical backdoor model is a daunting task, limiting the backdoor researchers and practitioners from studying such physical attack scenarios. This paper unleashes a recipe that empowers backdoor researchers to effortlessly create a malicious, physical backdoor dataset based on advances in generative modeling. Particularly, this recipe involves 3 automatic modules: suggesting the suitable physical triggers, generating the poisoned candidate samples (either by synthesizing new samples or editing existing clean samples), and finally refining for the most plausible ones. As such, it effectively mitigates the perceived complexity associated with creating a physical backdoor dataset, transforming it from a daunting task into an attainable objective. Extensive experiment results show that datasets created by our "recipe" enable adversaries to achieve an impressive attack success rate on real physical world data and exhibit similar properties compared to previous physical backdoor attack studies. This paper offers researchers a valuable toolkit for studies of physical backdoors, all within the confines of their laboratories.



## **47. Adversarial Training on Purification (AToP): Advancing Both Robustness and Generalization**

cs.CV

**SubmitDate**: 2024-03-15    [abs](http://arxiv.org/abs/2401.16352v3) [paper-pdf](http://arxiv.org/pdf/2401.16352v3)

**Authors**: Guang Lin, Chao Li, Jianhai Zhang, Toshihisa Tanaka, Qibin Zhao

**Abstract**: The deep neural networks are known to be vulnerable to well-designed adversarial attacks. The most successful defense technique based on adversarial training (AT) can achieve optimal robustness against particular attacks but cannot generalize well to unseen attacks. Another effective defense technique based on adversarial purification (AP) can enhance generalization but cannot achieve optimal robustness. Meanwhile, both methods share one common limitation on the degraded standard accuracy. To mitigate these issues, we propose a novel pipeline to acquire the robust purifier model, named Adversarial Training on Purification (AToP), which comprises two components: perturbation destruction by random transforms (RT) and purifier model fine-tuned (FT) by adversarial loss. RT is essential to avoid overlearning to known attacks, resulting in the robustness generalization to unseen attacks, and FT is essential for the improvement of robustness. To evaluate our method in an efficient and scalable way, we conduct extensive experiments on CIFAR-10, CIFAR-100, and ImageNette to demonstrate that our method achieves optimal robustness and exhibits generalization ability against unseen attacks.



## **48. Benchmarking Adversarial Robustness of Image Shadow Removal with Shadow-adaptive Attacks**

cs.CV

Accepted to ICASSP 2024

**SubmitDate**: 2024-03-15    [abs](http://arxiv.org/abs/2403.10076v1) [paper-pdf](http://arxiv.org/pdf/2403.10076v1)

**Authors**: Chong Wang, Yi Yu, Lanqing Guo, Bihan Wen

**Abstract**: Shadow removal is a task aimed at erasing regional shadows present in images and reinstating visually pleasing natural scenes with consistent illumination. While recent deep learning techniques have demonstrated impressive performance in image shadow removal, their robustness against adversarial attacks remains largely unexplored. Furthermore, many existing attack frameworks typically allocate a uniform budget for perturbations across the entire input image, which may not be suitable for attacking shadow images. This is primarily due to the unique characteristic of spatially varying illumination within shadow images. In this paper, we propose a novel approach, called shadow-adaptive adversarial attack. Different from standard adversarial attacks, our attack budget is adjusted based on the pixel intensity in different regions of shadow images. Consequently, the optimized adversarial noise in the shadowed regions becomes visually less perceptible while permitting a greater tolerance for perturbations in non-shadow regions. The proposed shadow-adaptive attacks naturally align with the varying illumination distribution in shadow images, resulting in perturbations that are less conspicuous. Building on this, we conduct a comprehensive empirical evaluation of existing shadow removal methods, subjecting them to various levels of attack on publicly available datasets.



## **49. Revisiting Adversarial Training under Long-Tailed Distributions**

cs.CV

Accepted to CVPR 2024

**SubmitDate**: 2024-03-15    [abs](http://arxiv.org/abs/2403.10073v1) [paper-pdf](http://arxiv.org/pdf/2403.10073v1)

**Authors**: Xinli Yue, Ningping Mou, Qian Wang, Lingchen Zhao

**Abstract**: Deep neural networks are vulnerable to adversarial attacks, often leading to erroneous outputs. Adversarial training has been recognized as one of the most effective methods to counter such attacks. However, existing adversarial training techniques have predominantly been tested on balanced datasets, whereas real-world data often exhibit a long-tailed distribution, casting doubt on the efficacy of these methods in practical scenarios.   In this paper, we delve into adversarial training under long-tailed distributions. Through an analysis of the previous work "RoBal", we discover that utilizing Balanced Softmax Loss alone can achieve performance comparable to the complete RoBal approach while significantly reducing training overheads. Additionally, we reveal that, similar to uniform distributions, adversarial training under long-tailed distributions also suffers from robust overfitting. To address this, we explore data augmentation as a solution and unexpectedly discover that, unlike results obtained with balanced data, data augmentation not only effectively alleviates robust overfitting but also significantly improves robustness. We further investigate the reasons behind the improvement of robustness through data augmentation and identify that it is attributable to the increased diversity of examples. Extensive experiments further corroborate that data augmentation alone can significantly improve robustness. Finally, building on these findings, we demonstrate that compared to RoBal, the combination of BSL and data augmentation leads to a +6.66% improvement in model robustness under AutoAttack on CIFAR-10-LT. Our code is available at https://github.com/NISPLab/AT-BSL .



## **50. Towards Adversarially Robust Dataset Distillation by Curvature Regularization**

cs.LG

17 pages, 3 figures

**SubmitDate**: 2024-03-15    [abs](http://arxiv.org/abs/2403.10045v1) [paper-pdf](http://arxiv.org/pdf/2403.10045v1)

**Authors**: Eric Xue, Yijiang Li, Haoyang Liu, Yifan Shen, Haohan Wang

**Abstract**: Dataset distillation (DD) allows datasets to be distilled to fractions of their original size while preserving the rich distributional information so that models trained on the distilled datasets can achieve a comparable accuracy while saving significant computational loads. Recent research in this area has been focusing on improving the accuracy of models trained on distilled datasets. In this paper, we aim to explore a new perspective of DD. We study how to embed adversarial robustness in distilled datasets, so that models trained on these datasets maintain the high accuracy and meanwhile acquire better adversarial robustness. We propose a new method that achieves this goal by incorporating curvature regularization into the distillation process with much less computational overhead than standard adversarial training. Extensive empirical experiments suggest that our method not only outperforms standard adversarial training on both accuracy and robustness with less computation overhead but is also capable of generating robust distilled datasets that can withstand various adversarial attacks.



