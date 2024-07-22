# Latest Adversarial Attack Papers
**update at 2024-07-22 09:46:47**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. Multi-Attribute Vision Transformers are Efficient and Robust Learners**

cs.CV

Accepted at IEEE ICIP 2024. arXiv admin note: text overlap with  arXiv:2207.08677 by other authors

**SubmitDate**: 2024-07-19    [abs](http://arxiv.org/abs/2402.08070v2) [paper-pdf](http://arxiv.org/pdf/2402.08070v2)

**Authors**: Hanan Gani, Nada Saadi, Noor Hussein, Karthik Nandakumar

**Abstract**: Since their inception, Vision Transformers (ViTs) have emerged as a compelling alternative to Convolutional Neural Networks (CNNs) across a wide spectrum of tasks. ViTs exhibit notable characteristics, including global attention, resilience against occlusions, and adaptability to distribution shifts. One underexplored aspect of ViTs is their potential for multi-attribute learning, referring to their ability to simultaneously grasp multiple attribute-related tasks. In this paper, we delve into the multi-attribute learning capability of ViTs, presenting a straightforward yet effective strategy for training various attributes through a single ViT network as distinct tasks. We assess the resilience of multi-attribute ViTs against adversarial attacks and compare their performance against ViTs designed for single attributes. Moreover, we further evaluate the robustness of multi-attribute ViTs against a recent transformer based attack called Patch-Fool. Our empirical findings on the CelebA dataset provide validation for our assertion. Our code is available at https://github.com/hananshafi/MTL-ViT



## **2. SlowPerception: Physical-World Latency Attack against Visual Perception in Autonomous Driving**

cs.CV

This submission was made without all contributors' consent

**SubmitDate**: 2024-07-19    [abs](http://arxiv.org/abs/2406.05800v2) [paper-pdf](http://arxiv.org/pdf/2406.05800v2)

**Authors**: Chen Ma, Ningfei Wang, Zhengyu Zhao, Qi Alfred Chen, Chao Shen

**Abstract**: Autonomous Driving (AD) systems critically depend on visual perception for real-time object detection and multiple object tracking (MOT) to ensure safe driving. However, high latency in these visual perception components can lead to significant safety risks, such as vehicle collisions. While previous research has extensively explored latency attacks within the digital realm, translating these methods effectively to the physical world presents challenges. For instance, existing attacks rely on perturbations that are unrealistic or impractical for AD, such as adversarial perturbations affecting areas like the sky, or requiring large patches that obscure most of a camera's view, thus making them impossible to be conducted effectively in the real world.   In this paper, we introduce SlowPerception, the first physical-world latency attack against AD perception, via generating projector-based universal perturbations. SlowPerception strategically creates numerous phantom objects on various surfaces in the environment, significantly increasing the computational load of Non-Maximum Suppression (NMS) and MOT, thereby inducing substantial latency. Our SlowPerception achieves second-level latency in physical-world settings, with an average latency of 2.5 seconds across different AD perception systems, scenarios, and hardware configurations. This performance significantly outperforms existing state-of-the-art latency attacks. Additionally, we conduct AD system-level impact assessments, such as vehicle collisions, using industry-grade AD systems with production-grade AD simulators with a 97% average rate. We hope that our analyses can inspire further research in this critical domain, enhancing the robustness of AD systems against emerging vulnerabilities.



## **3. Do Parameters Reveal More than Loss for Membership Inference?**

cs.LG

Accepted at High-dimensional Learning Dynamics (HiLD) Workshop, ICML  2024

**SubmitDate**: 2024-07-19    [abs](http://arxiv.org/abs/2406.11544v2) [paper-pdf](http://arxiv.org/pdf/2406.11544v2)

**Authors**: Anshuman Suri, Xiao Zhang, David Evans

**Abstract**: Membership inference attacks aim to infer whether an individual record was used to train a model, serving as a key tool for disclosure auditing. While such evaluations are useful to demonstrate risk, they are computationally expensive and often make strong assumptions about potential adversaries' access to models and training environments, and thus do not provide very tight bounds on leakage from potential attacks. We show how prior claims around black-box access being sufficient for optimal membership inference do not hold for most useful settings such as stochastic gradient descent, and that optimal membership inference indeed requires white-box access. We validate our findings with a new white-box inference attack IHA (Inverse Hessian Attack) that explicitly uses model parameters by taking advantage of computing inverse-Hessian vector products. Our results show that both audits and adversaries may be able to benefit from access to model parameters, and we advocate for further research into white-box methods for membership privacy auditing.



## **4. Does Refusal Training in LLMs Generalize to the Past Tense?**

cs.CL

Update in v2: Claude-3.5 Sonnet and GPT-4o mini. We provide code and  jailbreak artifacts at https://github.com/tml-epfl/llm-past-tense

**SubmitDate**: 2024-07-19    [abs](http://arxiv.org/abs/2407.11969v2) [paper-pdf](http://arxiv.org/pdf/2407.11969v2)

**Authors**: Maksym Andriushchenko, Nicolas Flammarion

**Abstract**: Refusal training is widely used to prevent LLMs from generating harmful, undesirable, or illegal outputs. We reveal a curious generalization gap in the current refusal training approaches: simply reformulating a harmful request in the past tense (e.g., "How to make a Molotov cocktail?" to "How did people make a Molotov cocktail?") is often sufficient to jailbreak many state-of-the-art LLMs. We systematically evaluate this method on Llama-3 8B, Claude-3.5 Sonnet, GPT-3.5 Turbo, Gemma-2 9B, Phi-3-Mini, GPT-4o mini, GPT-4o, and R2D2 models using GPT-3.5 Turbo as a reformulation model. For example, the success rate of this simple attack on GPT-4o increases from 1% using direct requests to 88% using 20 past tense reformulation attempts on harmful requests from JailbreakBench with GPT-4 as a jailbreak judge. Interestingly, we also find that reformulations in the future tense are less effective, suggesting that refusal guardrails tend to consider past historical questions more benign than hypothetical future questions. Moreover, our experiments on fine-tuning GPT-3.5 Turbo show that defending against past reformulations is feasible when past tense examples are explicitly included in the fine-tuning data. Overall, our findings highlight that the widely used alignment techniques -- such as SFT, RLHF, and adversarial training -- employed to align the studied models can be brittle and do not always generalize as intended. We provide code and jailbreak artifacts at https://github.com/tml-epfl/llm-past-tense.



## **5. Watermark Smoothing Attacks against Language Models**

cs.LG

**SubmitDate**: 2024-07-19    [abs](http://arxiv.org/abs/2407.14206v1) [paper-pdf](http://arxiv.org/pdf/2407.14206v1)

**Authors**: Hongyan Chang, Hamed Hassani, Reza Shokri

**Abstract**: Watermarking is a technique used to embed a hidden signal in the probability distribution of text generated by large language models (LLMs), enabling attribution of the text to the originating model. We introduce smoothing attacks and show that existing watermarking methods are not robust against minor modifications of text. An adversary can use weaker language models to smooth out the distribution perturbations caused by watermarks without significantly compromising the quality of the generated text. The modified text resulting from the smoothing attack remains close to the distribution of text that the original model (without watermark) would have produced. Our attack reveals a fundamental limitation of a wide range of watermarking techniques.



## **6. MVPatch: More Vivid Patch for Adversarial Camouflaged Attacks on Object Detectors in the Physical World**

cs.CR

16 pages, 8 figures. This work has been submitted to the IEEE for  possible publication. Copyright may be transferred without notice, after  which this version may no longer be accessible

**SubmitDate**: 2024-07-19    [abs](http://arxiv.org/abs/2312.17431v3) [paper-pdf](http://arxiv.org/pdf/2312.17431v3)

**Authors**: Zheng Zhou, Hongbo Zhao, Ju Liu, Qiaosheng Zhang, Liwei Geng, Shuchang Lyu, Wenquan Feng

**Abstract**: Recent studies have shown that Adversarial Patches (APs) can effectively manipulate object detection models. However, the conspicuous patterns often associated with these patches tend to attract human attention, posing a significant challenge. Existing research has primarily focused on enhancing attack efficacy in the physical domain while often neglecting the optimization of stealthiness and transferability. Furthermore, applying APs in real-world scenarios faces major challenges related to transferability, stealthiness, and practicality. To address these challenges, we introduce generalization theory into the context of APs, enabling our iterative process to simultaneously enhance transferability and refine visual correlation with realistic images. We propose a Dual-Perception-Based Framework (DPBF) to generate the More Vivid Patch (MVPatch), which enhances transferability, stealthiness, and practicality. The DPBF integrates two key components: the Model-Perception-Based Module (MPBM) and the Human-Perception-Based Module (HPBM), along with regularization terms. The MPBM employs ensemble strategy to reduce object confidence scores across multiple detectors, thereby improving AP transferability with robust theoretical support. Concurrently, the HPBM introduces a lightweight method for achieving visual similarity, creating natural and inconspicuous adversarial patches without relying on additional generative models. The regularization terms further enhance the practicality of the generated APs in the physical domain. Additionally, we introduce naturalness and transferability scores to provide an unbiased assessment of APs. Extensive experimental validation demonstrates that MVPatch achieves superior transferability and a natural appearance in both digital and physical domains, underscoring its effectiveness and stealthiness.



## **7. Adversarial Examples in the Physical World: A Survey**

cs.CV

Adversarial examples, physical-world scenarios, attacks and defenses

**SubmitDate**: 2024-07-19    [abs](http://arxiv.org/abs/2311.01473v2) [paper-pdf](http://arxiv.org/pdf/2311.01473v2)

**Authors**: Jiakai Wang, Xianglong Liu, Jin Hu, Donghua Wang, Siyang Wu, Tingsong Jiang, Yuanfang Guo, Aishan Liu, Aishan Liu, Jiantao Zhou

**Abstract**: Deep neural networks (DNNs) have demonstrated high vulnerability to adversarial examples, raising broad security concerns about their applications. Besides the attacks in the digital world, the practical implications of adversarial examples in the physical world present significant challenges and safety concerns. However, current research on physical adversarial examples (PAEs) lacks a comprehensive understanding of their unique characteristics, leading to limited significance and understanding. In this paper, we address this gap by thoroughly examining the characteristics of PAEs within a practical workflow encompassing training, manufacturing, and re-sampling processes. By analyzing the links between physical adversarial attacks, we identify manufacturing and re-sampling as the primary sources of distinct attributes and particularities in PAEs. Leveraging this knowledge, we develop a comprehensive analysis and classification framework for PAEs based on their specific characteristics, covering over 100 studies on physical-world adversarial examples. Furthermore, we investigate defense strategies against PAEs and identify open challenges and opportunities for future research. We aim to provide a fresh, thorough, and systematic understanding of PAEs, thereby promoting the development of robust adversarial learning and its application in open-world scenarios to provide the community with a continuously updated list of physical world adversarial sample resources, including papers, code, \etc, within the proposed framework



## **8. Resilient Consensus Sustained Collaboratively**

cs.CR

15 pages, 7 figures

**SubmitDate**: 2024-07-19    [abs](http://arxiv.org/abs/2302.02325v5) [paper-pdf](http://arxiv.org/pdf/2302.02325v5)

**Authors**: Junchao Chen, Suyash Gupta, Alberto Sonnino, Lefteris Kokoris-Kogias, Mohammad Sadoghi

**Abstract**: Decentralized systems built around blockchain technology promise clients an immutable ledger. They add a transaction to the ledger after it undergoes consensus among the replicas that run a Proof-of-Stake (PoS) or Byzantine Fault-Tolerant (BFT) consensus protocol. Unfortunately, these protocols face a long-range attack where an adversary having access to the private keys of the replicas can rewrite the ledger. One solution is forcing each committed block from these protocols to undergo another consensus, Proof-of-Work(PoW) consensus; PoW protocol leads to wastage of computational resources as miners compete to solve complex puzzles. In this paper, we present the design of our Power-of-Collaboration (PoC) protocol, which guards existing PoS/BFT blockchains against long-range attacks and requires miners to collaborate rather than compete. PoC guarantees fairness and accountability and only marginally degrades the throughput of the underlying system.



## **9. Personalized Privacy Protection Mask Against Unauthorized Facial Recognition**

cs.CV

ECCV 2024

**SubmitDate**: 2024-07-19    [abs](http://arxiv.org/abs/2407.13975v1) [paper-pdf](http://arxiv.org/pdf/2407.13975v1)

**Authors**: Ka-Ho Chow, Sihao Hu, Tiansheng Huang, Ling Liu

**Abstract**: Face recognition (FR) can be abused for privacy intrusion. Governments, private companies, or even individual attackers can collect facial images by web scraping to build an FR system identifying human faces without their consent. This paper introduces Chameleon, which learns to generate a user-centric personalized privacy protection mask, coined as P3-Mask, to protect facial images against unauthorized FR with three salient features. First, we use a cross-image optimization to generate one P3-Mask for each user instead of tailoring facial perturbation for each facial image of a user. It enables efficient and instant protection even for users with limited computing resources. Second, we incorporate a perceptibility optimization to preserve the visual quality of the protected facial images. Third, we strengthen the robustness of P3-Mask against unknown FR models by integrating focal diversity-optimized ensemble learning into the mask generation process. Extensive experiments on two benchmark datasets show that Chameleon outperforms three state-of-the-art methods with instant protection and minimal degradation of image quality. Furthermore, Chameleon enables cost-effective FR authorization using the P3-Mask as a personalized de-obfuscation key, and it demonstrates high resilience against adaptive adversaries.



## **10. A Closer Look at GAN Priors: Exploiting Intermediate Features for Enhanced Model Inversion Attacks**

cs.CV

**SubmitDate**: 2024-07-18    [abs](http://arxiv.org/abs/2407.13863v1) [paper-pdf](http://arxiv.org/pdf/2407.13863v1)

**Authors**: Yixiang Qiu, Hao Fang, Hongyao Yu, Bin Chen, MeiKang Qiu, Shu-Tao Xia

**Abstract**: Model Inversion (MI) attacks aim to reconstruct privacy-sensitive training data from released models by utilizing output information, raising extensive concerns about the security of Deep Neural Networks (DNNs). Recent advances in generative adversarial networks (GANs) have contributed significantly to the improved performance of MI attacks due to their powerful ability to generate realistic images with high fidelity and appropriate semantics. However, previous MI attacks have solely disclosed private information in the latent space of GAN priors, limiting their semantic extraction and transferability across multiple target models and datasets. To address this challenge, we propose a novel method, Intermediate Features enhanced Generative Model Inversion (IF-GMI), which disassembles the GAN structure and exploits features between intermediate blocks. This allows us to extend the optimization space from latent code to intermediate features with enhanced expressive capabilities. To prevent GAN priors from generating unrealistic images, we apply a L1 ball constraint to the optimization process. Experiments on multiple benchmarks demonstrate that our method significantly outperforms previous approaches and achieves state-of-the-art results under various settings, especially in the out-of-distribution (OOD) scenario. Our code is available at: https://github.com/final-solution/IF-GMI



## **11. Jailbreaking Black Box Large Language Models in Twenty Queries**

cs.LG

**SubmitDate**: 2024-07-18    [abs](http://arxiv.org/abs/2310.08419v4) [paper-pdf](http://arxiv.org/pdf/2310.08419v4)

**Authors**: Patrick Chao, Alexander Robey, Edgar Dobriban, Hamed Hassani, George J. Pappas, Eric Wong

**Abstract**: There is growing interest in ensuring that large language models (LLMs) align with human values. However, the alignment of such models is vulnerable to adversarial jailbreaks, which coax LLMs into overriding their safety guardrails. The identification of these vulnerabilities is therefore instrumental in understanding inherent weaknesses and preventing future misuse. To this end, we propose Prompt Automatic Iterative Refinement (PAIR), an algorithm that generates semantic jailbreaks with only black-box access to an LLM. PAIR -- which is inspired by social engineering attacks -- uses an attacker LLM to automatically generate jailbreaks for a separate targeted LLM without human intervention. In this way, the attacker LLM iteratively queries the target LLM to update and refine a candidate jailbreak. Empirically, PAIR often requires fewer than twenty queries to produce a jailbreak, which is orders of magnitude more efficient than existing algorithms. PAIR also achieves competitive jailbreaking success rates and transferability on open and closed-source LLMs, including GPT-3.5/4, Vicuna, and Gemini.



## **12. Black-Box Opinion Manipulation Attacks to Retrieval-Augmented Generation of Large Language Models**

cs.CL

10 pages, 3 figures, under review

**SubmitDate**: 2024-07-18    [abs](http://arxiv.org/abs/2407.13757v1) [paper-pdf](http://arxiv.org/pdf/2407.13757v1)

**Authors**: Zhuo Chen, Jiawei Liu, Haotan Liu, Qikai Cheng, Fan Zhang, Wei Lu, Xiaozhong Liu

**Abstract**: Retrieval-Augmented Generation (RAG) is applied to solve hallucination problems and real-time constraints of large language models, but it also induces vulnerabilities against retrieval corruption attacks. Existing research mainly explores the unreliability of RAG in white-box and closed-domain QA tasks. In this paper, we aim to reveal the vulnerabilities of Retrieval-Enhanced Generative (RAG) models when faced with black-box attacks for opinion manipulation. We explore the impact of such attacks on user cognition and decision-making, providing new insight to enhance the reliability and security of RAG models. We manipulate the ranking results of the retrieval model in RAG with instruction and use these results as data to train a surrogate model. By employing adversarial retrieval attack methods to the surrogate model, black-box transfer attacks on RAG are further realized. Experiments conducted on opinion datasets across multiple topics show that the proposed attack strategy can significantly alter the opinion polarity of the content generated by RAG. This demonstrates the model's vulnerability and, more importantly, reveals the potential negative impact on user cognition and decision-making, making it easier to mislead users into accepting incorrect or biased information.



## **13. Cross-Task Attack: A Self-Supervision Generative Framework Based on Attention Shift**

cs.CV

Has been accepted by IJCNN2024

**SubmitDate**: 2024-07-18    [abs](http://arxiv.org/abs/2407.13700v1) [paper-pdf](http://arxiv.org/pdf/2407.13700v1)

**Authors**: Qingyuan Zeng, Yunpeng Gong, Min Jiang

**Abstract**: Studying adversarial attacks on artificial intelligence (AI) systems helps discover model shortcomings, enabling the construction of a more robust system. Most existing adversarial attack methods only concentrate on single-task single-model or single-task cross-model scenarios, overlooking the multi-task characteristic of artificial intelligence systems. As a result, most of the existing attacks do not pose a practical threat to a comprehensive and collaborative AI system. However, implementing cross-task attacks is highly demanding and challenging due to the difficulty in obtaining the real labels of different tasks for the same picture and harmonizing the loss functions across different tasks. To address this issue, we propose a self-supervised Cross-Task Attack framework (CTA), which utilizes co-attention and anti-attention maps to generate cross-task adversarial perturbation. Specifically, the co-attention map reflects the area to which different visual task models pay attention, while the anti-attention map reflects the area that different visual task models neglect. CTA generates cross-task perturbations by shifting the attention area of samples away from the co-attention map and closer to the anti-attention map. We conduct extensive experiments on multiple vision tasks and the experimental results confirm the effectiveness of the proposed design for adversarial attacks.



## **14. Prover-Verifier Games improve legibility of LLM outputs**

cs.CL

**SubmitDate**: 2024-07-18    [abs](http://arxiv.org/abs/2407.13692v1) [paper-pdf](http://arxiv.org/pdf/2407.13692v1)

**Authors**: Jan Hendrik Kirchner, Yining Chen, Harri Edwards, Jan Leike, Nat McAleese, Yuri Burda

**Abstract**: One way to increase confidence in the outputs of Large Language Models (LLMs) is to support them with reasoning that is clear and easy to check -- a property we call legibility. We study legibility in the context of solving grade-school math problems and show that optimizing chain-of-thought solutions only for answer correctness can make them less legible. To mitigate the loss in legibility, we propose a training algorithm inspired by Prover-Verifier Game from Anil et al. (2021). Our algorithm iteratively trains small verifiers to predict solution correctness, "helpful" provers to produce correct solutions that the verifier accepts, and "sneaky" provers to produce incorrect solutions that fool the verifier. We find that the helpful prover's accuracy and the verifier's robustness to adversarial attacks increase over the course of training. Furthermore, we show that legibility training transfers to time-constrained humans tasked with verifying solution correctness. Over course of LLM training human accuracy increases when checking the helpful prover's solutions, and decreases when checking the sneaky prover's solutions. Hence, training for checkability by small verifiers is a plausible technique for increasing output legibility. Our results suggest legibility training against small verifiers as a practical avenue for increasing legibility of large LLMs to humans, and thus could help with alignment of superhuman models.



## **15. Beyond Dropout: Robust Convolutional Neural Networks Based on Local Feature Masking**

cs.CV

It has been accepted by IJCNN 2024

**SubmitDate**: 2024-07-18    [abs](http://arxiv.org/abs/2407.13646v1) [paper-pdf](http://arxiv.org/pdf/2407.13646v1)

**Authors**: Yunpeng Gong, Chuangliang Zhang, Yongjie Hou, Lifei Chen, Min Jiang

**Abstract**: In the contemporary of deep learning, where models often grapple with the challenge of simultaneously achieving robustness against adversarial attacks and strong generalization capabilities, this study introduces an innovative Local Feature Masking (LFM) strategy aimed at fortifying the performance of Convolutional Neural Networks (CNNs) on both fronts. During the training phase, we strategically incorporate random feature masking in the shallow layers of CNNs, effectively alleviating overfitting issues, thereby enhancing the model's generalization ability and bolstering its resilience to adversarial attacks. LFM compels the network to adapt by leveraging remaining features to compensate for the absence of certain semantic features, nurturing a more elastic feature learning mechanism. The efficacy of LFM is substantiated through a series of quantitative and qualitative assessments, collectively showcasing a consistent and significant improvement in CNN's generalization ability and resistance against adversarial attacks--a phenomenon not observed in current and prior methodologies. The seamless integration of LFM into established CNN frameworks underscores its potential to advance both generalization and adversarial robustness within the deep learning paradigm. Through comprehensive experiments, including robust person re-identification baseline generalization experiments and adversarial attack experiments, we demonstrate the substantial enhancements offered by LFM in addressing the aforementioned challenges. This contribution represents a noteworthy stride in advancing robust neural network architectures.



## **16. Not Just Change the Labels, Learn the Features: Watermarking Deep Neural Networks with Multi-View Data**

cs.CR

ECCV 2024

**SubmitDate**: 2024-07-18    [abs](http://arxiv.org/abs/2403.10663v2) [paper-pdf](http://arxiv.org/pdf/2403.10663v2)

**Authors**: Yuxuan Li, Sarthak Kumar Maharana, Yunhui Guo

**Abstract**: With the increasing prevalence of Machine Learning as a Service (MLaaS) platforms, there is a growing focus on deep neural network (DNN) watermarking techniques. These methods are used to facilitate the verification of ownership for a target DNN model to protect intellectual property. One of the most widely employed watermarking techniques involves embedding a trigger set into the source model. Unfortunately, existing methodologies based on trigger sets are still susceptible to functionality-stealing attacks, potentially enabling adversaries to steal the functionality of the source model without a reliable means of verifying ownership. In this paper, we first introduce a novel perspective on trigger set-based watermarking methods from a feature learning perspective. Specifically, we demonstrate that by selecting data exhibiting multiple features, also referred to as \emph{multi-view data}, it becomes feasible to effectively defend functionality stealing attacks. Based on this perspective, we introduce a novel watermarking technique based on Multi-view dATa, called MAT, for efficiently embedding watermarks within DNNs. This approach involves constructing a trigger set with multi-view data and incorporating a simple feature-based regularization method for training the source model. We validate our method across various benchmarks and demonstrate its efficacy in defending against model extraction attacks, surpassing relevant baselines by a significant margin. The code is available at: \href{https://github.com/liyuxuan-github/MAT}{https://github.com/liyuxuan-github/MAT}.



## **17. Distributionally and Adversarially Robust Logistic Regression via Intersecting Wasserstein Balls**

math.OC

34 pages, 3 color figures, under review at a conference

**SubmitDate**: 2024-07-18    [abs](http://arxiv.org/abs/2407.13625v1) [paper-pdf](http://arxiv.org/pdf/2407.13625v1)

**Authors**: Aras Selvi, Eleonora Kreacic, Mohsen Ghassemi, Vamsi Potluru, Tucker Balch, Manuela Veloso

**Abstract**: Empirical risk minimization often fails to provide robustness against adversarial attacks in test data, causing poor out-of-sample performance. Adversarially robust optimization (ARO) has thus emerged as the de facto standard for obtaining models that hedge against such attacks. However, while these models are robust against adversarial attacks, they tend to suffer severely from overfitting. To address this issue for logistic regression, we study the Wasserstein distributionally robust (DR) counterpart of ARO and show that this problem admits a tractable reformulation. Furthermore, we develop a framework to reduce the conservatism of this problem by utilizing an auxiliary dataset (e.g., synthetic, external, or out-of-domain data), whenever available, with instances independently sampled from a nonidentical but related ground truth. In particular, we intersect the ambiguity set of the DR problem with another Wasserstein ambiguity set that is built using the auxiliary dataset. We analyze the properties of the underlying optimization problem, develop efficient solution algorithms, and demonstrate that the proposed method consistently outperforms benchmark approaches on real-world datasets.



## **18. VeriQR: A Robustness Verification Tool for Quantum Machine Learning Models**

quant-ph

**SubmitDate**: 2024-07-18    [abs](http://arxiv.org/abs/2407.13533v1) [paper-pdf](http://arxiv.org/pdf/2407.13533v1)

**Authors**: Yanling Lin, Ji Guan, Wang Fang, Mingsheng Ying, Zhaofeng Su

**Abstract**: Adversarial noise attacks present a significant threat to quantum machine learning (QML) models, similar to their classical counterparts. This is especially true in the current Noisy Intermediate-Scale Quantum era, where noise is unavoidable. Therefore, it is essential to ensure the robustness of QML models before their deployment. To address this challenge, we introduce \textit{VeriQR}, the first tool designed specifically for formally verifying and improving the robustness of QML models, to the best of our knowledge. This tool mimics real-world quantum hardware's noisy impacts by incorporating random noise to formally validate a QML model's robustness. \textit{VeriQR} supports exact (sound and complete) algorithms for both local and global robustness verification. For enhanced efficiency, it implements an under-approximate (complete) algorithm and a tensor network-based algorithm to verify local and global robustness, respectively. As a formal verification tool, \textit{VeriQR} can detect adversarial examples and utilize them for further analysis and to enhance the local robustness through adversarial training, as demonstrated by experiments on real-world quantum machine learning models. Moreover, it permits users to incorporate customized noise. Based on this feature, we assess \textit{VeriQR} using various real-world examples, and experimental outcomes confirm that the addition of specific quantum noise can enhance the global robustness of QML models. These processes are made accessible through a user-friendly graphical interface provided by \textit{VeriQR}, catering to general users without requiring a deep understanding of the counter-intuitive probabilistic nature of quantum computing.



## **19. Correlation inference attacks against machine learning models**

cs.LG

Published in Science Advances. This version contains both the main  paper and supplementary material. There are minor editorial differences  between this version and the published version. The first two authors  contributed equally

**SubmitDate**: 2024-07-18    [abs](http://arxiv.org/abs/2112.08806v4) [paper-pdf](http://arxiv.org/pdf/2112.08806v4)

**Authors**: Ana-Maria Creţu, Florent Guépin, Yves-Alexandre de Montjoye

**Abstract**: Despite machine learning models being widely used today, the relationship between a model and its training dataset is not well understood. We explore correlation inference attacks, whether and when a model leaks information about the correlations between the input variables of its training dataset. We first propose a model-less attack, where an adversary exploits the spherical parametrization of correlation matrices alone to make an informed guess. Second, we propose a model-based attack, where an adversary exploits black-box model access to infer the correlations using minimal and realistic assumptions. Third, we evaluate our attacks against logistic regression and multilayer perceptron models on three tabular datasets and show the models to leak correlations. We finally show how extracted correlations can be used as building blocks for attribute inference attacks and enable weaker adversaries. Our results raise fundamental questions on what a model does and should remember from its training set.



## **20. NeuroPlug: Plugging Side-Channel Leaks in NPUs using Space Filling Curves**

cs.CR

**SubmitDate**: 2024-07-18    [abs](http://arxiv.org/abs/2407.13383v1) [paper-pdf](http://arxiv.org/pdf/2407.13383v1)

**Authors**: Nivedita Shrivastava, Smruti R. Sarangi

**Abstract**: Securing deep neural networks (DNNs) from side-channel attacks is an important problem as of today, given the substantial investment of time and resources in acquiring the raw data and training complex models. All published countermeasures (CMs) add noise N to a signal X (parameter of interest such as the net memory traffic that is leaked). The adversary observes X+N ; we shall show that it is easy to filter this noise out using targeted measurements, statistical analyses and different kinds of reasonably-assumed side information. We present a novel CM NeuroPlug that is immune to these attack methodologies mainly because we use a different formulation CX + N . We introduce a multiplicative variable C that naturally arises from feature map compression; it plays a key role in obfuscating the parameters of interest. Our approach is based on mapping all the computations to a 1-D space filling curve and then performing a sequence of tiling, compression and binning-based obfuscation operations. We follow up with proposing a theoretical framework based on Mellin transforms that allows us to accurately quantify the size of the search space as a function of the noise we add and the side information that an adversary possesses. The security guarantees provided by NeuroPlug are validated using a battery of statistical and information theory-based tests. We also demonstrate a substantial performance enhancement of 15% compared to the closest competing work.



## **21. AgentDojo: A Dynamic Environment to Evaluate Attacks and Defenses for LLM Agents**

cs.CR

Updated version after fixing a bug in the Llama implementation and  updating the travel suite

**SubmitDate**: 2024-07-18    [abs](http://arxiv.org/abs/2406.13352v2) [paper-pdf](http://arxiv.org/pdf/2406.13352v2)

**Authors**: Edoardo Debenedetti, Jie Zhang, Mislav Balunović, Luca Beurer-Kellner, Marc Fischer, Florian Tramèr

**Abstract**: AI agents aim to solve complex tasks by combining text-based reasoning with external tool calls. Unfortunately, AI agents are vulnerable to prompt injection attacks where data returned by external tools hijacks the agent to execute malicious tasks. To measure the adversarial robustness of AI agents, we introduce AgentDojo, an evaluation framework for agents that execute tools over untrusted data. To capture the evolving nature of attacks and defenses, AgentDojo is not a static test suite, but rather an extensible environment for designing and evaluating new agent tasks, defenses, and adaptive attacks. We populate the environment with 97 realistic tasks (e.g., managing an email client, navigating an e-banking website, or making travel bookings), 629 security test cases, and various attack and defense paradigms from the literature. We find that AgentDojo poses a challenge for both attacks and defenses: state-of-the-art LLMs fail at many tasks (even in the absence of attacks), and existing prompt injection attacks break some security properties but not all. We hope that AgentDojo can foster research on new design principles for AI agents that solve common tasks in a reliable and robust manner. We release the code for AgentDojo at https://github.com/ethz-spylab/agentdojo.



## **22. Benchmarking Robust Self-Supervised Learning Across Diverse Downstream Tasks**

cs.CV

Accepted at the ICML 2024 Workshop on Foundation Models in the Wild

**SubmitDate**: 2024-07-18    [abs](http://arxiv.org/abs/2407.12588v2) [paper-pdf](http://arxiv.org/pdf/2407.12588v2)

**Authors**: Antoni Kowalczuk, Jan Dubiński, Atiyeh Ashari Ghomi, Yi Sui, George Stein, Jiapeng Wu, Jesse C. Cresswell, Franziska Boenisch, Adam Dziedzic

**Abstract**: Large-scale vision models have become integral in many applications due to their unprecedented performance and versatility across downstream tasks. However, the robustness of these foundation models has primarily been explored for a single task, namely image classification. The vulnerability of other common vision tasks, such as semantic segmentation and depth estimation, remains largely unknown. We present a comprehensive empirical evaluation of the adversarial robustness of self-supervised vision encoders across multiple downstream tasks. Our attacks operate in the encoder embedding space and at the downstream task output level. In both cases, current state-of-the-art adversarial fine-tuning techniques tested only for classification significantly degrade clean and robust performance on other tasks. Since the purpose of a foundation model is to cater to multiple applications at once, our findings reveal the need to enhance encoder robustness more broadly. Our code is available at ${github.com/layer6ai-labs/ssl-robustness}$.



## **23. Enhancing TinyML Security: Study of Adversarial Attack Transferability**

cs.CR

Accepted and presented at tinyML Foundation EMEA Innovation Forum  2024

**SubmitDate**: 2024-07-18    [abs](http://arxiv.org/abs/2407.11599v2) [paper-pdf](http://arxiv.org/pdf/2407.11599v2)

**Authors**: Parin Shah, Yuvaraj Govindarajulu, Pavan Kulkarni, Manojkumar Parmar

**Abstract**: The recent strides in artificial intelligence (AI) and machine learning (ML) have propelled the rise of TinyML, a paradigm enabling AI computations at the edge without dependence on cloud connections. While TinyML offers real-time data analysis and swift responses critical for diverse applications, its devices' intrinsic resource limitations expose them to security risks. This research delves into the adversarial vulnerabilities of AI models on resource-constrained embedded hardware, with a focus on Model Extraction and Evasion Attacks. Our findings reveal that adversarial attacks from powerful host machines could be transferred to smaller, less secure devices like ESP32 and Raspberry Pi. This illustrates that adversarial attacks could be extended to tiny devices, underscoring vulnerabilities, and emphasizing the necessity for reinforced security measures in TinyML deployments. This exploration enhances the comprehension of security challenges in TinyML and offers insights for safeguarding sensitive data and ensuring device dependability in AI-powered edge computing settings.



## **24. Compressed models are NOT miniature versions of large models**

cs.LG

Accepted at the 33rd ACM International Conference on Information and  Knowledge Management (CIKM 2024) for the Short Research Paper track, 5 pages

**SubmitDate**: 2024-07-18    [abs](http://arxiv.org/abs/2407.13174v1) [paper-pdf](http://arxiv.org/pdf/2407.13174v1)

**Authors**: Rohit Raj Rai, Rishant Pal, Amit Awekar

**Abstract**: Large neural models are often compressed before deployment. Model compression is necessary for many practical reasons, such as inference latency, memory footprint, and energy consumption. Compressed models are assumed to be miniature versions of corresponding large neural models. However, we question this belief in our work. We compare compressed models with corresponding large neural models using four model characteristics: prediction errors, data representation, data distribution, and vulnerability to adversarial attack. We perform experiments using the BERT-large model and its five compressed versions. For all four model characteristics, compressed models significantly differ from the BERT-large model. Even among compressed models, they differ from each other on all four model characteristics. Apart from the expected loss in model performance, there are major side effects of using compressed models to replace large neural models.



## **25. ToDA: Target-oriented Diffusion Attacker against Recommendation System**

cs.CR

under-review

**SubmitDate**: 2024-07-18    [abs](http://arxiv.org/abs/2401.12578v3) [paper-pdf](http://arxiv.org/pdf/2401.12578v3)

**Authors**: Xiaohao Liu, Zhulin Tao, Ting Jiang, He Chang, Yunshan Ma, Yinwei Wei, Xiang Wang

**Abstract**: Recommendation systems (RS) have become indispensable tools for web services to address information overload, thus enhancing user experiences and bolstering platforms' revenues. However, with their increasing ubiquity, security concerns have also emerged. As the public accessibility of RS, they are susceptible to specific malicious attacks where adversaries can manipulate user profiles, leading to biased recommendations. Recent research often integrates additional modules using generative models to craft these deceptive user profiles, ensuring them are imperceptible while causing the intended harm. Albeit their efficacy, these models face challenges of unstable training and the exploration-exploitation dilemma, which can lead to suboptimal results. In this paper, we pioneer to investigate the potential of diffusion models (DMs), for shilling attacks. Specifically, we propose a novel Target-oriented Diffusion Attack model (ToDA). It incorporates a pre-trained autoencoder that transforms user profiles into a high dimensional space, paired with a Latent Diffusion Attacker (LDA)-the core component of ToDA. LDA introduces noise into the profiles within this latent space, adeptly steering the approximation towards targeted items through cross-attention mechanisms. The global horizon, implemented by a bipartite graph, is involved in LDA and derived from the encoded user profile feature. This makes LDA possible to extend the generation outwards the on-processing user feature itself, and bridges the gap between diffused user features and target item features. Extensive experiments compared to several SOTA baselines demonstrate ToDA's effectiveness. Specific studies exploit the elaborative design of ToDA and underscore the potency of advanced generative models in such contexts.



## **26. PG-Attack: A Precision-Guided Adversarial Attack Framework Against Vision Foundation Models for Autonomous Driving**

cs.MM

First-Place in the CVPR 2024 Workshop Challenge: Black-box  Adversarial Attacks on Vision Foundation Models

**SubmitDate**: 2024-07-18    [abs](http://arxiv.org/abs/2407.13111v1) [paper-pdf](http://arxiv.org/pdf/2407.13111v1)

**Authors**: Jiyuan Fu, Zhaoyu Chen, Kaixun Jiang, Haijing Guo, Shuyong Gao, Wenqiang Zhang

**Abstract**: Vision foundation models are increasingly employed in autonomous driving systems due to their advanced capabilities. However, these models are susceptible to adversarial attacks, posing significant risks to the reliability and safety of autonomous vehicles. Adversaries can exploit these vulnerabilities to manipulate the vehicle's perception of its surroundings, leading to erroneous decisions and potentially catastrophic consequences. To address this challenge, we propose a novel Precision-Guided Adversarial Attack (PG-Attack) framework that combines two techniques: Precision Mask Perturbation Attack (PMP-Attack) and Deceptive Text Patch Attack (DTP-Attack). PMP-Attack precisely targets the attack region to minimize the overall perturbation while maximizing its impact on the target object's representation in the model's feature space. DTP-Attack introduces deceptive text patches that disrupt the model's understanding of the scene, further enhancing the attack's effectiveness. Our experiments demonstrate that PG-Attack successfully deceives a variety of advanced multi-modal large models, including GPT-4V, Qwen-VL, and imp-V1. Additionally, we won First-Place in the CVPR 2024 Workshop Challenge: Black-box Adversarial Attacks on Vision Foundation Models and codes are available at https://github.com/fuhaha824/PG-Attack.



## **27. Krait: A Backdoor Attack Against Graph Prompt Tuning**

cs.LG

Previously submitted to CCS on 04/29

**SubmitDate**: 2024-07-18    [abs](http://arxiv.org/abs/2407.13068v1) [paper-pdf](http://arxiv.org/pdf/2407.13068v1)

**Authors**: Ying Song, Rita Singh, Balaji Palanisamy

**Abstract**: Graph prompt tuning has emerged as a promising paradigm to effectively transfer general graph knowledge from pre-trained models to various downstream tasks, particularly in few-shot contexts. However, its susceptibility to backdoor attacks, where adversaries insert triggers to manipulate outcomes, raises a critical concern. We conduct the first study to investigate such vulnerability, revealing that backdoors can disguise benign graph prompts, thus evading detection. We introduce Krait, a novel graph prompt backdoor. Specifically, we propose a simple yet effective model-agnostic metric called label non-uniformity homophily to select poisoned candidates, significantly reducing computational complexity. To accommodate diverse attack scenarios and advanced attack types, we design three customizable trigger generation methods to craft prompts as triggers. We propose a novel centroid similarity-based loss function to optimize prompt tuning for attack effectiveness and stealthiness. Experiments on four real-world graphs demonstrate that Krait can efficiently embed triggers to merely 0.15% to 2% of training nodes, achieving high attack success rates without sacrificing clean accuracy. Notably, in one-to-one and all-to-one attacks, Krait can achieve 100% attack success rates by poisoning as few as 2 and 22 nodes, respectively. Our experiments further show that Krait remains potent across different transfer cases, attack types, and graph neural network backbones. Additionally, Krait can be successfully extended to the black-box setting, posing more severe threats. Finally, we analyze why Krait can evade both classical and state-of-the-art defenses, and provide practical insights for detecting and mitigating this class of attacks.



## **28. Deep Generative Attacks and Countermeasures for Data-Driven Offline Signature Verification**

cs.CV

Ten pages, 6 figures, 1 table, Signature verification, Deep  generative models, attacks, generative attack explainability, data-driven  verification system

**SubmitDate**: 2024-07-17    [abs](http://arxiv.org/abs/2312.00987v2) [paper-pdf](http://arxiv.org/pdf/2312.00987v2)

**Authors**: An Ngo, Rajesh Kumar, Phuong Cao

**Abstract**: This study investigates the vulnerabilities of data-driven offline signature verification (DASV) systems to generative attacks and proposes robust countermeasures. Specifically, we explore the efficacy of Variational Autoencoders (VAEs) and Conditional Generative Adversarial Networks (CGANs) in creating deceptive signatures that challenge DASV systems. Using the Structural Similarity Index (SSIM) to evaluate the quality of forged signatures, we assess their impact on DASV systems built with Xception, ResNet152V2, and DenseNet201 architectures. Initial results showed False Accept Rates (FARs) ranging from 0% to 5.47% across all models and datasets. However, exposure to synthetic signatures significantly increased FARs, with rates ranging from 19.12% to 61.64%. The proposed countermeasure, i.e., retraining the models with real + synthetic datasets, was very effective, reducing FARs between 0% and 0.99%. These findings emphasize the necessity of investigating vulnerabilities in security systems like DASV and reinforce the role of generative methods in enhancing the security of data-driven systems.



## **29. Investigating Adversarial Vulnerability and Implicit Bias through Frequency Analysis**

cs.LG

**SubmitDate**: 2024-07-17    [abs](http://arxiv.org/abs/2305.15203v2) [paper-pdf](http://arxiv.org/pdf/2305.15203v2)

**Authors**: Lorenzo Basile, Nikos Karantzas, Alberto D'Onofrio, Luca Bortolussi, Alex Rodriguez, Fabio Anselmi

**Abstract**: Despite their impressive performance in classification tasks, neural networks are known to be vulnerable to adversarial attacks, subtle perturbations of the input data designed to deceive the model. In this work, we investigate the relation between these perturbations and the implicit bias of neural networks trained with gradient-based algorithms. To this end, we analyse the network's implicit bias through the lens of the Fourier transform. Specifically, we identify the minimal and most critical frequencies necessary for accurate classification or misclassification respectively for each input image and its adversarially perturbed version, and uncover the correlation among those. To this end, among other methods, we use a newly introduced technique capable of detecting non-linear correlations between high-dimensional datasets. Our results provide empirical evidence that the network bias in Fourier space and the target frequencies of adversarial attacks are highly correlated and suggest new potential strategies for adversarial defence.



## **30. Muting Whisper: A Universal Acoustic Adversarial Attack on Speech Foundation Models**

cs.CL

**SubmitDate**: 2024-07-17    [abs](http://arxiv.org/abs/2405.06134v2) [paper-pdf](http://arxiv.org/pdf/2405.06134v2)

**Authors**: Vyas Raina, Rao Ma, Charles McGhee, Kate Knill, Mark Gales

**Abstract**: Recent developments in large speech foundation models like Whisper have led to their widespread use in many automatic speech recognition (ASR) applications. These systems incorporate `special tokens' in their vocabulary, such as $\texttt{<|endoftext|>}$, to guide their language generation process. However, we demonstrate that these tokens can be exploited by adversarial attacks to manipulate the model's behavior. We propose a simple yet effective method to learn a universal acoustic realization of Whisper's $\texttt{<|endoftext|>}$ token, which, when prepended to any speech signal, encourages the model to ignore the speech and only transcribe the special token, effectively `muting' the model. Our experiments demonstrate that the same, universal 0.64-second adversarial audio segment can successfully mute a target Whisper ASR model for over 97\% of speech samples. Moreover, we find that this universal adversarial audio segment often transfers to new datasets and tasks. Overall this work demonstrates the vulnerability of Whisper models to `muting' adversarial attacks, where such attacks can pose both risks and potential benefits in real-world settings: for example the attack can be used to bypass speech moderation systems, or conversely the attack can also be used to protect private speech data.



## **31. Similarity of Neural Architectures using Adversarial Attack Transferability**

cs.LG

ECCV 2024; 35pages, 2.56MB

**SubmitDate**: 2024-07-17    [abs](http://arxiv.org/abs/2210.11407v4) [paper-pdf](http://arxiv.org/pdf/2210.11407v4)

**Authors**: Jaehui Hwang, Dongyoon Han, Byeongho Heo, Song Park, Sanghyuk Chun, Jong-Seok Lee

**Abstract**: In recent years, many deep neural architectures have been developed for image classification. Whether they are similar or dissimilar and what factors contribute to their (dis)similarities remains curious. To address this question, we aim to design a quantitative and scalable similarity measure between neural architectures. We propose Similarity by Attack Transferability (SAT) from the observation that adversarial attack transferability contains information related to input gradients and decision boundaries widely used to understand model behaviors. We conduct a large-scale analysis on 69 state-of-the-art ImageNet classifiers using our proposed similarity function to answer the question. Moreover, we observe neural architecture-related phenomena using model similarity that model diversity can lead to better performance on model ensembles and knowledge distillation under specific conditions. Our results provide insights into why developing diverse neural architectures with distinct components is necessary.



## **32. Open-Vocabulary Object Detectors: Robustness Challenges under Distribution Shifts**

cs.CV

14 + 3 single column pages

**SubmitDate**: 2024-07-17    [abs](http://arxiv.org/abs/2405.14874v3) [paper-pdf](http://arxiv.org/pdf/2405.14874v3)

**Authors**: Prakash Chandra Chhipa, Kanjar De, Meenakshi Subhash Chippa, Rajkumar Saini, Marcus Liwicki

**Abstract**: The challenge of Out-Of-Distribution (OOD) robustness remains a critical hurdle towards deploying deep vision models. Vision-Language Models (VLMs) have recently achieved groundbreaking results. VLM-based open-vocabulary object detection extends the capabilities of traditional object detection frameworks, enabling the recognition and classification of objects beyond predefined categories. Investigating OOD robustness in recent open-vocabulary object detection is essential to increase the trustworthiness of these models. This study presents a comprehensive robustness evaluation of the zero-shot capabilities of three recent open-vocabulary (OV) foundation object detection models: OWL-ViT, YOLO World, and Grounding DINO. Experiments carried out on the robustness benchmarks COCO-O, COCO-DC, and COCO-C encompassing distribution shifts due to information loss, corruption, adversarial attacks, and geometrical deformation, highlighting the challenges of the model's robustness to foster the research for achieving robustness. Source code shall be made available to the research community on GitHub.



## **33. Preventing Catastrophic Overfitting in Fast Adversarial Training: A Bi-level Optimization Perspective**

cs.LG

**SubmitDate**: 2024-07-17    [abs](http://arxiv.org/abs/2407.12443v1) [paper-pdf](http://arxiv.org/pdf/2407.12443v1)

**Authors**: Zhaoxin Wang, Handing Wang, Cong Tian, Yaochu Jin

**Abstract**: Adversarial training (AT) has become an effective defense method against adversarial examples (AEs) and it is typically framed as a bi-level optimization problem. Among various AT methods, fast AT (FAT), which employs a single-step attack strategy to guide the training process, can achieve good robustness against adversarial attacks at a low cost. However, FAT methods suffer from the catastrophic overfitting problem, especially on complex tasks or with large-parameter models. In this work, we propose a FAT method termed FGSM-PCO, which mitigates catastrophic overfitting by averting the collapse of the inner optimization problem in the bi-level optimization process. FGSM-PCO generates current-stage AEs from the historical AEs and incorporates them into the training process using an adaptive mechanism. This mechanism determines an appropriate fusion ratio according to the performance of the AEs on the training model. Coupled with a loss function tailored to the training framework, FGSM-PCO can alleviate catastrophic overfitting and help the recovery of an overfitted model to effective training. We evaluate our algorithm across three models and three datasets to validate its effectiveness. Comparative empirical studies against other FAT algorithms demonstrate that our proposed method effectively addresses unresolved overfitting issues in existing algorithms.



## **34. DIFFender: Diffusion-Based Adversarial Defense against Patch Attacks**

cs.CV

**SubmitDate**: 2024-07-17    [abs](http://arxiv.org/abs/2306.09124v4) [paper-pdf](http://arxiv.org/pdf/2306.09124v4)

**Authors**: Caixin Kang, Yinpeng Dong, Zhengyi Wang, Shouwei Ruan, Yubo Chen, Hang Su, Xingxing Wei

**Abstract**: Adversarial attacks, particularly patch attacks, pose significant threats to the robustness and reliability of deep learning models. Developing reliable defenses against patch attacks is crucial for real-world applications. This paper introduces DIFFender, a novel defense framework that harnesses the capabilities of a text-guided diffusion model to combat patch attacks. Central to our approach is the discovery of the Adversarial Anomaly Perception (AAP) phenomenon, which empowers the diffusion model to detect and localize adversarial patches through the analysis of distributional discrepancies. DIFFender integrates dual tasks of patch localization and restoration within a single diffusion model framework, utilizing their close interaction to enhance defense efficacy. Moreover, DIFFender utilizes vision-language pre-training coupled with an efficient few-shot prompt-tuning algorithm, which streamlines the adaptation of the pre-trained diffusion model to defense tasks, thus eliminating the need for extensive retraining. Our comprehensive evaluation spans image classification and face recognition tasks, extending to real-world scenarios, where DIFFender shows good robustness against adversarial attacks. The versatility and generalizability of DIFFender are evident across a variety of settings, classifiers, and attack methodologies, marking an advancement in adversarial patch defense strategies.



## **35. Bribe & Fork: Cheap Bribing Attacks via Forking Threat**

cs.CR

This is a full version of the paper Bribe & Fork: Cheap Bribing  Attacks via Forking Threat which was accepted to AFT'24

**SubmitDate**: 2024-07-17    [abs](http://arxiv.org/abs/2402.01363v2) [paper-pdf](http://arxiv.org/pdf/2402.01363v2)

**Authors**: Zeta Avarikioti, Paweł Kędzior, Tomasz Lizurej, Tomasz Michalak

**Abstract**: In this work, we reexamine the vulnerability of Payment Channel Networks (PCNs) to bribing attacks, where an adversary incentivizes blockchain miners to deliberately ignore a specific transaction to undermine the punishment mechanism of PCNs. While previous studies have posited a prohibitive cost for such attacks, we show that this cost may be dramatically reduced (to approximately \$125), thereby increasing the likelihood of these attacks. To this end, we introduce Bribe & Fork, a modified bribing attack that leverages the threat of a so-called feather fork which we analyze with a novel formal model for the mining game with forking. We empirically analyze historical data of some real-world blockchain implementations to evaluate the scale of this cost reduction. Our findings shed more light on the potential vulnerability of PCNs and highlight the need for robust solutions.



## **36. Revisiting the Adversarial Robustness of Vision Language Models: a Multimodal Perspective**

cs.CV

16 pages, 14 figures

**SubmitDate**: 2024-07-17    [abs](http://arxiv.org/abs/2404.19287v2) [paper-pdf](http://arxiv.org/pdf/2404.19287v2)

**Authors**: Wanqi Zhou, Shuanghao Bai, Qibin Zhao, Badong Chen

**Abstract**: Pretrained vision-language models (VLMs) like CLIP have shown impressive generalization performance across various downstream tasks, yet they remain vulnerable to adversarial attacks. While prior research has primarily concentrated on improving the adversarial robustness of image encoders to guard against attacks on images, the exploration of text-based and multimodal attacks has largely been overlooked. In this work, we initiate the first known and comprehensive effort to study adapting vision-language models for adversarial robustness under the multimodal attack. Firstly, we introduce a multimodal attack strategy and investigate the impact of different attacks. We then propose a multimodal contrastive adversarial training loss, aligning the clean and adversarial text embeddings with the adversarial and clean visual features, to enhance the adversarial robustness of both image and text encoders of CLIP. Extensive experiments on 15 datasets across two tasks demonstrate that our method significantly improves the adversarial robustness of CLIP. Interestingly, we find that the model fine-tuned against multimodal adversarial attacks exhibits greater robustness than its counterpart fine-tuned solely against image-based attacks, even in the context of image attacks, which may open up new possibilities for enhancing the security of VLMs.



## **37. Augmented Neural Fine-Tuning for Efficient Backdoor Purification**

cs.CV

Accepted to ECCV 2024

**SubmitDate**: 2024-07-17    [abs](http://arxiv.org/abs/2407.10052v2) [paper-pdf](http://arxiv.org/pdf/2407.10052v2)

**Authors**: Nazmul Karim, Abdullah Al Arafat, Umar Khalid, Zhishan Guo, Nazanin Rahnavard

**Abstract**: Recent studies have revealed the vulnerability of deep neural networks (DNNs) to various backdoor attacks, where the behavior of DNNs can be compromised by utilizing certain types of triggers or poisoning mechanisms. State-of-the-art (SOTA) defenses employ too-sophisticated mechanisms that require either a computationally expensive adversarial search module for reverse-engineering the trigger distribution or an over-sensitive hyper-parameter selection module. Moreover, they offer sub-par performance in challenging scenarios, e.g., limited validation data and strong attacks. In this paper, we propose Neural mask Fine-Tuning (NFT) with an aim to optimally re-organize the neuron activities in a way that the effect of the backdoor is removed. Utilizing a simple data augmentation like MixUp, NFT relaxes the trigger synthesis process and eliminates the requirement of the adversarial search module. Our study further reveals that direct weight fine-tuning under limited validation data results in poor post-purification clean test accuracy, primarily due to overfitting issue. To overcome this, we propose to fine-tune neural masks instead of model weights. In addition, a mask regularizer has been devised to further mitigate the model drift during the purification process. The distinct characteristics of NFT render it highly efficient in both runtime and sample usage, as it can remove the backdoor even when a single sample is available from each class. We validate the effectiveness of NFT through extensive experiments covering the tasks of image classification, object detection, video action recognition, 3D point cloud, and natural language processing. We evaluate our method against 14 different attacks (LIRA, WaNet, etc.) on 11 benchmark data sets such as ImageNet, UCF101, Pascal VOC, ModelNet, OpenSubtitles2012, etc.



## **38. Asymmetric Bias in Text-to-Image Generation with Adversarial Attacks**

cs.LG

camera-ready version

**SubmitDate**: 2024-07-17    [abs](http://arxiv.org/abs/2312.14440v3) [paper-pdf](http://arxiv.org/pdf/2312.14440v3)

**Authors**: Haz Sameen Shahgir, Xianghao Kong, Greg Ver Steeg, Yue Dong

**Abstract**: The widespread use of Text-to-Image (T2I) models in content generation requires careful examination of their safety, including their robustness to adversarial attacks. Despite extensive research on adversarial attacks, the reasons for their effectiveness remain underexplored. This paper presents an empirical study on adversarial attacks against T2I models, focusing on analyzing factors associated with attack success rates (ASR). We introduce a new attack objective - entity swapping using adversarial suffixes and two gradient-based attack algorithms. Human and automatic evaluations reveal the asymmetric nature of ASRs on entity swap: for example, it is easier to replace "human" with "robot" in the prompt "a human dancing in the rain." with an adversarial suffix, but the reverse replacement is significantly harder. We further propose probing metrics to establish indicative signals from the model's beliefs to the adversarial ASR. We identify conditions that result in a success probability of 60% for adversarial attacks and others where this likelihood drops below 5%.



## **39. Any Target Can be Offense: Adversarial Example Generation via Generalized Latent Infection**

cs.CV

ECCV 2024

**SubmitDate**: 2024-07-17    [abs](http://arxiv.org/abs/2407.12292v1) [paper-pdf](http://arxiv.org/pdf/2407.12292v1)

**Authors**: Youheng Sun, Shengming Yuan, Xuanhan Wang, Lianli Gao, Jingkuan Song

**Abstract**: Targeted adversarial attack, which aims to mislead a model to recognize any image as a target object by imperceptible perturbations, has become a mainstream tool for vulnerability assessment of deep neural networks (DNNs). Since existing targeted attackers only learn to attack known target classes, they cannot generalize well to unknown classes. To tackle this issue, we propose $\bf{G}$eneralized $\bf{A}$dversarial attac$\bf{KER}$ ($\bf{GAKer}$), which is able to construct adversarial examples to any target class. The core idea behind GAKer is to craft a latently infected representation during adversarial example generation. To this end, the extracted latent representations of the target object are first injected into intermediate features of an input image in an adversarial generator. Then, the generator is optimized to ensure visual consistency with the input image while being close to the target object in the feature space. Since the GAKer is class-agnostic yet model-agnostic, it can be regarded as a general tool that not only reveals the vulnerability of more DNNs but also identifies deficiencies of DNNs in a wider range of classes. Extensive experiments have demonstrated the effectiveness of our proposed method in generating adversarial examples for both known and unknown classes. Notably, compared with other generative methods, our method achieves an approximately $14.13\%$ higher attack success rate for unknown classes and an approximately $4.23\%$ higher success rate for known classes. Our code is available in https://github.com/VL-Group/GAKer.



## **40. JailbreakBench: An Open Robustness Benchmark for Jailbreaking Large Language Models**

cs.CR

JailbreakBench v1.0: more attack artifacts, more test-time defenses,  a more accurate jailbreak judge (Llama-3-70B with a custom prompt), a larger  dataset of human preferences for selecting a jailbreak judge (300 examples),  an over-refusal evaluation dataset (100 benign/borderline behaviors), a  semantic refusal judge based on Llama-3-8B

**SubmitDate**: 2024-07-16    [abs](http://arxiv.org/abs/2404.01318v4) [paper-pdf](http://arxiv.org/pdf/2404.01318v4)

**Authors**: Patrick Chao, Edoardo Debenedetti, Alexander Robey, Maksym Andriushchenko, Francesco Croce, Vikash Sehwag, Edgar Dobriban, Nicolas Flammarion, George J. Pappas, Florian Tramer, Hamed Hassani, Eric Wong

**Abstract**: Jailbreak attacks cause large language models (LLMs) to generate harmful, unethical, or otherwise objectionable content. Evaluating these attacks presents a number of challenges, which the current collection of benchmarks and evaluation techniques do not adequately address. First, there is no clear standard of practice regarding jailbreaking evaluation. Second, existing works compute costs and success rates in incomparable ways. And third, numerous works are not reproducible, as they withhold adversarial prompts, involve closed-source code, or rely on evolving proprietary APIs. To address these challenges, we introduce JailbreakBench, an open-sourced benchmark with the following components: (1) an evolving repository of state-of-the-art adversarial prompts, which we refer to as jailbreak artifacts; (2) a jailbreaking dataset comprising 100 behaviors -- both original and sourced from prior work (Zou et al., 2023; Mazeika et al., 2023, 2024) -- which align with OpenAI's usage policies; (3) a standardized evaluation framework at https://github.com/JailbreakBench/jailbreakbench that includes a clearly defined threat model, system prompts, chat templates, and scoring functions; and (4) a leaderboard at https://jailbreakbench.github.io/ that tracks the performance of attacks and defenses for various LLMs. We have carefully considered the potential ethical implications of releasing this benchmark, and believe that it will be a net positive for the community.



## **41. Towards Reliable Evaluation and Fast Training of Robust Semantic Segmentation Models**

cs.CV

ECCV 2024

**SubmitDate**: 2024-07-16    [abs](http://arxiv.org/abs/2306.12941v2) [paper-pdf](http://arxiv.org/pdf/2306.12941v2)

**Authors**: Francesco Croce, Naman D Singh, Matthias Hein

**Abstract**: Adversarial robustness has been studied extensively in image classification, especially for the $\ell_\infty$-threat model, but significantly less so for related tasks such as object detection and semantic segmentation, where attacks turn out to be a much harder optimization problem than for image classification. We propose several problem-specific novel attacks minimizing different metrics in accuracy and mIoU. The ensemble of our attacks, SEA, shows that existing attacks severely overestimate the robustness of semantic segmentation models. Surprisingly, existing attempts of adversarial training for semantic segmentation models turn out to be weak or even completely non-robust. We investigate why previous adaptations of adversarial training to semantic segmentation failed and show how recently proposed robust ImageNet backbones can be used to obtain adversarially robust semantic segmentation models with up to six times less training time for PASCAL-VOC and the more challenging ADE20k. The associated code and robust models are available at https://github.com/nmndeep/robust-segmentation



## **42. Variational Randomized Smoothing for Sample-Wise Adversarial Robustness**

cs.LG

20 pages, under preparation

**SubmitDate**: 2024-07-16    [abs](http://arxiv.org/abs/2407.11844v1) [paper-pdf](http://arxiv.org/pdf/2407.11844v1)

**Authors**: Ryo Hase, Ye Wang, Toshiaki Koike-Akino, Jing Liu, Kieran Parsons

**Abstract**: Randomized smoothing is a defensive technique to achieve enhanced robustness against adversarial examples which are small input perturbations that degrade the performance of neural network models. Conventional randomized smoothing adds random noise with a fixed noise level for every input sample to smooth out adversarial perturbations. This paper proposes a new variational framework that uses a per-sample noise level suitable for each input by introducing a noise level selector. Our experimental results demonstrate enhancement of empirical robustness against adversarial attacks. We also provide and analyze the certified robustness for our sample-wise smoothing method.



## **43. Exploring the Robustness of Decision-Level Through Adversarial Attacks on LLM-Based Embodied Models**

cs.MM

**SubmitDate**: 2024-07-16    [abs](http://arxiv.org/abs/2405.19802v3) [paper-pdf](http://arxiv.org/pdf/2405.19802v3)

**Authors**: Shuyuan Liu, Jiawei Chen, Shouwei Ruan, Hang Su, Zhaoxia Yin

**Abstract**: Embodied intelligence empowers agents with a profound sense of perception, enabling them to respond in a manner closely aligned with real-world situations. Large Language Models (LLMs) delve into language instructions with depth, serving a crucial role in generating plans for intricate tasks. Thus, LLM-based embodied models further enhance the agent's capacity to comprehend and process information. However, this amalgamation also ushers in new challenges in the pursuit of heightened intelligence. Specifically, attackers can manipulate LLMs to produce irrelevant or even malicious outputs by altering their prompts. Confronted with this challenge, we observe a notable absence of multi-modal datasets essential for comprehensively evaluating the robustness of LLM-based embodied models. Consequently, we construct the Embodied Intelligent Robot Attack Dataset (EIRAD), tailored specifically for robustness evaluation. Additionally, two attack strategies are devised, including untargeted attacks and targeted attacks, to effectively simulate a range of diverse attack scenarios. At the same time, during the attack process, to more accurately ascertain whether our method is successful in attacking the LLM-based embodied model, we devise a new attack success evaluation method utilizing the BLIP2 model. Recognizing the time and cost-intensive nature of the GCG algorithm in attacks, we devise a scheme for prompt suffix initialization based on various target tasks, thus expediting the convergence process. Experimental results demonstrate that our method exhibits a superior attack success rate when targeting LLM-based embodied models, indicating a lower level of decision-level robustness in these models.



## **44. Relaxing Graph Transformers for Adversarial Attacks**

cs.LG

**SubmitDate**: 2024-07-16    [abs](http://arxiv.org/abs/2407.11764v1) [paper-pdf](http://arxiv.org/pdf/2407.11764v1)

**Authors**: Philipp Foth, Lukas Gosch, Simon Geisler, Leo Schwinn, Stephan Günnemann

**Abstract**: Existing studies have shown that Graph Neural Networks (GNNs) are vulnerable to adversarial attacks. Even though Graph Transformers (GTs) surpassed Message-Passing GNNs on several benchmarks, their adversarial robustness properties are unexplored. However, attacking GTs is challenging due to their Positional Encodings (PEs) and special attention mechanisms which can be difficult to differentiate. We overcome these challenges by targeting three representative architectures based on (1) random-walk PEs, (2) pair-wise-shortest-path PEs, and (3) spectral PEs - and propose the first adaptive attacks for GTs. We leverage our attacks to evaluate robustness to (a) structure perturbations on node classification; and (b) node injection attacks for (fake-news) graph classification. Our evaluation reveals that they can be catastrophically fragile and underlines our work's importance and the necessity for adaptive attacks.



## **45. AEMIM: Adversarial Examples Meet Masked Image Modeling**

cs.CV

Under review of International Journal of Computer Vision (IJCV)

**SubmitDate**: 2024-07-16    [abs](http://arxiv.org/abs/2407.11537v1) [paper-pdf](http://arxiv.org/pdf/2407.11537v1)

**Authors**: Wenzhao Xiang, Chang Liu, Hang Su, Hongyang Yu

**Abstract**: Masked image modeling (MIM) has gained significant traction for its remarkable prowess in representation learning. As an alternative to the traditional approach, the reconstruction from corrupted images has recently emerged as a promising pretext task. However, the regular corrupted images are generated using generic generators, often lacking relevance to the specific reconstruction task involved in pre-training. Hence, reconstruction from regular corrupted images cannot ensure the difficulty of the pretext task, potentially leading to a performance decline. Moreover, generating corrupted images might introduce an extra generator, resulting in a notable computational burden. To address these issues, we propose to incorporate adversarial examples into masked image modeling, as the new reconstruction targets. Adversarial examples, generated online using only the trained models, can directly aim to disrupt tasks associated with pre-training. Therefore, the incorporation not only elevates the level of challenge in reconstruction but also enhances efficiency, contributing to the acquisition of superior representations by the model. In particular, we introduce a novel auxiliary pretext task that reconstructs the adversarial examples corresponding to the original images. We also devise an innovative adversarial attack to craft more suitable adversarial examples for MIM pre-training. It is noted that our method is not restricted to specific model architectures and MIM strategies, rendering it an adaptable plug-in capable of enhancing all MIM methods. Experimental findings substantiate the remarkable capability of our approach in amplifying the generalization and robustness of existing MIM methods. Notably, our method surpasses the performance of baselines on various tasks, including ImageNet, its variants, and other downstream tasks.



## **46. Learning on Graphs with Large Language Models(LLMs): A Deep Dive into Model Robustness**

cs.LG

**SubmitDate**: 2024-07-16    [abs](http://arxiv.org/abs/2407.12068v1) [paper-pdf](http://arxiv.org/pdf/2407.12068v1)

**Authors**: Kai Guo, Zewen Liu, Zhikai Chen, Hongzhi Wen, Wei Jin, Jiliang Tang, Yi Chang

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable performance across various natural language processing tasks. Recently, several LLMs-based pipelines have been developed to enhance learning on graphs with text attributes, showcasing promising performance. However, graphs are well-known to be susceptible to adversarial attacks and it remains unclear whether LLMs exhibit robustness in learning on graphs. To address this gap, our work aims to explore the potential of LLMs in the context of adversarial attacks on graphs. Specifically, we investigate the robustness against graph structural and textual perturbations in terms of two dimensions: LLMs-as-Enhancers and LLMs-as-Predictors. Through extensive experiments, we find that, compared to shallow models, both LLMs-as-Enhancers and LLMs-as-Predictors offer superior robustness against structural and textual attacks.Based on these findings, we carried out additional analyses to investigate the underlying causes. Furthermore, we have made our benchmark library openly available to facilitate quick and fair evaluations, and to encourage ongoing innovative research in this field.



## **47. Boosting the Transferability of Adversarial Attacks with Global Momentum Initialization**

cs.CV

Accepted by Expert Systems with Applications (ESWA)

**SubmitDate**: 2024-07-16    [abs](http://arxiv.org/abs/2211.11236v3) [paper-pdf](http://arxiv.org/pdf/2211.11236v3)

**Authors**: Jiafeng Wang, Zhaoyu Chen, Kaixun Jiang, Dingkang Yang, Lingyi Hong, Pinxue Guo, Haijing Guo, Wenqiang Zhang

**Abstract**: Deep Neural Networks (DNNs) are vulnerable to adversarial examples, which are crafted by adding human-imperceptible perturbations to the benign inputs. Simultaneously, adversarial examples exhibit transferability across models, enabling practical black-box attacks. However, existing methods are still incapable of achieving the desired transfer attack performance. In this work, focusing on gradient optimization and consistency, we analyse the gradient elimination phenomenon as well as the local momentum optimum dilemma. To tackle these challenges, we introduce Global Momentum Initialization (GI), providing global momentum knowledge to mitigate gradient elimination. Specifically, we perform gradient pre-convergence before the attack and a global search during this stage. GI seamlessly integrates with existing transfer methods, significantly improving the success rate of transfer attacks by an average of 6.4% under various advanced defense mechanisms compared to the state-of-the-art method. Ultimately, GI demonstrates strong transferability in both image and video attack domains. Particularly, when attacking advanced defense methods in the image domain, it achieves an average attack success rate of 95.4%. The code is available at $\href{https://github.com/Omenzychen/Global-Momentum-Initialization}{https://github.com/Omenzychen/Global-Momentum-Initialization}$.



## **48. Investigating Imperceptibility of Adversarial Attacks on Tabular Data: An Empirical Analysis**

cs.LG

33 pages

**SubmitDate**: 2024-07-16    [abs](http://arxiv.org/abs/2407.11463v1) [paper-pdf](http://arxiv.org/pdf/2407.11463v1)

**Authors**: Zhipeng He, Chun Ouyang, Laith Alzubaidi, Alistair Barros, Catarina Moreira

**Abstract**: Adversarial attacks are a potential threat to machine learning models, as they can cause the model to make incorrect predictions by introducing imperceptible perturbations to the input data. While extensively studied in unstructured data like images, their application to structured data like tabular data presents unique challenges due to the heterogeneity and intricate feature interdependencies of tabular data. Imperceptibility in tabular data involves preserving data integrity while potentially causing misclassification, underscoring the need for tailored imperceptibility criteria for tabular data. However, there is currently a lack of standardised metrics for assessing adversarial attacks specifically targeted at tabular data. To address this gap, we derive a set of properties for evaluating the imperceptibility of adversarial attacks on tabular data. These properties are defined to capture seven perspectives of perturbed data: proximity to original inputs, sparsity of alterations, deviation to datapoints in the original dataset, sensitivity of altering sensitive features, immutability of perturbation, feasibility of perturbed values and intricate feature interdepencies among tabular features. Furthermore, we conduct both quantitative empirical evaluation and case-based qualitative examples analysis for seven properties. The evaluation reveals a trade-off between attack success and imperceptibility, particularly concerning proximity, sensitivity, and deviation. Although no evaluated attacks can achieve optimal effectiveness and imperceptibility simultaneously, unbounded attacks prove to be more promised for tabular data in crafting imperceptible adversarial examples. The study also highlights the limitation of evaluated algorithms in controlling sparsity effectively. We suggest incorporating a sparsity metric in future attack design to regulate the number of perturbed features.



## **49. PromptRobust: Towards Evaluating the Robustness of Large Language Models on Adversarial Prompts**

cs.CL

Technical report; code is at:  https://github.com/microsoft/promptbench

**SubmitDate**: 2024-07-16    [abs](http://arxiv.org/abs/2306.04528v5) [paper-pdf](http://arxiv.org/pdf/2306.04528v5)

**Authors**: Kaijie Zhu, Jindong Wang, Jiaheng Zhou, Zichen Wang, Hao Chen, Yidong Wang, Linyi Yang, Wei Ye, Yue Zhang, Neil Zhenqiang Gong, Xing Xie

**Abstract**: The increasing reliance on Large Language Models (LLMs) across academia and industry necessitates a comprehensive understanding of their robustness to prompts. In response to this vital need, we introduce PromptRobust, a robustness benchmark designed to measure LLMs' resilience to adversarial prompts. This study uses a plethora of adversarial textual attacks targeting prompts across multiple levels: character, word, sentence, and semantic. The adversarial prompts, crafted to mimic plausible user errors like typos or synonyms, aim to evaluate how slight deviations can affect LLM outcomes while maintaining semantic integrity. These prompts are then employed in diverse tasks including sentiment analysis, natural language inference, reading comprehension, machine translation, and math problem-solving. Our study generates 4,788 adversarial prompts, meticulously evaluated over 8 tasks and 13 datasets. Our findings demonstrate that contemporary LLMs are not robust to adversarial prompts. Furthermore, we present a comprehensive analysis to understand the mystery behind prompt robustness and its transferability. We then offer insightful robustness analysis and pragmatic recommendations for prompt composition, beneficial to both researchers and everyday users.



## **50. Gradients Look Alike: Sensitivity is Often Overestimated in DP-SGD**

cs.LG

published in 33rd USENIX Security Symposium

**SubmitDate**: 2024-07-16    [abs](http://arxiv.org/abs/2307.00310v3) [paper-pdf](http://arxiv.org/pdf/2307.00310v3)

**Authors**: Anvith Thudi, Hengrui Jia, Casey Meehan, Ilia Shumailov, Nicolas Papernot

**Abstract**: Differentially private stochastic gradient descent (DP-SGD) is the canonical approach to private deep learning. While the current privacy analysis of DP-SGD is known to be tight in some settings, several empirical results suggest that models trained on common benchmark datasets leak significantly less privacy for many datapoints. Yet, despite past attempts, a rigorous explanation for why this is the case has not been reached. Is it because there exist tighter privacy upper bounds when restricted to these dataset settings, or are our attacks not strong enough for certain datapoints? In this paper, we provide the first per-instance (i.e., ``data-dependent") DP analysis of DP-SGD. Our analysis captures the intuition that points with similar neighbors in the dataset enjoy better data-dependent privacy than outliers. Formally, this is done by modifying the per-step privacy analysis of DP-SGD to introduce a dependence on the distribution of model updates computed from a training dataset. We further develop a new composition theorem to effectively use this new per-step analysis to reason about an entire training run. Put all together, our evaluation shows that this novel DP-SGD analysis allows us to now formally show that DP-SGD leaks significantly less privacy for many datapoints (when trained on common benchmarks) than the current data-independent guarantee. This implies privacy attacks will necessarily fail against many datapoints if the adversary does not have sufficient control over the possible training datasets.



