# Latest Adversarial Attack Papers
**update at 2024-03-11 09:24:08**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. Can LLMs Follow Simple Rules?**

cs.AI

Project website: https://eecs.berkeley.edu/~normanmu/llm_rules;  revised content

**SubmitDate**: 2024-03-08    [abs](http://arxiv.org/abs/2311.04235v3) [paper-pdf](http://arxiv.org/pdf/2311.04235v3)

**Authors**: Norman Mu, Sarah Chen, Zifan Wang, Sizhe Chen, David Karamardian, Lulwa Aljeraisy, Basel Alomair, Dan Hendrycks, David Wagner

**Abstract**: As Large Language Models (LLMs) are deployed with increasing real-world responsibilities, it is important to be able to specify and constrain the behavior of these systems in a reliable manner. Model developers may wish to set explicit rules for the model, such as "do not generate abusive content", but these may be circumvented by jailbreaking techniques. Existing evaluations of adversarial attacks and defenses on LLMs generally require either expensive manual review or unreliable heuristic checks. To address this issue, we propose Rule-following Language Evaluation Scenarios (RuLES), a programmatic framework for measuring rule-following ability in LLMs. RuLES consists of 14 simple text scenarios in which the model is instructed to obey various rules while interacting with the user. Each scenario has a programmatic evaluation function to determine whether the model has broken any rules in a conversation. Our evaluations of proprietary and open models show that almost all current models struggle to follow scenario rules, even on straightforward test cases. We also demonstrate that simple optimization attacks suffice to significantly increase failure rates on test cases. We conclude by exploring two potential avenues for improvement: test-time steering and supervised fine-tuning.



## **2. On Practicality of Using ARM TrustZone Trusted Execution Environment for Securing Programmable Logic Controllers**

cs.CR

To appear at ACM AsiaCCS 2024

**SubmitDate**: 2024-03-08    [abs](http://arxiv.org/abs/2403.05448v1) [paper-pdf](http://arxiv.org/pdf/2403.05448v1)

**Authors**: Zhiang Li, Daisuke Mashima, Wen Shei Ong, Ertem Esiner, Zbigniew Kalbarczyk, Ee-Chien Chang

**Abstract**: Programmable logic controllers (PLCs) are crucial devices for implementing automated control in various industrial control systems (ICS), such as smart power grids, water treatment systems, manufacturing, and transportation systems. Owing to their importance, PLCs are often the target of cyber attackers that are aiming at disrupting the operation of ICS, including the nation's critical infrastructure, by compromising the integrity of control logic execution. While a wide range of cybersecurity solutions for ICS have been proposed, they cannot counter strong adversaries with a foothold on the PLC devices, which could manipulate memory, I/O interface, or PLC logic itself. These days, many ICS devices in the market, including PLCs, run on ARM-based processors, and there is a promising security technology called ARM TrustZone, to offer a Trusted Execution Environment (TEE) on embedded devices. Envisioning that such a hardware-assisted security feature becomes available for ICS devices in the near future, this paper investigates the application of the ARM TrustZone TEE technology for enhancing the security of PLC. Our aim is to evaluate the feasibility and practicality of the TEE-based PLCs through the proof-of-concept design and implementation using open-source software such as OP-TEE and OpenPLC. Our evaluation assesses the performance and resource consumption in real-world ICS configurations, and based on the results, we discuss bottlenecks in the OP-TEE secure OS towards a large-scale ICS and desired changes for its application on ICS devices. Our implementation is made available to public for further study and research.



## **3. EVD4UAV: An Altitude-Sensitive Benchmark to Evade Vehicle Detection in UAV**

cs.CV

**SubmitDate**: 2024-03-08    [abs](http://arxiv.org/abs/2403.05422v1) [paper-pdf](http://arxiv.org/pdf/2403.05422v1)

**Authors**: Huiming Sun, Jiacheng Guo, Zibo Meng, Tianyun Zhang, Jianwu Fang, Yuewei Lin, Hongkai Yu

**Abstract**: Vehicle detection in Unmanned Aerial Vehicle (UAV) captured images has wide applications in aerial photography and remote sensing. There are many public benchmark datasets proposed for the vehicle detection and tracking in UAV images. Recent studies show that adding an adversarial patch on objects can fool the well-trained deep neural networks based object detectors, posing security concerns to the downstream tasks. However, the current public UAV datasets might ignore the diverse altitudes, vehicle attributes, fine-grained instance-level annotation in mostly side view with blurred vehicle roof, so none of them is good to study the adversarial patch based vehicle detection attack problem. In this paper, we propose a new dataset named EVD4UAV as an altitude-sensitive benchmark to evade vehicle detection in UAV with 6,284 images and 90,886 fine-grained annotated vehicles. The EVD4UAV dataset has diverse altitudes (50m, 70m, 90m), vehicle attributes (color, type), fine-grained annotation (horizontal and rotated bounding boxes, instance-level mask) in top view with clear vehicle roof. One white-box and two black-box patch based attack methods are implemented to attack three classic deep neural networks based object detectors on EVD4UAV. The experimental results show that these representative attack methods could not achieve the robust altitude-insensitive attack performance.



## **4. The Impact of Quantization on the Robustness of Transformer-based Text Classifiers**

cs.CL

**SubmitDate**: 2024-03-08    [abs](http://arxiv.org/abs/2403.05365v1) [paper-pdf](http://arxiv.org/pdf/2403.05365v1)

**Authors**: Seyed Parsa Neshaei, Yasaman Boreshban, Gholamreza Ghassem-Sani, Seyed Abolghasem Mirroshandel

**Abstract**: Transformer-based models have made remarkable advancements in various NLP areas. Nevertheless, these models often exhibit vulnerabilities when confronted with adversarial attacks. In this paper, we explore the effect of quantization on the robustness of Transformer-based models. Quantization usually involves mapping a high-precision real number to a lower-precision value, aiming at reducing the size of the model at hand. To the best of our knowledge, this work is the first application of quantization on the robustness of NLP models. In our experiments, we evaluate the impact of quantization on BERT and DistilBERT models in text classification using SST-2, Emotion, and MR datasets. We also evaluate the performance of these models against TextFooler, PWWS, and PSO adversarial attacks. Our findings show that quantization significantly improves (by an average of 18.68%) the adversarial accuracy of the models. Furthermore, we compare the effect of quantization versus that of the adversarial training approach on robustness. Our experiments indicate that quantization increases the robustness of the model by 18.80% on average compared to adversarial training without imposing any extra computational overhead during training. Therefore, our results highlight the effectiveness of quantization in improving the robustness of NLP models.



## **5. Hide in Thicket: Generating Imperceptible and Rational Adversarial Perturbations on 3D Point Clouds**

cs.CV

Accepted by CVPR 2024

**SubmitDate**: 2024-03-08    [abs](http://arxiv.org/abs/2403.05247v1) [paper-pdf](http://arxiv.org/pdf/2403.05247v1)

**Authors**: Tianrui Lou, Xiaojun Jia, Jindong Gu, Li Liu, Siyuan Liang, Bangyan He, Xiaochun Cao

**Abstract**: Adversarial attack methods based on point manipulation for 3D point cloud classification have revealed the fragility of 3D models, yet the adversarial examples they produce are easily perceived or defended against. The trade-off between the imperceptibility and adversarial strength leads most point attack methods to inevitably introduce easily detectable outlier points upon a successful attack. Another promising strategy, shape-based attack, can effectively eliminate outliers, but existing methods often suffer significant reductions in imperceptibility due to irrational deformations. We find that concealing deformation perturbations in areas insensitive to human eyes can achieve a better trade-off between imperceptibility and adversarial strength, specifically in parts of the object surface that are complex and exhibit drastic curvature changes. Therefore, we propose a novel shape-based adversarial attack method, HiT-ADV, which initially conducts a two-stage search for attack regions based on saliency and imperceptibility scores, and then adds deformation perturbations in each attack region using Gaussian kernel functions. Additionally, HiT-ADV is extendable to physical attack. We propose that by employing benign resampling and benign rigid transformations, we can further enhance physical adversarial strength with little sacrifice to imperceptibility. Extensive experiments have validated the superiority of our method in terms of adversarial and imperceptible properties in both digital and physical spaces. Our code is avaliable at: https://github.com/TRLou/HiT-ADV.



## **6. Adversarial Sparse Teacher: Defense Against Distillation-Based Model Stealing Attacks Using Adversarial Examples**

cs.LG

12 pages, 3 figures, 6 tables

**SubmitDate**: 2024-03-08    [abs](http://arxiv.org/abs/2403.05181v1) [paper-pdf](http://arxiv.org/pdf/2403.05181v1)

**Authors**: Eda Yilmaz, Hacer Yalim Keles

**Abstract**: Knowledge Distillation (KD) facilitates the transfer of discriminative capabilities from an advanced teacher model to a simpler student model, ensuring performance enhancement without compromising accuracy. It is also exploited for model stealing attacks, where adversaries use KD to mimic the functionality of a teacher model. Recent developments in this domain have been influenced by the Stingy Teacher model, which provided empirical analysis showing that sparse outputs can significantly degrade the performance of student models. Addressing the risk of intellectual property leakage, our work introduces an approach to train a teacher model that inherently protects its logits, influenced by the Nasty Teacher concept. Differing from existing methods, we incorporate sparse outputs of adversarial examples with standard training data to strengthen the teacher's defense against student distillation. Our approach carefully reduces the relative entropy between the original and adversarially perturbed outputs, allowing the model to produce adversarial logits with minimal impact on overall performance. The source codes will be made publicly available soon.



## **7. Warfare:Breaking the Watermark Protection of AI-Generated Content**

cs.CV

**SubmitDate**: 2024-03-08    [abs](http://arxiv.org/abs/2310.07726v3) [paper-pdf](http://arxiv.org/pdf/2310.07726v3)

**Authors**: Guanlin Li, Yifei Chen, Jie Zhang, Jiwei Li, Shangwei Guo, Tianwei Zhang

**Abstract**: AI-Generated Content (AIGC) is gaining great popularity, with many emerging commercial services and applications. These services leverage advanced generative models, such as latent diffusion models and large language models, to generate creative content (e.g., realistic images and fluent sentences) for users. The usage of such generated content needs to be highly regulated, as the service providers need to ensure the users do not violate the usage policies (e.g., abuse for commercialization, generating and distributing unsafe content). A promising solution to achieve this goal is watermarking, which adds unique and imperceptible watermarks on the content for service verification and attribution. Numerous watermarking approaches have been proposed recently. However, in this paper, we show that an adversary can easily break these watermarking mechanisms. Specifically, we consider two possible attacks. (1) Watermark removal: the adversary can easily erase the embedded watermark from the generated content and then use it freely bypassing the regulation of the service provider. (2) Watermark forging: the adversary can create illegal content with forged watermarks from another user, causing the service provider to make wrong attributions. We propose Warfare, a unified methodology to achieve both attacks in a holistic way. The key idea is to leverage a pre-trained diffusion model for content processing and a generative adversarial network for watermark removal or forging. We evaluate Warfare on different datasets and embedding setups. The results prove that it can achieve high success rates while maintaining the quality of the generated content. Compared to existing diffusion model-based attacks, Warfare is 5,050~11,000x faster.



## **8. Benchmarking and Defending Against Indirect Prompt Injection Attacks on Large Language Models**

cs.CL

**SubmitDate**: 2024-03-08    [abs](http://arxiv.org/abs/2312.14197v3) [paper-pdf](http://arxiv.org/pdf/2312.14197v3)

**Authors**: Jingwei Yi, Yueqi Xie, Bin Zhu, Emre Kiciman, Guangzhong Sun, Xing Xie, Fangzhao Wu

**Abstract**: The integration of large language models (LLMs) with external content has enabled more up-to-date and wide-ranging applications of LLMs, such as Microsoft Copilot. However, this integration has also exposed LLMs to the risk of indirect prompt injection attacks, where an attacker can embed malicious instructions within external content, compromising LLM output and causing responses to deviate from user expectations. To investigate this important but underexplored issue, we introduce the first benchmark for indirect prompt injection attacks, named BIPIA, to evaluate the risk of such attacks. Based on the evaluation, our work makes a key analysis of the underlying reason for the success of the attack, namely the inability of LLMs to distinguish between instructions and external content and the absence of LLMs' awareness to not execute instructions within external content. Building upon this analysis, we develop two black-box methods based on prompt learning and a white-box defense method based on fine-tuning with adversarial training accordingly. Experimental results demonstrate that black-box defenses are highly effective in mitigating these attacks, while the white-box defense reduces the attack success rate to near-zero levels. Overall, our work systematically investigates indirect prompt injection attacks by introducing a benchmark, analyzing the underlying reason for the success of the attack, and developing an initial set of defenses.



## **9. Exploring the Adversarial Frontier: Quantifying Robustness via Adversarial Hypervolume**

cs.CR

**SubmitDate**: 2024-03-08    [abs](http://arxiv.org/abs/2403.05100v1) [paper-pdf](http://arxiv.org/pdf/2403.05100v1)

**Authors**: Ping Guo, Cheng Gong, Xi Lin, Zhiyuan Yang, Qingfu Zhang

**Abstract**: The escalating threat of adversarial attacks on deep learning models, particularly in security-critical fields, has underscored the need for robust deep learning systems. Conventional robustness evaluations have relied on adversarial accuracy, which measures a model's performance under a specific perturbation intensity. However, this singular metric does not fully encapsulate the overall resilience of a model against varying degrees of perturbation. To address this gap, we propose a new metric termed adversarial hypervolume, assessing the robustness of deep learning models comprehensively over a range of perturbation intensities from a multi-objective optimization standpoint. This metric allows for an in-depth comparison of defense mechanisms and recognizes the trivial improvements in robustness afforded by less potent defensive strategies. Additionally, we adopt a novel training algorithm that enhances adversarial robustness uniformly across various perturbation intensities, in contrast to methods narrowly focused on optimizing adversarial accuracy. Our extensive empirical studies validate the effectiveness of the adversarial hypervolume metric, demonstrating its ability to reveal subtle differences in robustness that adversarial accuracy overlooks. This research contributes a new measure of robustness and establishes a standard for assessing and benchmarking the resilience of current and future defensive models against adversarial threats.



## **10. Defending Against Unforeseen Failure Modes with Latent Adversarial Training**

cs.CR

**SubmitDate**: 2024-03-08    [abs](http://arxiv.org/abs/2403.05030v1) [paper-pdf](http://arxiv.org/pdf/2403.05030v1)

**Authors**: Stephen Casper, Lennart Schulze, Oam Patel, Dylan Hadfield-Menell

**Abstract**: AI systems sometimes exhibit harmful unintended behaviors post-deployment. This is often despite extensive diagnostics and debugging by developers. Minimizing risks from models is challenging because the attack surface is so large. It is not tractable to exhaustively search for inputs that may cause a model to fail. Red-teaming and adversarial training (AT) are commonly used to make AI systems more robust. However, they have not been sufficient to avoid many real-world failure modes that differ from the ones adversarially trained on. In this work, we utilize latent adversarial training (LAT) to defend against vulnerabilities without generating inputs that elicit them. LAT leverages the compressed, abstract, and structured latent representations of concepts that the network actually uses for prediction. We use LAT to remove trojans and defend against held-out classes of adversarial attacks. We show in image classification, text classification, and text generation tasks that LAT usually improves both robustness and performance on clean data relative to AT. This suggests that LAT can be a promising tool for defending against failure modes that are not explicitly identified by developers.



## **11. Fooling Neural Networks for Motion Forecasting via Adversarial Attacks**

cs.CV

11 pages, 8 figures, VISSAP 2024

**SubmitDate**: 2024-03-07    [abs](http://arxiv.org/abs/2403.04954v1) [paper-pdf](http://arxiv.org/pdf/2403.04954v1)

**Authors**: Edgar Medina, Leyong Loh

**Abstract**: Human motion prediction is still an open problem, which is extremely important for autonomous driving and safety applications. Although there are great advances in this area, the widely studied topic of adversarial attacks has not been applied to multi-regression models such as GCNs and MLP-based architectures in human motion prediction. This work intends to reduce this gap using extensive quantitative and qualitative experiments in state-of-the-art architectures similar to the initial stages of adversarial attacks in image classification. The results suggest that models are susceptible to attacks even on low levels of perturbation. We also show experiments with 3D transformations that affect the model performance, in particular, we show that most models are sensitive to simple rotations and translations which do not alter joint distances. We conclude that similar to earlier CNN models, motion forecasting tasks are susceptible to small perturbations and simple 3D transformations.



## **12. Optimal Denial-of-Service Attacks Against Status Updating**

cs.IT

**SubmitDate**: 2024-03-07    [abs](http://arxiv.org/abs/2403.04489v1) [paper-pdf](http://arxiv.org/pdf/2403.04489v1)

**Authors**: Saad Kriouile, Mohamad Assaad, Deniz Gündüz, Touraj Soleymani

**Abstract**: In this paper, we investigate denial-of-service attacks against status updating. The target system is modeled by a Markov chain and an unreliable wireless channel, and the performance of status updating in the target system is measured based on two metrics: age of information and age of incorrect information. Our objective is to devise optimal attack policies that strike a balance between the deterioration of the system's performance and the adversary's energy. We model the optimal problem as a Markov decision process and prove rigorously that the optimal jamming policy is a threshold-based policy under both metrics. In addition, we provide a low-complexity algorithm to obtain the optimal threshold value of the jamming policy. Our numerical results show that the networked system with the age-of-incorrect-information metric is less sensitive to jamming attacks than with the age-of-information metric. Index Terms-age of incorrect information, age of information, cyber-physical systems, status updating, remote monitoring.



## **13. Pilot Spoofing Attack on the Downlink of Cell-Free Massive MIMO: From the Perspective of Adversaries**

cs.IT

**SubmitDate**: 2024-03-07    [abs](http://arxiv.org/abs/2403.04435v1) [paper-pdf](http://arxiv.org/pdf/2403.04435v1)

**Authors**: Weiyang Xu, Yuan Zhang, Ruiguang Wang, Hien Quoc Ngo, Wei Xiang

**Abstract**: The channel hardening effect is less pronounced in the cell-free massive multiple-input multiple-output (mMIMO) system compared to its cellular counterpart, making it necessary to estimate the downlink effective channel gains to ensure decent performance. However, the downlink training inadvertently creates an opportunity for adversarial nodes to launch pilot spoofing attacks (PSAs). First, we demonstrate that adversarial distributed access points (APs) can severely degrade the achievable downlink rate. They achieve this by estimating their channels to users in the uplink training phase and then precoding and sending the same pilot sequences as those used by legitimate APs during the downlink training phase. Then, the impact of the downlink PSA is investigated by rigorously deriving a closed-form expression of the per-user achievable downlink rate. By employing the min-max criterion to optimize the power allocation coefficients, the maximum per-user achievable rate of downlink transmission is minimized from the perspective of adversarial APs. As an alternative to the downlink PSA, adversarial APs may opt to precode random interference during the downlink data transmission phase in order to disrupt legitimate communications. In this scenario, the achievable downlink rate is derived, and then power optimization algorithms are also developed. We present numerical results to showcase the detrimental impact of the downlink PSA and compare the effects of these two types of attacks.



## **14. Evaluating the security of CRYSTALS-Dilithium in the quantum random oracle model**

cs.CR

23 pages; v2: added description of CRYSTALS-Dilithium, improved  analysis of concrete parameters

**SubmitDate**: 2024-03-07    [abs](http://arxiv.org/abs/2312.16619v2) [paper-pdf](http://arxiv.org/pdf/2312.16619v2)

**Authors**: Kelsey A. Jackson, Carl A. Miller, Daochen Wang

**Abstract**: In the wake of recent progress on quantum computing hardware, the National Institute of Standards and Technology (NIST) is standardizing cryptographic protocols that are resistant to attacks by quantum adversaries. The primary digital signature scheme that NIST has chosen is CRYSTALS-Dilithium. The hardness of this scheme is based on the hardness of three computational problems: Module Learning with Errors (MLWE), Module Short Integer Solution (MSIS), and SelfTargetMSIS. MLWE and MSIS have been well-studied and are widely believed to be secure. However, SelfTargetMSIS is novel and, though classically as hard as MSIS, its quantum hardness is unclear. In this paper, we provide the first proof of the hardness of SelfTargetMSIS via a reduction from MLWE in the Quantum Random Oracle Model (QROM). Our proof uses recently developed techniques in quantum reprogramming and rewinding. A central part of our approach is a proof that a certain hash function, derived from the MSIS problem, is collapsing. From this approach, we deduce a new security proof for Dilithium under appropriate parameter settings. Compared to the previous work by Kiltz, Lyubashevsky, and Schaffner (EUROCRYPT 2018) that gave the only other rigorous security proof for a variant of Dilithium, our proof has the advantage of being applicable under the condition q = 1 mod 2n, where q denotes the modulus and n the dimension of the underlying algebraic ring. This condition is part of the original Dilithium proposal and is crucial for the efficient implementation of the scheme. We provide new secure parameter sets for Dilithium under the condition q = 1 mod 2n, finding that our public key size and signature size are about 2.9 times and 1.3 times larger, respectively, than those proposed by Kiltz et al. at the same security level.



## **15. Multi-Agent Reinforcement Learning for Assessing False-Data Injection Attacks on Transportation Networks**

cs.AI

**SubmitDate**: 2024-03-06    [abs](http://arxiv.org/abs/2312.14625v2) [paper-pdf](http://arxiv.org/pdf/2312.14625v2)

**Authors**: Taha Eghtesad, Sirui Li, Yevgeniy Vorobeychik, Aron Laszka

**Abstract**: The increasing reliance of drivers on navigation applications has made transportation networks more susceptible to data-manipulation attacks by malicious actors. Adversaries may exploit vulnerabilities in the data collection or processing of navigation services to inject false information, and to thus interfere with the drivers' route selection. Such attacks can significantly increase traffic congestions, resulting in substantial waste of time and resources, and may even disrupt essential services that rely on road networks. To assess the threat posed by such attacks, we introduce a computational framework to find worst-case data-injection attacks against transportation networks. First, we devise an adversarial model with a threat actor who can manipulate drivers by increasing the travel times that they perceive on certain roads. Then, we employ hierarchical multi-agent reinforcement learning to find an approximate optimal adversarial strategy for data manipulation. We demonstrate the applicability of our approach through simulating attacks on the Sioux Falls, ND network topology.



## **16. Improving Adversarial Training using Vulnerability-Aware Perturbation Budget**

cs.LG

19 pages, 2 figures

**SubmitDate**: 2024-03-06    [abs](http://arxiv.org/abs/2403.04070v1) [paper-pdf](http://arxiv.org/pdf/2403.04070v1)

**Authors**: Olukorede Fakorede, Modeste Atsague, Jin Tian

**Abstract**: Adversarial Training (AT) effectively improves the robustness of Deep Neural Networks (DNNs) to adversarial attacks. Generally, AT involves training DNN models with adversarial examples obtained within a pre-defined, fixed perturbation bound. Notably, individual natural examples from which these adversarial examples are crafted exhibit varying degrees of intrinsic vulnerabilities, and as such, crafting adversarial examples with fixed perturbation radius for all instances may not sufficiently unleash the potency of AT. Motivated by this observation, we propose two simple, computationally cheap vulnerability-aware reweighting functions for assigning perturbation bounds to adversarial examples used for AT, named Margin-Weighted Perturbation Budget (MWPB) and Standard-Deviation-Weighted Perturbation Budget (SDWPB). The proposed methods assign perturbation radii to individual adversarial samples based on the vulnerability of their corresponding natural examples. Experimental results show that the proposed methods yield genuine improvements in the robustness of AT algorithms against various adversarial attacks.



## **17. Improving Adversarial Attacks on Latent Diffusion Model**

cs.CV

**SubmitDate**: 2024-03-06    [abs](http://arxiv.org/abs/2310.04687v3) [paper-pdf](http://arxiv.org/pdf/2310.04687v3)

**Authors**: Boyang Zheng, Chumeng Liang, Xiaoyu Wu, Yan Liu

**Abstract**: Adversarial attacks on Latent Diffusion Model (LDM), the state-of-the-art image generative model, have been adopted as effective protection against malicious finetuning of LDM on unauthorized images. We show that these attacks add an extra error to the score function of adversarial examples predicted by LDM. LDM finetuned on these adversarial examples learns to lower the error by a bias, from which the model is attacked and predicts the score function with biases.   Based on the dynamics, we propose to improve the adversarial attack on LDM by Attacking with Consistent score-function Errors (ACE). ACE unifies the pattern of the extra error added to the predicted score function. This induces the finetuned LDM to learn the same pattern as a bias in predicting the score function. We then introduce a well-crafted pattern to improve the attack. Our method outperforms state-of-the-art methods in adversarial attacks on LDM.



## **18. A Survey on Adversarial Contention Resolution**

cs.DC

**SubmitDate**: 2024-03-06    [abs](http://arxiv.org/abs/2403.03876v1) [paper-pdf](http://arxiv.org/pdf/2403.03876v1)

**Authors**: Ioana Banicescu, Trisha Chakraborty, Seth Gilbert, Maxwell Young

**Abstract**: Contention resolution addresses the challenge of coordinating access by multiple processes to a shared resource such as memory, disk storage, or a communication channel. Originally spurred by challenges in database systems and bus networks, contention resolution has endured as an important abstraction for resource sharing, despite decades of technological change. Here, we survey the literature on resolving worst-case contention, where the number of processes and the time at which each process may start seeking access to the resource is dictated by an adversary. We highlight the evolution of contention resolution, where new concerns -- such as security, quality of service, and energy efficiency -- are motivated by modern systems. These efforts have yielded insights into the limits of randomized and deterministic approaches, as well as the impact of different model assumptions such as global clock synchronization, knowledge of the number of processors, feedback from access attempts, and attacks on the availability of the shared resource.



## **19. Effect of Ambient-Intrinsic Dimension Gap on Adversarial Vulnerability**

cs.LG

**SubmitDate**: 2024-03-06    [abs](http://arxiv.org/abs/2403.03967v1) [paper-pdf](http://arxiv.org/pdf/2403.03967v1)

**Authors**: Rajdeep Haldar, Yue Xing, Qifan Song

**Abstract**: The existence of adversarial attacks on machine learning models imperceptible to a human is still quite a mystery from a theoretical perspective. In this work, we introduce two notions of adversarial attacks: natural or on-manifold attacks, which are perceptible by a human/oracle, and unnatural or off-manifold attacks, which are not. We argue that the existence of the off-manifold attacks is a natural consequence of the dimension gap between the intrinsic and ambient dimensions of the data. For 2-layer ReLU networks, we prove that even though the dimension gap does not affect generalization performance on samples drawn from the observed data space, it makes the clean-trained model more vulnerable to adversarial perturbations in the off-manifold direction of the data space. Our main results provide an explicit relationship between the $\ell_2,\ell_{\infty}$ attack strength of the on/off-manifold attack and the dimension gap.



## **20. Neural Exec: Learning (and Learning from) Execution Triggers for Prompt Injection Attacks**

cs.CR

v0.1

**SubmitDate**: 2024-03-06    [abs](http://arxiv.org/abs/2403.03792v1) [paper-pdf](http://arxiv.org/pdf/2403.03792v1)

**Authors**: Dario Pasquini, Martin Strohmeier, Carmela Troncoso

**Abstract**: We introduce a new family of prompt injection attacks, termed Neural Exec. Unlike known attacks that rely on handcrafted strings (e.g., "Ignore previous instructions and..."), we show that it is possible to conceptualize the creation of execution triggers as a differentiable search problem and use learning-based methods to autonomously generate them.   Our results demonstrate that a motivated adversary can forge triggers that are not only drastically more effective than current handcrafted ones but also exhibit inherent flexibility in shape, properties, and functionality. In this direction, we show that an attacker can design and generate Neural Execs capable of persisting through multi-stage preprocessing pipelines, such as in the case of Retrieval-Augmented Generation (RAG)-based applications. More critically, our findings show that attackers can produce triggers that deviate markedly in form and shape from any known attack, sidestepping existing blacklist-based detection and sanitation approaches.



## **21. PPTC-R benchmark: Towards Evaluating the Robustness of Large Language Models for PowerPoint Task Completion**

cs.CL

LLM evaluation, Multi-turn, Multi-language, Multi-modal benchmark

**SubmitDate**: 2024-03-06    [abs](http://arxiv.org/abs/2403.03788v1) [paper-pdf](http://arxiv.org/pdf/2403.03788v1)

**Authors**: Zekai Zhang, Yiduo Guo, Yaobo Liang, Dongyan Zhao, Nan Duan

**Abstract**: The growing dependence on Large Language Models (LLMs) for finishing user instructions necessitates a comprehensive understanding of their robustness to complex task completion in real-world situations. To address this critical need, we propose the PowerPoint Task Completion Robustness benchmark (PPTC-R) to measure LLMs' robustness to the user PPT task instruction and software version. Specifically, we construct adversarial user instructions by attacking user instructions at sentence, semantic, and multi-language levels. To assess the robustness of Language Models to software versions, we vary the number of provided APIs to simulate both the newest version and earlier version settings. Subsequently, we test 3 closed-source and 4 open-source LLMs using a benchmark that incorporates these robustness settings, aiming to evaluate how deviations impact LLMs' API calls for task completion. We find that GPT-4 exhibits the highest performance and strong robustness in our benchmark, particularly in the version update and the multilingual settings. However, we find that all LLMs lose their robustness when confronted with multiple challenges (e.g., multi-turn) simultaneously, leading to significant performance drops. We further analyze the robustness behavior and error reasons of LLMs in our benchmark, which provide valuable insights for researchers to understand the LLM's robustness in task completion and develop more robust LLMs and agents. We release the code and data at \url{https://github.com/ZekaiGalaxy/PPTCR}.



## **22. Verification of Neural Networks' Global Robustness**

cs.LG

**SubmitDate**: 2024-03-06    [abs](http://arxiv.org/abs/2402.19322v2) [paper-pdf](http://arxiv.org/pdf/2402.19322v2)

**Authors**: Anan Kabaha, Dana Drachsler-Cohen

**Abstract**: Neural networks are successful in various applications but are also susceptible to adversarial attacks. To show the safety of network classifiers, many verifiers have been introduced to reason about the local robustness of a given input to a given perturbation. While successful, local robustness cannot generalize to unseen inputs. Several works analyze global robustness properties, however, neither can provide a precise guarantee about the cases where a network classifier does not change its classification. In this work, we propose a new global robustness property for classifiers aiming at finding the minimal globally robust bound, which naturally extends the popular local robustness property for classifiers. We introduce VHAGaR, an anytime verifier for computing this bound. VHAGaR relies on three main ideas: encoding the problem as a mixed-integer programming and pruning the search space by identifying dependencies stemming from the perturbation or the network's computation and generalizing adversarial attacks to unknown inputs. We evaluate VHAGaR on several datasets and classifiers and show that, given a three hour timeout, the average gap between the lower and upper bound on the minimal globally robust bound computed by VHAGaR is 1.9, while the gap of an existing global robustness verifier is 154.7. Moreover, VHAGaR is 130.6x faster than this verifier. Our results further indicate that leveraging dependencies and adversarial attacks makes VHAGaR 78.6x faster.



## **23. Simplified PCNet with Robustness**

cs.LG

10 pages, 3 figures

**SubmitDate**: 2024-03-06    [abs](http://arxiv.org/abs/2403.03676v1) [paper-pdf](http://arxiv.org/pdf/2403.03676v1)

**Authors**: Bingheng Li, Xuanting Xie, Haoxiang Lei, Ruiyi Fang, Zhao Kang

**Abstract**: Graph Neural Networks (GNNs) have garnered significant attention for their success in learning the representation of homophilic or heterophilic graphs. However, they cannot generalize well to real-world graphs with different levels of homophily. In response, the Possion-Charlier Network (PCNet) \cite{li2024pc}, the previous work, allows graph representation to be learned from heterophily to homophily. Although PCNet alleviates the heterophily issue, there remain some challenges in further improving the efficacy and efficiency. In this paper, we simplify PCNet and enhance its robustness. We first extend the filter order to continuous values and reduce its parameters. Two variants with adaptive neighborhood sizes are implemented. Theoretical analysis shows our model's robustness to graph structure perturbations or adversarial attacks. We validate our approach through semi-supervised learning tasks on various datasets representing both homophilic and heterophilic graphs.



## **24. Adversarial Infrared Geometry: Using Geometry to Perform Adversarial Attack against Infrared Pedestrian Detectors**

cs.CV

arXiv admin note: text overlap with arXiv:2312.14217 by other authors

**SubmitDate**: 2024-03-06    [abs](http://arxiv.org/abs/2403.03674v1) [paper-pdf](http://arxiv.org/pdf/2403.03674v1)

**Authors**: Kalibinuer Tiliwalidi

**Abstract**: Currently, infrared imaging technology enjoys widespread usage, with infrared object detection technology experiencing a surge in prominence. While previous studies have delved into physical attacks on infrared object detectors, the implementation of these techniques remains complex. For instance, some approaches entail the use of bulb boards or infrared QR suits as perturbations to execute attacks, which entail costly optimization and cumbersome deployment processes. Other methodologies involve the utilization of irregular aerogel as physical perturbations for infrared attacks, albeit at the expense of optimization expenses and perceptibility issues. In this study, we propose a novel infrared physical attack termed Adversarial Infrared Geometry (\textbf{AdvIG}), which facilitates efficient black-box query attacks by modeling diverse geometric shapes (lines, triangles, ellipses) and optimizing their physical parameters using Particle Swarm Optimization (PSO). Extensive experiments are conducted to evaluate the effectiveness, stealthiness, and robustness of AdvIG. In digital attack experiments, line, triangle, and ellipse patterns achieve attack success rates of 93.1\%, 86.8\%, and 100.0\%, respectively, with average query times of 71.7, 113.1, and 2.57, respectively, thereby confirming the efficiency of AdvIG. Physical attack experiments are conducted to assess the attack success rate of AdvIG at different distances. On average, the line, triangle, and ellipse achieve attack success rates of 61.1\%, 61.2\%, and 96.2\%, respectively. Further experiments are conducted to comprehensively analyze AdvIG, including ablation experiments, transfer attack experiments, and adversarial defense mechanisms. Given the superior performance of our method as a simple and efficient black-box adversarial attack in both digital and physical environments, we advocate for widespread attention to AdvIG.



## **25. Lotto: Secure Participant Selection against Adversarial Servers in Federated Learning**

cs.CR

This article has been accepted to USENIX Security '24

**SubmitDate**: 2024-03-06    [abs](http://arxiv.org/abs/2401.02880v2) [paper-pdf](http://arxiv.org/pdf/2401.02880v2)

**Authors**: Zhifeng Jiang, Peng Ye, Shiqi He, Wei Wang, Ruichuan Chen, Bo Li

**Abstract**: In Federated Learning (FL), common privacy-enhancing techniques, such as secure aggregation and distributed differential privacy, rely on the critical assumption of an honest majority among participants to withstand various attacks. In practice, however, servers are not always trusted, and an adversarial server can strategically select compromised clients to create a dishonest majority, thereby undermining the system's security guarantees. In this paper, we present Lotto, an FL system that addresses this fundamental, yet underexplored issue by providing secure participant selection against an adversarial server. Lotto supports two selection algorithms: random and informed. To ensure random selection without a trusted server, Lotto enables each client to autonomously determine their participation using verifiable randomness. For informed selection, which is more vulnerable to manipulation, Lotto approximates the algorithm by employing random selection within a refined client pool. Our theoretical analysis shows that Lotto effectively aligns the proportion of server-selected compromised participants with the base rate of dishonest clients in the population. Large-scale experiments further reveal that Lotto achieves time-to-accuracy performance comparable to that of insecure selection methods, indicating a low computational overhead for secure selection.



## **26. Noise-BERT: A Unified Perturbation-Robust Framework with Noise Alignment Pre-training for Noisy Slot Filling Task**

cs.CL

Accepted by ICASSP 2024

**SubmitDate**: 2024-03-06    [abs](http://arxiv.org/abs/2402.14494v3) [paper-pdf](http://arxiv.org/pdf/2402.14494v3)

**Authors**: Jinxu Zhao, Guanting Dong, Yueyan Qiu, Tingfeng Hui, Xiaoshuai Song, Daichi Guo, Weiran Xu

**Abstract**: In a realistic dialogue system, the input information from users is often subject to various types of input perturbations, which affects the slot-filling task. Although rule-based data augmentation methods have achieved satisfactory results, they fail to exhibit the desired generalization when faced with unknown noise disturbances. In this study, we address the challenges posed by input perturbations in slot filling by proposing Noise-BERT, a unified Perturbation-Robust Framework with Noise Alignment Pre-training. Our framework incorporates two Noise Alignment Pre-training tasks: Slot Masked Prediction and Sentence Noisiness Discrimination, aiming to guide the pre-trained language model in capturing accurate slot information and noise distribution. During fine-tuning, we employ a contrastive learning loss to enhance the semantic representation of entities and labels. Additionally, we introduce an adversarial attack training strategy to improve the model's robustness. Experimental results demonstrate the superiority of our proposed approach over state-of-the-art models, and further analysis confirms its effectiveness and generalization ability.



## **27. TTPXHunter: Actionable Threat Intelligence Extraction as TTPs form Finished Cyber Threat Reports**

cs.CR

**SubmitDate**: 2024-03-05    [abs](http://arxiv.org/abs/2403.03267v1) [paper-pdf](http://arxiv.org/pdf/2403.03267v1)

**Authors**: Nanda Rani, Bikash Saha, Vikas Maurya, Sandeep Kumar Shukla

**Abstract**: Understanding the modus operandi of adversaries aids organizations in employing efficient defensive strategies and sharing intelligence in the community. This knowledge is often present in unstructured natural language text within threat analysis reports. A translation tool is needed to interpret the modus operandi explained in the sentences of the threat report and translate it into a structured format. This research introduces a methodology named TTPXHunter for the automated extraction of threat intelligence in terms of Tactics, Techniques, and Procedures (TTPs) from finished cyber threat reports. It leverages cyber domain-specific state-of-the-art natural language processing (NLP) to augment sentences for minority class TTPs and refine pinpointing the TTPs in threat analysis reports significantly. The knowledge of threat intelligence in terms of TTPs is essential for comprehensively understanding cyber threats and enhancing detection and mitigation strategies. We create two datasets: an augmented sentence-TTP dataset of 39,296 samples and a 149 real-world cyber threat intelligence report-to-TTP dataset. Further, we evaluate TTPXHunter on the augmented sentence dataset and the cyber threat reports. The TTPXHunter achieves the highest performance of 92.42% f1-score on the augmented dataset, and it also outperforms existing state-of-the-art solutions in TTP extraction by achieving an f1-score of 97.09% when evaluated over the report dataset. TTPXHunter significantly improves cybersecurity threat intelligence by offering quick, actionable insights into attacker behaviors. This advancement automates threat intelligence analysis, providing a crucial tool for cybersecurity professionals fighting cyber threats.



## **28. Attacks on Node Attributes in Graph Neural Networks**

cs.SI

Accepted to AAAI 2024 AICS workshop

**SubmitDate**: 2024-03-05    [abs](http://arxiv.org/abs/2402.12426v2) [paper-pdf](http://arxiv.org/pdf/2402.12426v2)

**Authors**: Ying Xu, Michael Lanier, Anindya Sarkar, Yevgeniy Vorobeychik

**Abstract**: Graphs are commonly used to model complex networks prevalent in modern social media and literacy applications. Our research investigates the vulnerability of these graphs through the application of feature based adversarial attacks, focusing on both decision time attacks and poisoning attacks. In contrast to state of the art models like Net Attack and Meta Attack, which target node attributes and graph structure, our study specifically targets node attributes. For our analysis, we utilized the text dataset Hellaswag and graph datasets Cora and CiteSeer, providing a diverse basis for evaluation. Our findings indicate that decision time attacks using Projected Gradient Descent (PGD) are more potent compared to poisoning attacks that employ Mean Node Embeddings and Graph Contrastive Learning strategies. This provides insights for graph data security, pinpointing where graph-based models are most vulnerable and thereby informing the development of stronger defense mechanisms against such attacks.



## **29. Mitigating Label Flipping Attacks in Malicious URL Detectors Using Ensemble Trees**

cs.CR

**SubmitDate**: 2024-03-05    [abs](http://arxiv.org/abs/2403.02995v1) [paper-pdf](http://arxiv.org/pdf/2403.02995v1)

**Authors**: Ehsan Nowroozi, Nada Jadalla, Samaneh Ghelichkhani, Alireza Jolfaei

**Abstract**: Malicious URLs provide adversarial opportunities across various industries, including transportation, healthcare, energy, and banking which could be detrimental to business operations. Consequently, the detection of these URLs is of crucial importance; however, current Machine Learning (ML) models are susceptible to backdoor attacks. These attacks involve manipulating a small percentage of training data labels, such as Label Flipping (LF), which changes benign labels to malicious ones and vice versa. This manipulation results in misclassification and leads to incorrect model behavior. Therefore, integrating defense mechanisms into the architecture of ML models becomes an imperative consideration to fortify against potential attacks.   The focus of this study is on backdoor attacks in the context of URL detection using ensemble trees. By illuminating the motivations behind such attacks, highlighting the roles of attackers, and emphasizing the critical importance of effective defense strategies, this paper contributes to the ongoing efforts to fortify ML models against adversarial threats within the ML domain in network security. We propose an innovative alarm system that detects the presence of poisoned labels and a defense mechanism designed to uncover the original class labels with the aim of mitigating backdoor attacks on ensemble tree classifiers. We conducted a case study using the Alexa and Phishing Site URL datasets and showed that LF attacks can be addressed using our proposed defense mechanism. Our experimental results prove that the LF attack achieved an Attack Success Rate (ASR) between 50-65% within 2-5%, and the innovative defense method successfully detected poisoned labels with an accuracy of up to 100%.



## **30. Federated Learning Under Attack: Exposing Vulnerabilities through Data Poisoning Attacks in Computer Networks**

cs.CR

**SubmitDate**: 2024-03-05    [abs](http://arxiv.org/abs/2403.02983v1) [paper-pdf](http://arxiv.org/pdf/2403.02983v1)

**Authors**: Ehsan Nowroozi, Imran Haider, Rahim Taheri, Mauro Conti

**Abstract**: Federated Learning (FL) is a machine learning (ML) approach that enables multiple decentralized devices or edge servers to collaboratively train a shared model without exchanging raw data. During the training and sharing of model updates between clients and servers, data and models are susceptible to different data-poisoning attacks.   In this study, our motivation is to explore the severity of data poisoning attacks in the computer network domain because they are easy to implement but difficult to detect. We considered two types of data-poisoning attacks, label flipping (LF) and feature poisoning (FP), and applied them with a novel approach. In LF, we randomly flipped the labels of benign data and trained the model on the manipulated data. For FP, we randomly manipulated the highly contributing features determined using the Random Forest algorithm. The datasets used in this experiment were CIC and UNSW related to computer networks. We generated adversarial samples using the two attacks mentioned above, which were applied to a small percentage of datasets. Subsequently, we trained and tested the accuracy of the model on adversarial datasets. We recorded the results for both benign and manipulated datasets and observed significant differences between the accuracy of the models on different datasets. From the experimental results, it is evident that the LF attack failed, whereas the FP attack showed effective results, which proved its significance in fooling a server. With a 1% LF attack on the CIC, the accuracy was approximately 0.0428 and the ASR was 0.9564; hence, the attack is easily detectable, while with a 1% FP attack, the accuracy and ASR were both approximately 0.9600, hence, FP attacks are difficult to detect. We repeated the experiment with different poisoning percentages.



## **31. Gradient Cuff: Detecting Jailbreak Attacks on Large Language Models by Exploring Refusal Loss Landscapes**

cs.CR

Project page:  https://huggingface.co/spaces/TrustSafeAI/GradientCuff-Jailbreak-Defense

**SubmitDate**: 2024-03-05    [abs](http://arxiv.org/abs/2403.00867v2) [paper-pdf](http://arxiv.org/pdf/2403.00867v2)

**Authors**: Xiaomeng Hu, Pin-Yu Chen, Tsung-Yi Ho

**Abstract**: Large Language Models (LLMs) are becoming a prominent generative AI tool, where the user enters a query and the LLM generates an answer. To reduce harm and misuse, efforts have been made to align these LLMs to human values using advanced training techniques such as Reinforcement Learning from Human Feedback (RLHF). However, recent studies have highlighted the vulnerability of LLMs to adversarial jailbreak attempts aiming at subverting the embedded safety guardrails. To address this challenge, this paper defines and investigates the Refusal Loss of LLMs and then proposes a method called Gradient Cuff to detect jailbreak attempts. Gradient Cuff exploits the unique properties observed in the refusal loss landscape, including functional values and its smoothness, to design an effective two-step detection strategy. Experimental results on two aligned LLMs (LLaMA-2-7B-Chat and Vicuna-7B-V1.5) and six types of jailbreak attacks (GCG, AutoDAN, PAIR, TAP, Base64, and LRL) show that Gradient Cuff can significantly improve the LLM's rejection capability for malicious jailbreak queries, while maintaining the model's performance for benign user queries by adjusting the detection threshold.



## **32. XAI-Based Detection of Adversarial Attacks on Deepfake Detectors**

cs.CR

**SubmitDate**: 2024-03-05    [abs](http://arxiv.org/abs/2403.02955v1) [paper-pdf](http://arxiv.org/pdf/2403.02955v1)

**Authors**: Ben Pinhasov, Raz Lapid, Rony Ohayon, Moshe Sipper, Yehudit Aperstein

**Abstract**: We introduce a novel methodology for identifying adversarial attacks on deepfake detectors using eXplainable Artificial Intelligence (XAI). In an era characterized by digital advancement, deepfakes have emerged as a potent tool, creating a demand for efficient detection systems. However, these systems are frequently targeted by adversarial attacks that inhibit their performance. We address this gap, developing a defensible deepfake detector by leveraging the power of XAI. The proposed methodology uses XAI to generate interpretability maps for a given method, providing explicit visualizations of decision-making factors within the AI models. We subsequently employ a pretrained feature extractor that processes both the input image and its corresponding XAI image. The feature embeddings extracted from this process are then used for training a simple yet effective classifier. Our approach contributes not only to the detection of deepfakes but also enhances the understanding of possible adversarial attacks, pinpointing potential vulnerabilities. Furthermore, this approach does not change the performance of the deepfake detector. The paper demonstrates promising results suggesting a potential pathway for future deepfake detection mechanisms. We believe this study will serve as a valuable contribution to the community, sparking much-needed discourse on safeguarding deepfake detectors.



## **33. Precise Extraction of Deep Learning Models via Side-Channel Attacks on Edge/Endpoint Devices**

cs.AI

Accepted by 27th European Symposium on Research in Computer Security  (ESORICS 2022)

**SubmitDate**: 2024-03-05    [abs](http://arxiv.org/abs/2403.02870v1) [paper-pdf](http://arxiv.org/pdf/2403.02870v1)

**Authors**: Younghan Lee, Sohee Jun, Yungi Cho, Woorim Han, Hyungon Moon, Yunheung Paek

**Abstract**: With growing popularity, deep learning (DL) models are becoming larger-scale, and only the companies with vast training datasets and immense computing power can manage their business serving such large models. Most of those DL models are proprietary to the companies who thus strive to keep their private models safe from the model extraction attack (MEA), whose aim is to steal the model by training surrogate models. Nowadays, companies are inclined to offload the models from central servers to edge/endpoint devices. As revealed in the latest studies, adversaries exploit this opportunity as new attack vectors to launch side-channel attack (SCA) on the device running victim model and obtain various pieces of the model information, such as the model architecture (MA) and image dimension (ID). Our work provides a comprehensive understanding of such a relationship for the first time and would benefit future MEA studies in both offensive and defensive sides in that they may learn which pieces of information exposed by SCA are more important than the others. Our analysis additionally reveals that by grasping the victim model information from SCA, MEA can get highly effective and successful even without any prior knowledge of the model. Finally, to evince the practicality of our analysis results, we empirically apply SCA, and subsequently, carry out MEA under realistic threat assumptions. The results show up to 5.8 times better performance than when the adversary has no model information about the victim model.



## **34. FLGuard: Byzantine-Robust Federated Learning via Ensemble of Contrastive Models**

cs.LG

Accepted by 28th European Symposium on Research in Computer Security  (ESORICS 2023)

**SubmitDate**: 2024-03-05    [abs](http://arxiv.org/abs/2403.02846v1) [paper-pdf](http://arxiv.org/pdf/2403.02846v1)

**Authors**: Younghan Lee, Yungi Cho, Woorim Han, Ho Bae, Yunheung Paek

**Abstract**: Federated Learning (FL) thrives in training a global model with numerous clients by only sharing the parameters of their local models trained with their private training datasets. Therefore, without revealing the private dataset, the clients can obtain a deep learning (DL) model with high performance. However, recent research proposed poisoning attacks that cause a catastrophic loss in the accuracy of the global model when adversaries, posed as benign clients, are present in a group of clients. Therefore, recent studies suggested byzantine-robust FL methods that allow the server to train an accurate global model even with the adversaries present in the system. However, many existing methods require the knowledge of the number of malicious clients or the auxiliary (clean) dataset or the effectiveness reportedly decreased hugely when the private dataset was non-independently and identically distributed (non-IID). In this work, we propose FLGuard, a novel byzantine-robust FL method that detects malicious clients and discards malicious local updates by utilizing the contrastive learning technique, which showed a tremendous improvement as a self-supervised learning method. With contrastive models, we design FLGuard as an ensemble scheme to maximize the defensive capability. We evaluate FLGuard extensively under various poisoning attacks and compare the accuracy of the global model with existing byzantine-robust FL methods. FLGuard outperforms the state-of-the-art defense methods in most cases and shows drastic improvement, especially in non-IID settings. https://github.com/201younghanlee/FLGuard



## **35. Here Comes The AI Worm: Unleashing Zero-click Worms that Target GenAI-Powered Applications**

cs.CR

Website: https://sites.google.com/view/compromptmized

**SubmitDate**: 2024-03-05    [abs](http://arxiv.org/abs/2403.02817v1) [paper-pdf](http://arxiv.org/pdf/2403.02817v1)

**Authors**: Stav Cohen, Ron Bitton, Ben Nassi

**Abstract**: In the past year, numerous companies have incorporated Generative AI (GenAI) capabilities into new and existing applications, forming interconnected Generative AI (GenAI) ecosystems consisting of semi/fully autonomous agents powered by GenAI services. While ongoing research highlighted risks associated with the GenAI layer of agents (e.g., dialog poisoning, membership inference, prompt leaking, jailbreaking), a critical question emerges: Can attackers develop malware to exploit the GenAI component of an agent and launch cyber-attacks on the entire GenAI ecosystem? This paper introduces Morris II, the first worm designed to target GenAI ecosystems through the use of adversarial self-replicating prompts. The study demonstrates that attackers can insert such prompts into inputs that, when processed by GenAI models, prompt the model to replicate the input as output (replication), engaging in malicious activities (payload). Additionally, these inputs compel the agent to deliver them (propagate) to new agents by exploiting the connectivity within the GenAI ecosystem. We demonstrate the application of Morris II against GenAIpowered email assistants in two use cases (spamming and exfiltrating personal data), under two settings (black-box and white-box accesses), using two types of input data (text and images). The worm is tested against three different GenAI models (Gemini Pro, ChatGPT 4.0, and LLaVA), and various factors (e.g., propagation rate, replication, malicious activity) influencing the performance of the worm are evaluated.



## **36. Towards Robust Federated Learning via Logits Calibration on Non-IID Data**

cs.CV

Accepted by IEEE NOMS 2024

**SubmitDate**: 2024-03-05    [abs](http://arxiv.org/abs/2403.02803v1) [paper-pdf](http://arxiv.org/pdf/2403.02803v1)

**Authors**: Yu Qiao, Apurba Adhikary, Chaoning Zhang, Choong Seon Hong

**Abstract**: Federated learning (FL) is a privacy-preserving distributed management framework based on collaborative model training of distributed devices in edge networks. However, recent studies have shown that FL is vulnerable to adversarial examples (AEs), leading to a significant drop in its performance. Meanwhile, the non-independent and identically distributed (non-IID) challenge of data distribution between edge devices can further degrade the performance of models. Consequently, both AEs and non-IID pose challenges to deploying robust learning models at the edge. In this work, we adopt the adversarial training (AT) framework to improve the robustness of FL models against adversarial example (AE) attacks, which can be termed as federated adversarial training (FAT). Moreover, we address the non-IID challenge by implementing a simple yet effective logits calibration strategy under the FAT framework, which can enhance the robustness of models when subjected to adversarial attacks. Specifically, we employ a direct strategy to adjust the logits output by assigning higher weights to classes with small samples during training. This approach effectively tackles the class imbalance in the training data, with the goal of mitigating biases between local and global models. Experimental results on three dataset benchmarks, MNIST, Fashion-MNIST, and CIFAR-10 show that our strategy achieves competitive results in natural and robust accuracy compared to several baselines.



## **37. On the Alignment of Group Fairness with Attribute Privacy**

cs.LG

arXiv admin note: text overlap with arXiv:2202.02242

**SubmitDate**: 2024-03-05    [abs](http://arxiv.org/abs/2211.10209v3) [paper-pdf](http://arxiv.org/pdf/2211.10209v3)

**Authors**: Jan Aalmoes, Vasisht Duddu, Antoine Boutet

**Abstract**: Group fairness and privacy are fundamental aspects in designing trustworthy machine learning models. Previous research has highlighted conflicts between group fairness and different privacy notions. We are the first to demonstrate the alignment of group fairness with the specific privacy notion of attribute privacy in a blackbox setting. Attribute privacy, quantified by the resistance to attribute inference attacks (AIAs), requires indistinguishability in the target model's output predictions. Group fairness guarantees this thereby mitigating AIAs and achieving attribute privacy. To demonstrate this, we first introduce AdaptAIA, an enhancement of existing AIAs, tailored for real-world datasets with class imbalances in sensitive attributes. Through theoretical and extensive empirical analyses, we demonstrate the efficacy of two standard group fairness algorithms (i.e., adversarial debiasing and exponentiated gradient descent) against AdaptAIA. Additionally, since using group fairness results in attribute privacy, it acts as a defense against AIAs, which is currently lacking. Overall, we show that group fairness aligns with attribute privacy at no additional cost other than the already existing trade-off with model utility.



## **38. Minimum Topology Attacks for Graph Neural Networks**

cs.AI

Published on WWW 2023. Proceedings of the ACM Web Conference 2023

**SubmitDate**: 2024-03-05    [abs](http://arxiv.org/abs/2403.02723v1) [paper-pdf](http://arxiv.org/pdf/2403.02723v1)

**Authors**: Mengmei Zhang, Xiao Wang, Chuan Shi, Lingjuan Lyu, Tianchi Yang, Junping Du

**Abstract**: With the great popularity of Graph Neural Networks (GNNs), their robustness to adversarial topology attacks has received significant attention. Although many attack methods have been proposed, they mainly focus on fixed-budget attacks, aiming at finding the most adversarial perturbations within a fixed budget for target node. However, considering the varied robustness of each node, there is an inevitable dilemma caused by the fixed budget, i.e., no successful perturbation is found when the budget is relatively small, while if it is too large, the yielding redundant perturbations will hurt the invisibility. To break this dilemma, we propose a new type of topology attack, named minimum-budget topology attack, aiming to adaptively find the minimum perturbation sufficient for a successful attack on each node. To this end, we propose an attack model, named MiBTack, based on a dynamic projected gradient descent algorithm, which can effectively solve the involving non-convex constraint optimization on discrete topology. Extensive results on three GNNs and four real-world datasets show that MiBTack can successfully lead all target nodes misclassified with the minimum perturbation edges. Moreover, the obtained minimum budget can be used to measure node robustness, so we can explore the relationships of robustness, topology, and uncertainty for nodes, which is beyond what the current fixed-budget topology attacks can offer.



## **39. ScAR: Scaling Adversarial Robustness for LiDAR Object Detection**

cs.CV

**SubmitDate**: 2024-03-04    [abs](http://arxiv.org/abs/2312.03085v2) [paper-pdf](http://arxiv.org/pdf/2312.03085v2)

**Authors**: Xiaohu Lu, Hayder Radha

**Abstract**: The adversarial robustness of a model is its ability to resist adversarial attacks in the form of small perturbations to input data. Universal adversarial attack methods such as Fast Sign Gradient Method (FSGM) and Projected Gradient Descend (PGD) are popular for LiDAR object detection, but they are often deficient compared to task-specific adversarial attacks. Additionally, these universal methods typically require unrestricted access to the model's information, which is difficult to obtain in real-world applications. To address these limitations, we present a black-box Scaling Adversarial Robustness (ScAR) method for LiDAR object detection. By analyzing the statistical characteristics of 3D object detection datasets such as KITTI, Waymo, and nuScenes, we have found that the model's prediction is sensitive to scaling of 3D instances. We propose three black-box scaling adversarial attack methods based on the available information: model-aware attack, distribution-aware attack, and blind attack. We also introduce a strategy for generating scaling adversarial examples to improve the model's robustness against these three scaling adversarial attacks. Comparison with other methods on public datasets under different 3D object detection architectures demonstrates the effectiveness of our proposed method. Our code is available at https://github.com/xiaohulugo/ScAR-IROS2023.



## **40. Towards Poisoning Fair Representations**

cs.LG

**SubmitDate**: 2024-03-04    [abs](http://arxiv.org/abs/2309.16487v2) [paper-pdf](http://arxiv.org/pdf/2309.16487v2)

**Authors**: Tianci Liu, Haoyu Wang, Feijie Wu, Hengtong Zhang, Pan Li, Lu Su, Jing Gao

**Abstract**: Fair machine learning seeks to mitigate model prediction bias against certain demographic subgroups such as elder and female. Recently, fair representation learning (FRL) trained by deep neural networks has demonstrated superior performance, whereby representations containing no demographic information are inferred from the data and then used as the input to classification or other downstream tasks. Despite the development of FRL methods, their vulnerability under data poisoning attack, a popular protocol to benchmark model robustness under adversarial scenarios, is under-explored. Data poisoning attacks have been developed for classical fair machine learning methods which incorporate fairness constraints into shallow-model classifiers. Nonetheless, these attacks fall short in FRL due to notably different fairness goals and model architectures. This work proposes the first data poisoning framework attacking FRL. We induce the model to output unfair representations that contain as much demographic information as possible by injecting carefully crafted poisoning samples into the training data. This attack entails a prohibitive bilevel optimization, wherefore an effective approximated solution is proposed. A theoretical analysis on the needed number of poisoning samples is derived and sheds light on defending against the attack. Experiments on benchmark fairness datasets and state-of-the-art fair representation learning models demonstrate the superiority of our attack.



## **41. COMMIT: Certifying Robustness of Multi-Sensor Fusion Systems against Semantic Attacks**

cs.LG

**SubmitDate**: 2024-03-04    [abs](http://arxiv.org/abs/2403.02329v1) [paper-pdf](http://arxiv.org/pdf/2403.02329v1)

**Authors**: Zijian Huang, Wenda Chu, Linyi Li, Chejian Xu, Bo Li

**Abstract**: Multi-sensor fusion systems (MSFs) play a vital role as the perception module in modern autonomous vehicles (AVs). Therefore, ensuring their robustness against common and realistic adversarial semantic transformations, such as rotation and shifting in the physical world, is crucial for the safety of AVs. While empirical evidence suggests that MSFs exhibit improved robustness compared to single-modal models, they are still vulnerable to adversarial semantic transformations. Despite the proposal of empirical defenses, several works show that these defenses can be attacked again by new adaptive attacks. So far, there is no certified defense proposed for MSFs. In this work, we propose the first robustness certification framework COMMIT certify robustness of multi-sensor fusion systems against semantic attacks. In particular, we propose a practical anisotropic noise mechanism that leverages randomized smoothing with multi-modal data and performs a grid-based splitting method to characterize complex semantic transformations. We also propose efficient algorithms to compute the certification in terms of object detection accuracy and IoU for large-scale MSF models. Empirically, we evaluate the efficacy of COMMIT in different settings and provide a comprehensive benchmark of certified robustness for different MSF models using the CARLA simulation platform. We show that the certification for MSF models is at most 48.39% higher than that of single-modal models, which validates the advantages of MSF models. We believe our certification framework and benchmark will contribute an important step towards certifiably robust AVs in practice.



## **42. Mirage: Defense against CrossPath Attacks in Software Defined Networks**

cs.CR

**SubmitDate**: 2024-03-04    [abs](http://arxiv.org/abs/2403.02172v1) [paper-pdf](http://arxiv.org/pdf/2403.02172v1)

**Authors**: Shariq Murtuza, Krishna Asawa

**Abstract**: The Software-Defined Networks (SDNs) face persistent threats from various adversaries that attack them using different methods to mount Denial of Service attacks. These attackers have different motives and follow diverse tactics to achieve their nefarious objectives. In this work, we focus on the impact of CrossPath attacks in SDNs and introduce our framework, Mirage, which not only detects but also mitigates this attack. Our framework, Mirage, detects SDN switches that become unreachable due to being under attack, takes proactive measures to prevent Adversarial Path Reconnaissance, and effectively mitigates CrossPath attacks in SDNs. A CrossPath attack is a form of link flood attack that indirectly attacks the control plane by overwhelming the shared links that connect the data and control planes with data plane traffic. This attack is exclusive to in band SDN, where the data and the control plane, both utilize the same physical links for transmitting and receiving traffic. Our framework, Mirage, prevents attackers from launching adversarial path reconnaissance to identify shared links in a network, thereby thwarting their abuse and preventing this attack. Mirage not only stops adversarial path reconnaissance but also includes features to quickly counter ongoing attacks once detected. Mirage uses path diversity to reroute network packet to prevent timing based measurement. Mirage can also enforce short lived flow table rules to prevent timing attacks. These measures are carefully designed to enhance the security of the SDN environment. Moreover, we share the results of our experiments, which clearly show Mirage's effectiveness in preventing path reconnaissance, detecting CrossPath attacks, and mitigating ongoing threats. Our framework successfully protects the network from these harmful activities, giving valuable insights into SDN security.



## **43. Rethinking Model Ensemble in Transfer-based Adversarial Attacks**

cs.CV

**SubmitDate**: 2024-03-04    [abs](http://arxiv.org/abs/2303.09105v2) [paper-pdf](http://arxiv.org/pdf/2303.09105v2)

**Authors**: Huanran Chen, Yichi Zhang, Yinpeng Dong, Xiao Yang, Hang Su, Jun Zhu

**Abstract**: It is widely recognized that deep learning models lack robustness to adversarial examples. An intriguing property of adversarial examples is that they can transfer across different models, which enables black-box attacks without any knowledge of the victim model. An effective strategy to improve the transferability is attacking an ensemble of models. However, previous works simply average the outputs of different models, lacking an in-depth analysis on how and why model ensemble methods can strongly improve the transferability. In this paper, we rethink the ensemble in adversarial attacks and define the common weakness of model ensemble with two properties: 1) the flatness of loss landscape; and 2) the closeness to the local optimum of each model. We empirically and theoretically show that both properties are strongly correlated with the transferability and propose a Common Weakness Attack (CWA) to generate more transferable adversarial examples by promoting these two properties. Experimental results on both image classification and object detection tasks validate the effectiveness of our approach to improving the adversarial transferability, especially when attacking adversarially trained models. We also successfully apply our method to attack a black-box large vision-language model -- Google's Bard, showing the practical effectiveness. Code is available at \url{https://github.com/huanranchen/AdversarialAttacks}.



## **44. Robustness Bounds on the Successful Adversarial Examples: Theory and Practice**

cs.LG

**SubmitDate**: 2024-03-04    [abs](http://arxiv.org/abs/2403.01896v1) [paper-pdf](http://arxiv.org/pdf/2403.01896v1)

**Authors**: Hiroaki Maeshima, Akira Otsuka

**Abstract**: Adversarial example (AE) is an attack method for machine learning, which is crafted by adding imperceptible perturbation to the data inducing misclassification. In the current paper, we investigated the upper bound of the probability of successful AEs based on the Gaussian Process (GP) classification. We proved a new upper bound that depends on AE's perturbation norm, the kernel function used in GP, and the distance of the closest pair with different labels in the training dataset. Surprisingly, the upper bound is determined regardless of the distribution of the sample dataset. We showed that our theoretical result was confirmed through the experiment using ImageNet. In addition, we showed that changing the parameters of the kernel function induces a change of the upper bound of the probability of successful AEs.



## **45. One Prompt Word is Enough to Boost Adversarial Robustness for Pre-trained Vision-Language Models**

cs.CV

CVPR2024

**SubmitDate**: 2024-03-04    [abs](http://arxiv.org/abs/2403.01849v1) [paper-pdf](http://arxiv.org/pdf/2403.01849v1)

**Authors**: Lin Li, Haoyan Guan, Jianing Qiu, Michael Spratling

**Abstract**: Large pre-trained Vision-Language Models (VLMs) like CLIP, despite having remarkable generalization ability, are highly vulnerable to adversarial examples. This work studies the adversarial robustness of VLMs from the novel perspective of the text prompt instead of the extensively studied model weights (frozen in this work). We first show that the effectiveness of both adversarial attack and defense are sensitive to the used text prompt. Inspired by this, we propose a method to improve resilience to adversarial attacks by learning a robust text prompt for VLMs. The proposed method, named Adversarial Prompt Tuning (APT), is effective while being both computationally and data efficient. Extensive experiments are conducted across 15 datasets and 4 data sparsity schemes (from 1-shot to full training data settings) to show APT's superiority over hand-engineered prompts and other state-of-the-art adaption methods. APT demonstrated excellent abilities in terms of the in-distribution performance and the generalization under input distribution shift and across datasets. Surprisingly, by simply adding one learned word to the prompts, APT can significantly boost the accuracy and robustness (epsilon=4/255) over the hand-engineered prompts by +13% and +8.5% on average respectively. The improvement further increases, in our most effective setting, to +26.4% for accuracy and +16.7% for robustness. Code is available at https://github.com/TreeLLi/APT.



## **46. Coupling Graph Neural Networks with Fractional Order Continuous Dynamics: A Robustness Study**

cs.LG

in Proc. AAAI Conference on Artificial Intelligence, Vancouver,  Canada, Feb. 2024

**SubmitDate**: 2024-03-04    [abs](http://arxiv.org/abs/2401.04331v2) [paper-pdf](http://arxiv.org/pdf/2401.04331v2)

**Authors**: Qiyu Kang, Kai Zhao, Yang Song, Yihang Xie, Yanan Zhao, Sijie Wang, Rui She, Wee Peng Tay

**Abstract**: In this work, we rigorously investigate the robustness of graph neural fractional-order differential equation (FDE) models. This framework extends beyond traditional graph neural (integer-order) ordinary differential equation (ODE) models by implementing the time-fractional Caputo derivative. Utilizing fractional calculus allows our model to consider long-term memory during the feature updating process, diverging from the memoryless Markovian updates seen in traditional graph neural ODE models. The superiority of graph neural FDE models over graph neural ODE models has been established in environments free from attacks or perturbations. While traditional graph neural ODE models have been verified to possess a degree of stability and resilience in the presence of adversarial attacks in existing literature, the robustness of graph neural FDE models, especially under adversarial conditions, remains largely unexplored. This paper undertakes a detailed assessment of the robustness of graph neural FDE models. We establish a theoretical foundation outlining the robustness characteristics of graph neural FDE models, highlighting that they maintain more stringent output perturbation bounds in the face of input and graph topology disturbances, compared to their integer-order counterparts. Our empirical evaluations further confirm the enhanced robustness of graph neural FDE models, highlighting their potential in adversarially robust applications.



## **47. LLMs Can Defend Themselves Against Jailbreaking in a Practical Manner: A Vision Paper**

cs.CR

Fixed the bibliography reference issue in our LLM jailbreak defense  vision paper submitted on 24 Feb 2024

**SubmitDate**: 2024-03-04    [abs](http://arxiv.org/abs/2402.15727v2) [paper-pdf](http://arxiv.org/pdf/2402.15727v2)

**Authors**: Daoyuan Wu, Shuai Wang, Yang Liu, Ning Liu

**Abstract**: Jailbreaking is an emerging adversarial attack that bypasses the safety alignment deployed in off-the-shelf large language models (LLMs). A considerable amount of research exists proposing more effective jailbreak attacks, including the recent Greedy Coordinate Gradient (GCG) attack, jailbreak template-based attacks such as using "Do-Anything-Now" (DAN), and multilingual jailbreak. In contrast, the defensive side has been relatively less explored. This paper proposes a lightweight yet practical defense called SELFDEFEND, which can defend against all existing jailbreak attacks with minimal delay for jailbreak prompts and negligible delay for normal user prompts. Our key insight is that regardless of the kind of jailbreak strategies employed, they eventually need to include a harmful prompt (e.g., "how to make a bomb") in the prompt sent to LLMs, and we found that existing LLMs can effectively recognize such harmful prompts that violate their safety policies. Based on this insight, we design a shadow stack that concurrently checks whether a harmful prompt exists in the user prompt and triggers a checkpoint in the normal stack once a token of "No" or a harmful prompt is output. The latter could also generate an explainable LLM response to adversarial prompts. We demonstrate our idea of SELFDEFEND works in various jailbreak scenarios through manual analysis in GPT-3.5/4. We also list three future directions to further enhance SELFDEFEND.



## **48. Exploring Privacy and Fairness Risks in Sharing Diffusion Models: An Adversarial Perspective**

cs.LG

**SubmitDate**: 2024-03-04    [abs](http://arxiv.org/abs/2402.18607v2) [paper-pdf](http://arxiv.org/pdf/2402.18607v2)

**Authors**: Xinjian Luo, Yangfan Jiang, Fei Wei, Yuncheng Wu, Xiaokui Xiao, Beng Chin Ooi

**Abstract**: Diffusion models have recently gained significant attention in both academia and industry due to their impressive generative performance in terms of both sampling quality and distribution coverage. Accordingly, proposals are made for sharing pre-trained diffusion models across different organizations, as a way of improving data utilization while enhancing privacy protection by avoiding sharing private data directly. However, the potential risks associated with such an approach have not been comprehensively examined.   In this paper, we take an adversarial perspective to investigate the potential privacy and fairness risks associated with the sharing of diffusion models. Specifically, we investigate the circumstances in which one party (the sharer) trains a diffusion model using private data and provides another party (the receiver) black-box access to the pre-trained model for downstream tasks. We demonstrate that the sharer can execute fairness poisoning attacks to undermine the receiver's downstream models by manipulating the training data distribution of the diffusion model. Meanwhile, the receiver can perform property inference attacks to reveal the distribution of sensitive features in the sharer's dataset. Our experiments conducted on real-world datasets demonstrate remarkable attack performance on different types of diffusion models, which highlights the critical importance of robust data auditing and privacy protection protocols in pertinent applications.



## **49. Near Optimal Adversarial Attack on UCB Bandits**

cs.LG

Accepted at AISTATS 2024, a preliminary version appeared at ICML 2023  AdvML Workshop

**SubmitDate**: 2024-03-03    [abs](http://arxiv.org/abs/2008.09312v7) [paper-pdf](http://arxiv.org/pdf/2008.09312v7)

**Authors**: Shiliang Zuo

**Abstract**: I study adversarial attacks against stochastic bandit algorithms. At each round, the learner chooses an arm, and a stochastic reward is generated. The adversary strategically adds corruption to the reward, and the learner is only able to observe the corrupted reward at each round. Two sets of results are presented in this paper. The first set studies the optimal attack strategies for the adversary. The adversary has a target arm he wishes to promote, and his goal is to manipulate the learner into choosing this target arm $T - o(T)$ times. I design attack strategies against UCB and Thompson Sampling that only spends $\widehat{O}(\sqrt{\log T})$ cost. Matching lower bounds are presented, and the vulnerability of UCB, Thompson sampling and $\varepsilon$-greedy are exactly characterized. The second set studies how the learner can defend against the adversary. Inspired by literature on smoothed analysis and behavioral economics, I present two simple algorithms that achieve a competitive ratio arbitrarily close to 1.



## **50. Breaking Down the Defenses: A Comparative Survey of Attacks on Large Language Models**

cs.CR

**SubmitDate**: 2024-03-03    [abs](http://arxiv.org/abs/2403.04786v1) [paper-pdf](http://arxiv.org/pdf/2403.04786v1)

**Authors**: Arijit Ghosh Chowdhury, Md Mofijul Islam, Vaibhav Kumar, Faysal Hossain Shezan, Vaibhav Kumar, Vinija Jain, Aman Chadha

**Abstract**: Large Language Models (LLMs) have become a cornerstone in the field of Natural Language Processing (NLP), offering transformative capabilities in understanding and generating human-like text. However, with their rising prominence, the security and vulnerability aspects of these models have garnered significant attention. This paper presents a comprehensive survey of the various forms of attacks targeting LLMs, discussing the nature and mechanisms of these attacks, their potential impacts, and current defense strategies. We delve into topics such as adversarial attacks that aim to manipulate model outputs, data poisoning that affects model training, and privacy concerns related to training data exploitation. The paper also explores the effectiveness of different attack methodologies, the resilience of LLMs against these attacks, and the implications for model integrity and user trust. By examining the latest research, we provide insights into the current landscape of LLM vulnerabilities and defense mechanisms. Our objective is to offer a nuanced understanding of LLM attacks, foster awareness within the AI community, and inspire robust solutions to mitigate these risks in future developments.



