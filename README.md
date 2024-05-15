# Latest Adversarial Attack Papers
**update at 2024-05-15 11:02:33**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. S3C2 Summit 2024-03: Industry Secure Supply Chain Summit**

cs.CR

This is our WIP paper on the Summit. More versions will be released  soon

**SubmitDate**: 2024-05-14    [abs](http://arxiv.org/abs/2405.08762v1) [paper-pdf](http://arxiv.org/pdf/2405.08762v1)

**Authors**: Greg Tystahl, Yasemin Acar, Michel Cukier, William Enck, Christian Kastner, Alexandros Kapravelos, Dominik Wermke, Laurie Williams

**Abstract**: Supply chain security has become a very important vector to consider when defending against adversary attacks. Due to this, more and more developers are keen on improving their supply chains to make them more robust against future threats. On March 7th, 2024 researchers from the Secure Software Supply Chain Center (S3C2) gathered 14 industry leaders, developers and consumers of the open source ecosystem to discuss the state of supply chain security. The goal of the summit is to share insights between companies and developers alike to foster new collaborations and ideas moving forward. Through this meeting, participants were questions on best practices and thoughts how to improve things for the future. In this paper we summarize the responses and discussions of the summit. The panel questions can be found in the appendix.



## **2. Design and Analysis of Resilient Vehicular Platoon Systems over Wireless Networks**

eess.SY

6 pages, 4 figures, in submission of Globecom 2024

**SubmitDate**: 2024-05-14    [abs](http://arxiv.org/abs/2405.08706v1) [paper-pdf](http://arxiv.org/pdf/2405.08706v1)

**Authors**: Tingyu Shui, Walid Saad

**Abstract**: Connected vehicular platoons provide a promising solution to improve traffic efficiency and ensure road safety. Vehicles in a platoon utilize on-board sensors and wireless vehicle-to-vehicle (V2V) links to share traffic information for cooperative adaptive cruise control. To process real-time control and alert information, there is a need to ensure clock synchronization among the platoon's vehicles. However, adversaries can jeopardize the operation of the platoon by attacking the local clocks of vehicles, leading to clock offsets with the platoon's reference clock. In this paper, a novel framework is proposed for analyzing the resilience of vehicular platoons that are connected using V2V links. In particular, a resilient design based on a diffusion protocol is proposed to re-synchronize the attacked vehicle through wireless V2V links thereby mitigating the impact of variance of the transmission delay during recovery. Then, a novel metric named temporal conditional mean exceedance is defined and analyzed in order to characterize the resilience of the platoon. Subsequently, the conditions pertaining to the V2V links and recovery time needed for a resilient design are derived. Numerical results show that the proposed resilient design is feasible in face of a nine-fold increase in the variance of transmission delay compared to a baseline designed for reliability. Moreover, the proposed approach improves the reliability, defined as the probability of meeting a desired clock offset error requirement, by 45% compared to the baseline.



## **3. Quantum Oblivious LWE Sampling and Insecurity of Standard Model Lattice-Based SNARKs**

cs.CR

**SubmitDate**: 2024-05-14    [abs](http://arxiv.org/abs/2401.03807v2) [paper-pdf](http://arxiv.org/pdf/2401.03807v2)

**Authors**: Thomas Debris-Alazard, Pouria Fallahpour, Damien Stehlé

**Abstract**: The Learning With Errors ($\mathsf{LWE}$) problem asks to find $\mathbf{s}$ from an input of the form $(\mathbf{A}, \mathbf{b} = \mathbf{A}\mathbf{s}+\mathbf{e}) \in (\mathbb{Z}/q\mathbb{Z})^{m \times n} \times (\mathbb{Z}/q\mathbb{Z})^{m}$, for a vector $\mathbf{e}$ that has small-magnitude entries. In this work, we do not focus on solving $\mathsf{LWE}$ but on the task of sampling instances. As these are extremely sparse in their range, it may seem plausible that the only way to proceed is to first create $\mathbf{s}$ and $\mathbf{e}$ and then set $\mathbf{b} = \mathbf{A}\mathbf{s}+\mathbf{e}$. In particular, such an instance sampler knows the solution. This raises the question whether it is possible to obliviously sample $(\mathbf{A}, \mathbf{A}\mathbf{s}+\mathbf{e})$, namely, without knowing the underlying $\mathbf{s}$. A variant of the assumption that oblivious $\mathsf{LWE}$ sampling is hard has been used in a series of works to analyze the security of candidate constructions of Succinct Non interactive Arguments of Knowledge (SNARKs). As the assumption is related to $\mathsf{LWE}$, these SNARKs have been conjectured to be secure in the presence of quantum adversaries.   Our main result is a quantum polynomial-time algorithm that samples well-distributed $\mathsf{LWE}$ instances while provably not knowing the solution, under the assumption that $\mathsf{LWE}$ is hard. Moreover, the approach works for a vast range of $\mathsf{LWE}$ parametrizations, including those used in the above-mentioned SNARKs. This invalidates the assumptions used in their security analyses, although it does not yield attacks against the constructions themselves.



## **4. PLeak: Prompt Leaking Attacks against Large Language Model Applications**

cs.CR

To appear in the Proceedings of The ACM Conference on Computer and  Communications Security (CCS), 2024

**SubmitDate**: 2024-05-14    [abs](http://arxiv.org/abs/2405.06823v2) [paper-pdf](http://arxiv.org/pdf/2405.06823v2)

**Authors**: Bo Hui, Haolin Yuan, Neil Gong, Philippe Burlina, Yinzhi Cao

**Abstract**: Large Language Models (LLMs) enable a new ecosystem with many downstream applications, called LLM applications, with different natural language processing tasks. The functionality and performance of an LLM application highly depend on its system prompt, which instructs the backend LLM on what task to perform. Therefore, an LLM application developer often keeps a system prompt confidential to protect its intellectual property. As a result, a natural attack, called prompt leaking, is to steal the system prompt from an LLM application, which compromises the developer's intellectual property. Existing prompt leaking attacks primarily rely on manually crafted queries, and thus achieve limited effectiveness.   In this paper, we design a novel, closed-box prompt leaking attack framework, called PLeak, to optimize an adversarial query such that when the attacker sends it to a target LLM application, its response reveals its own system prompt. We formulate finding such an adversarial query as an optimization problem and solve it with a gradient-based method approximately. Our key idea is to break down the optimization goal by optimizing adversary queries for system prompts incrementally, i.e., starting from the first few tokens of each system prompt step by step until the entire length of the system prompt.   We evaluate PLeak in both offline settings and for real-world LLM applications, e.g., those on Poe, a popular platform hosting such applications. Our results show that PLeak can effectively leak system prompts and significantly outperforms not only baselines that manually curate queries but also baselines with optimized queries that are modified and adapted from existing jailbreaking attacks. We responsibly reported the issues to Poe and are still waiting for their response. Our implementation is available at this repository: https://github.com/BHui97/PLeak.



## **5. Certifying Robustness of Graph Convolutional Networks for Node Perturbation with Polyhedra Abstract Interpretation**

cs.LG

**SubmitDate**: 2024-05-14    [abs](http://arxiv.org/abs/2405.08645v1) [paper-pdf](http://arxiv.org/pdf/2405.08645v1)

**Authors**: Boqi Chen, Kristóf Marussy, Oszkár Semeráth, Gunter Mussbacher, Dániel Varró

**Abstract**: Graph convolutional neural networks (GCNs) are powerful tools for learning graph-based knowledge representations from training data. However, they are vulnerable to small perturbations in the input graph, which makes them susceptible to input faults or adversarial attacks. This poses a significant problem for GCNs intended to be used in critical applications, which need to provide certifiably robust services even in the presence of adversarial perturbations. We propose an improved GCN robustness certification technique for node classification in the presence of node feature perturbations. We introduce a novel polyhedra-based abstract interpretation approach to tackle specific challenges of graph data and provide tight upper and lower bounds for the robustness of the GCN. Experiments show that our approach simultaneously improves the tightness of robustness bounds as well as the runtime performance of certification. Moreover, our method can be used during training to further improve the robustness of GCNs.



## **6. HookChain: A new perspective for Bypassing EDR Solutions**

cs.CR

46 pages, 22 figures, HookChain, Bypass EDR, Evading EDR, IAT Hook,  Halo's Gate

**SubmitDate**: 2024-05-14    [abs](http://arxiv.org/abs/2404.16856v2) [paper-pdf](http://arxiv.org/pdf/2404.16856v2)

**Authors**: Helvio Carvalho Junior

**Abstract**: In the current digital security ecosystem, where threats evolve rapidly and with complexity, companies developing Endpoint Detection and Response (EDR) solutions are in constant search for innovations that not only keep up but also anticipate emerging attack vectors. In this context, this article introduces the HookChain, a look from another perspective at widely known techniques, which when combined, provide an additional layer of sophisticated evasion against traditional EDR systems. Through a precise combination of IAT Hooking techniques, dynamic SSN resolution, and indirect system calls, HookChain redirects the execution flow of Windows subsystems in a way that remains invisible to the vigilant eyes of EDRs that only act on Ntdll.dll, without requiring changes to the source code of the applications and malwares involved. This work not only challenges current conventions in cybersecurity but also sheds light on a promising path for future protection strategies, leveraging the understanding that continuous evolution is key to the effectiveness of digital security. By developing and exploring the HookChain technique, this study significantly contributes to the body of knowledge in endpoint security, stimulating the development of more robust and adaptive solutions that can effectively address the ever-changing dynamics of digital threats. This work aspires to inspire deep reflection and advancement in the research and development of security technologies that are always several steps ahead of adversaries.   UNDER CONSTRUCTION RESEARCH: This paper is not the final version, as it is currently undergoing final tests against several EDRs. We expect to release the final version by August 2024.



## **7. Secure Aggregation Meets Sparsification in Decentralized Learning**

cs.LG

**SubmitDate**: 2024-05-14    [abs](http://arxiv.org/abs/2405.07708v2) [paper-pdf](http://arxiv.org/pdf/2405.07708v2)

**Authors**: Sayan Biswas, Anne-Marie Kermarrec, Rafael Pires, Rishi Sharma, Milos Vujasinovic

**Abstract**: Decentralized learning (DL) faces increased vulnerability to privacy breaches due to sophisticated attacks on machine learning (ML) models. Secure aggregation is a computationally efficient cryptographic technique that enables multiple parties to compute an aggregate of their private data while keeping their individual inputs concealed from each other and from any central aggregator. To enhance communication efficiency in DL, sparsification techniques are used, selectively sharing only the most crucial parameters or gradients in a model, thereby maintaining efficiency without notably compromising accuracy. However, applying secure aggregation to sparsified models in DL is challenging due to the transmission of disjoint parameter sets by distinct nodes, which can prevent masks from canceling out effectively. This paper introduces CESAR, a novel secure aggregation protocol for DL designed to be compatible with existing sparsification mechanisms. CESAR provably defends against honest-but-curious adversaries and can be formally adapted to counteract collusion between them. We provide a foundational understanding of the interaction between the sparsification carried out by the nodes and the proportion of the parameters shared under CESAR in both colluding and non-colluding environments, offering analytical insight into the working and applicability of the protocol. Experiments on a network with 48 nodes in a 3-regular topology show that with random subsampling, CESAR is always within 0.5% accuracy of decentralized parallel stochastic gradient descent (D-PSGD), while adding only 11% of data overhead. Moreover, it surpasses the accuracy on TopK by up to 0.3% on independent and identically distributed (IID) data.



## **8. UnMarker: A Universal Attack on Defensive Watermarking**

cs.CR

**SubmitDate**: 2024-05-14    [abs](http://arxiv.org/abs/2405.08363v1) [paper-pdf](http://arxiv.org/pdf/2405.08363v1)

**Authors**: Andre Kassis, Urs Hengartner

**Abstract**: Reports regarding the misuse of $\textit{Generative AI}$ ($\textit{GenAI}$) to create harmful deepfakes are emerging daily. Recently, defensive watermarking, which enables $\textit{GenAI}$ providers to hide fingerprints in their images to later use for deepfake detection, has been on the rise. Yet, its potential has not been fully explored. We present $\textit{UnMarker}$ -- the first practical $\textit{universal}$ attack on defensive watermarking. Unlike existing attacks, $\textit{UnMarker}$ requires no detector feedback, no unrealistic knowledge of the scheme or similar models, and no advanced denoising pipelines that may not be available. Instead, being the product of an in-depth analysis of the watermarking paradigm revealing that robust schemes must construct their watermarks in the spectral amplitudes, $\textit{UnMarker}$ employs two novel adversarial optimizations to disrupt the spectra of watermarked images, erasing the watermarks. Evaluations against the $\textit{SOTA}$ prove its effectiveness, not only defeating traditional schemes while retaining superior quality compared to existing attacks but also breaking $\textit{semantic}$ watermarks that alter the image's structure, reducing the best detection rate to $43\%$ and rendering them useless. To our knowledge, $\textit{UnMarker}$ is the first practical attack on $\textit{semantic}$ watermarks, which have been deemed the future of robust watermarking. $\textit{UnMarker}$ casts doubts on the very penitential of this countermeasure and exposes its paradoxical nature as designing schemes for robustness inevitably compromises other robustness aspects.



## **9. SpeechGuard: Exploring the Adversarial Robustness of Multimodal Large Language Models**

cs.CL

9+6 pages, Submitted to ACL 2024

**SubmitDate**: 2024-05-14    [abs](http://arxiv.org/abs/2405.08317v1) [paper-pdf](http://arxiv.org/pdf/2405.08317v1)

**Authors**: Raghuveer Peri, Sai Muralidhar Jayanthi, Srikanth Ronanki, Anshu Bhatia, Karel Mundnich, Saket Dingliwal, Nilaksh Das, Zejiang Hou, Goeric Huybrechts, Srikanth Vishnubhotla, Daniel Garcia-Romero, Sundararajan Srinivasan, Kyu J Han, Katrin Kirchhoff

**Abstract**: Integrated Speech and Large Language Models (SLMs) that can follow speech instructions and generate relevant text responses have gained popularity lately. However, the safety and robustness of these models remains largely unclear. In this work, we investigate the potential vulnerabilities of such instruction-following speech-language models to adversarial attacks and jailbreaking. Specifically, we design algorithms that can generate adversarial examples to jailbreak SLMs in both white-box and black-box attack settings without human involvement. Additionally, we propose countermeasures to thwart such jailbreaking attacks. Our models, trained on dialog data with speech instructions, achieve state-of-the-art performance on spoken question-answering task, scoring over 80% on both safety and helpfulness metrics. Despite safety guardrails, experiments on jailbreaking demonstrate the vulnerability of SLMs to adversarial perturbations and transfer attacks, with average attack success rates of 90% and 10% respectively when evaluated on a dataset of carefully designed harmful questions spanning 12 different toxic categories. However, we demonstrate that our proposed countermeasures reduce the attack success significantly.



## **10. Adversarial Nibbler: An Open Red-Teaming Method for Identifying Diverse Harms in Text-to-Image Generation**

cs.CY

10 pages, 6 figures

**SubmitDate**: 2024-05-14    [abs](http://arxiv.org/abs/2403.12075v3) [paper-pdf](http://arxiv.org/pdf/2403.12075v3)

**Authors**: Jessica Quaye, Alicia Parrish, Oana Inel, Charvi Rastogi, Hannah Rose Kirk, Minsuk Kahng, Erin van Liemt, Max Bartolo, Jess Tsang, Justin White, Nathan Clement, Rafael Mosquera, Juan Ciro, Vijay Janapa Reddi, Lora Aroyo

**Abstract**: With the rise of text-to-image (T2I) generative AI models reaching wide audiences, it is critical to evaluate model robustness against non-obvious attacks to mitigate the generation of offensive images. By focusing on ``implicitly adversarial'' prompts (those that trigger T2I models to generate unsafe images for non-obvious reasons), we isolate a set of difficult safety issues that human creativity is well-suited to uncover. To this end, we built the Adversarial Nibbler Challenge, a red-teaming methodology for crowdsourcing a diverse set of implicitly adversarial prompts. We have assembled a suite of state-of-the-art T2I models, employed a simple user interface to identify and annotate harms, and engaged diverse populations to capture long-tail safety issues that may be overlooked in standard testing. The challenge is run in consecutive rounds to enable a sustained discovery and analysis of safety pitfalls in T2I models.   In this paper, we present an in-depth account of our methodology, a systematic study of novel attack strategies and discussion of safety failures revealed by challenge participants. We also release a companion visualization tool for easy exploration and derivation of insights from the dataset. The first challenge round resulted in over 10k prompt-image pairs with machine annotations for safety. A subset of 1.5k samples contains rich human annotations of harm types and attack styles. We find that 14% of images that humans consider harmful are mislabeled as ``safe'' by machines. We have identified new attack strategies that highlight the complexity of ensuring T2I model robustness. Our findings emphasize the necessity of continual auditing and adaptation as new vulnerabilities emerge. We are confident that this work will enable proactive, iterative safety assessments and promote responsible development of T2I models.



## **11. RAID: A Shared Benchmark for Robust Evaluation of Machine-Generated Text Detectors**

cs.CL

To appear at ACL 2024

**SubmitDate**: 2024-05-13    [abs](http://arxiv.org/abs/2405.07940v1) [paper-pdf](http://arxiv.org/pdf/2405.07940v1)

**Authors**: Liam Dugan, Alyssa Hwang, Filip Trhlik, Josh Magnus Ludan, Andrew Zhu, Hainiu Xu, Daphne Ippolito, Chris Callison-Burch

**Abstract**: Many commercial and open-source models claim to detect machine-generated text with very high accuracy (99\% or higher). However, very few of these detectors are evaluated on shared benchmark datasets and even when they are, the datasets used for evaluation are insufficiently challenging -- lacking variations in sampling strategy, adversarial attacks, and open-source generative models. In this work we present RAID: the largest and most challenging benchmark dataset for machine-generated text detection. RAID includes over 6 million generations spanning 11 models, 8 domains, 11 adversarial attacks and 4 decoding strategies. Using RAID, we evaluate the out-of-domain and adversarial robustness of 8 open- and 4 closed-source detectors and find that current detectors are easily fooled by adversarial attacks, variations in sampling strategies, repetition penalties, and unseen generative models. We release our dataset and tools to encourage further exploration into detector robustness.



## **12. On the Adversarial Robustness of Learning-based Image Compression Against Rate-Distortion Attacks**

eess.IV

**SubmitDate**: 2024-05-13    [abs](http://arxiv.org/abs/2405.07717v1) [paper-pdf](http://arxiv.org/pdf/2405.07717v1)

**Authors**: Chenhao Wu, Qingbo Wu, Haoran Wei, Shuai Chen, Lei Wang, King Ngi Ngan, Fanman Meng, Hongliang Li

**Abstract**: Despite demonstrating superior rate-distortion (RD) performance, learning-based image compression (LIC) algorithms have been found to be vulnerable to malicious perturbations in recent studies. Adversarial samples in these studies are designed to attack only one dimension of either bitrate or distortion, targeting a submodel with a specific compression ratio. However, adversaries in real-world scenarios are neither confined to singular dimensional attacks nor always have control over compression ratios. This variability highlights the inadequacy of existing research in comprehensively assessing the adversarial robustness of LIC algorithms in practical applications. To tackle this issue, this paper presents two joint rate-distortion attack paradigms at both submodel and algorithm levels, i.e., Specific-ratio Rate-Distortion Attack (SRDA) and Agnostic-ratio Rate-Distortion Attack (ARDA). Additionally, a suite of multi-granularity assessment tools is introduced to evaluate the attack results from various perspectives. On this basis, extensive experiments on eight prominent LIC algorithms are conducted to offer a thorough analysis of their inherent vulnerabilities. Furthermore, we explore the efficacy of two defense techniques in improving the performance under joint rate-distortion attacks. The findings from these experiments can provide a valuable reference for the development of compression algorithms with enhanced adversarial robustness.



## **13. DP-DCAN: Differentially Private Deep Contrastive Autoencoder Network for Single-cell Clustering**

cs.LG

**SubmitDate**: 2024-05-13    [abs](http://arxiv.org/abs/2311.03410v2) [paper-pdf](http://arxiv.org/pdf/2311.03410v2)

**Authors**: Huifa Li, Jie Fu, Zhili Chen, Xiaomin Yang, Haitao Liu, Xinpeng Ling

**Abstract**: Single-cell RNA sequencing (scRNA-seq) is important to transcriptomic analysis of gene expression. Recently, deep learning has facilitated the analysis of high-dimensional single-cell data. Unfortunately, deep learning models may leak sensitive information about users. As a result, Differential Privacy (DP) is increasingly used to protect privacy. However, existing DP methods usually perturb whole neural networks to achieve differential privacy, and hence result in great performance overheads. To address this challenge, in this paper, we take advantage of the uniqueness of the autoencoder that it outputs only the dimension-reduced vector in the middle of the network, and design a Differentially Private Deep Contrastive Autoencoder Network (DP-DCAN) by partial network perturbation for single-cell clustering. Since only partial network is added with noise, the performance improvement is obvious and twofold: one part of network is trained with less noise due to a bigger privacy budget, and the other part is trained without any noise. Experimental results of six datasets have verified that DP-DCAN is superior to the traditional DP scheme with whole network perturbation. Moreover, DP-DCAN demonstrates strong robustness to adversarial attacks.



## **14. CrossCert: A Cross-Checking Detection Approach to Patch Robustness Certification for Deep Learning Models**

cs.SE

23 pages, 2 figures, accepted by FSE 2024 (The ACM International  Conference on the Foundations of Software Engineering)

**SubmitDate**: 2024-05-13    [abs](http://arxiv.org/abs/2405.07668v1) [paper-pdf](http://arxiv.org/pdf/2405.07668v1)

**Authors**: Qilin Zhou, Zhengyuan Wei, Haipeng Wang, Bo Jiang, W. K. Chan

**Abstract**: Patch robustness certification is an emerging kind of defense technique against adversarial patch attacks with provable guarantees. There are two research lines: certified recovery and certified detection. They aim to label malicious samples with provable guarantees correctly and issue warnings for malicious samples predicted to non-benign labels with provable guarantees, respectively. However, existing certified detection defenders suffer from protecting labels subject to manipulation, and existing certified recovery defenders cannot systematically warn samples about their labels. A certified defense that simultaneously offers robust labels and systematic warning protection against patch attacks is desirable. This paper proposes a novel certified defense technique called CrossCert. CrossCert formulates a novel approach by cross-checking two certified recovery defenders to provide unwavering certification and detection certification. Unwavering certification ensures that a certified sample, when subjected to a patched perturbation, will always be returned with a benign label without triggering any warnings with a provable guarantee. To our knowledge, CrossCert is the first certified detection technique to offer this guarantee. Our experiments show that, with a slightly lower performance than ViP and comparable performance with PatchCensor in terms of detection certification, CrossCert certifies a significant proportion of samples with the guarantee of unwavering certification.



## **15. Backdoor Removal for Generative Large Language Models**

cs.CR

**SubmitDate**: 2024-05-13    [abs](http://arxiv.org/abs/2405.07667v1) [paper-pdf](http://arxiv.org/pdf/2405.07667v1)

**Authors**: Haoran Li, Yulin Chen, Zihao Zheng, Qi Hu, Chunkit Chan, Heshan Liu, Yangqiu Song

**Abstract**: With rapid advances, generative large language models (LLMs) dominate various Natural Language Processing (NLP) tasks from understanding to reasoning. Yet, language models' inherent vulnerabilities may be exacerbated due to increased accessibility and unrestricted model training on massive textual data from the Internet. A malicious adversary may publish poisoned data online and conduct backdoor attacks on the victim LLMs pre-trained on the poisoned data. Backdoored LLMs behave innocuously for normal queries and generate harmful responses when the backdoor trigger is activated. Despite significant efforts paid to LLMs' safety issues, LLMs are still struggling against backdoor attacks. As Anthropic recently revealed, existing safety training strategies, including supervised fine-tuning (SFT) and Reinforcement Learning from Human Feedback (RLHF), fail to revoke the backdoors once the LLM is backdoored during the pre-training stage. In this paper, we present Simulate and Eliminate (SANDE) to erase the undesired backdoored mappings for generative LLMs. We initially propose Overwrite Supervised Fine-tuning (OSFT) for effective backdoor removal when the trigger is known. Then, to handle the scenarios where the trigger patterns are unknown, we integrate OSFT into our two-stage framework, SANDE. Unlike previous works that center on the identification of backdoors, our safety-enhanced LLMs are able to behave normally even when the exact triggers are activated. We conduct comprehensive experiments to show that our proposed SANDE is effective against backdoor attacks while bringing minimal harm to LLMs' powerful capability without any additional access to unbackdoored clean models. We will release the reproducible code.



## **16. Environmental Matching Attack Against Unmanned Aerial Vehicles Object Detection**

cs.CV

**SubmitDate**: 2024-05-13    [abs](http://arxiv.org/abs/2405.07595v1) [paper-pdf](http://arxiv.org/pdf/2405.07595v1)

**Authors**: Dehong Kong, Siyuan Liang, Wenqi Ren

**Abstract**: Object detection techniques for Unmanned Aerial Vehicles (UAVs) rely on Deep Neural Networks (DNNs), which are vulnerable to adversarial attacks. Nonetheless, adversarial patches generated by existing algorithms in the UAV domain pay very little attention to the naturalness of adversarial patches. Moreover, imposing constraints directly on adversarial patches makes it difficult to generate patches that appear natural to the human eye while ensuring a high attack success rate. We notice that patches are natural looking when their overall color is consistent with the environment. Therefore, we propose a new method named Environmental Matching Attack(EMA) to address the issue of optimizing the adversarial patch under the constraints of color. To the best of our knowledge, this paper is the first to consider natural patches in the domain of UAVs. The EMA method exploits strong prior knowledge of a pretrained stable diffusion to guide the optimization direction of the adversarial patch, where the text guidance can restrict the color of the patch. To better match the environment, the contrast and brightness of the patch are appropriately adjusted. Instead of optimizing the adversarial patch itself, we optimize an adversarial perturbation patch which initializes to zero so that the model can better trade off attacking performance and naturalness. Experiments conducted on the DroneVehicle and Carpk datasets have shown that our work can reach nearly the same attack performance in the digital attack(no greater than 2 in mAP$\%$), surpass the baseline method in the physical specific scenarios, and exhibit a significant advantage in terms of naturalness in visualization and color difference with the environment.



## **17. Towards Rational Consensus in Honest Majority**

cs.GT

**SubmitDate**: 2024-05-13    [abs](http://arxiv.org/abs/2405.07557v1) [paper-pdf](http://arxiv.org/pdf/2405.07557v1)

**Authors**: Varul Srivastava, Sujit Gujar

**Abstract**: Distributed consensus protocols reach agreement among $n$ players in the presence of $f$ adversaries; different protocols support different values of $f$. Existing works study this problem for different adversary types (captured by threat models). There are three primary threat models: (i) Crash fault tolerance (CFT), (ii) Byzantine fault tolerance (BFT), and (iii) Rational fault tolerance (RFT), each more general than the previous. Agreement in repeated rounds on both (1) the proposed value in each round and (2) the ordering among agreed-upon values across multiple rounds is called Atomic BroadCast (ABC). ABC is more generalized than consensus and is employed in blockchains.   This work studies ABC under the RFT threat model. We consider $t$ byzantine and $k$ rational adversaries among $n$ players. We also study different types of rational players based on their utility towards (1) liveness attack, (2) censorship or (3) disagreement (forking attack). We study the problem of ABC under this general threat model in partially-synchronous networks. We show (1) ABC is impossible for $n/3< (t+k) <n/2$ if rational players prefer liveness or censorship attacks and (2) the consensus protocol proposed by Ranchal-Pedrosa and Gramoli cannot be generalized to solve ABC due to insecure Nash equilibrium (resulting in disagreement). For ABC in partially synchronous network settings, we propose a novel protocol \textsf{pRFT}(practical Rational Fault Tolerance). We show \textsf{pRFT} achieves ABC if (a) rational players prefer only disagreement attacks and (b) $t < \frac{n}{4}$ and $(t + k) < \frac{n}{2}$. In \textsf{pRFT}, we incorporate accountability (capturing deviating players) within the protocol by leveraging honest players. We also show that the message complexity of \textsf{pRFT} is at par with the best consensus protocols that guarantee accountability.



## **18. On Securing Analog Lagrange Coded Computing from Colluding Adversaries**

cs.IT

To appear in the proceedings of IEEE ISIT 2024

**SubmitDate**: 2024-05-13    [abs](http://arxiv.org/abs/2405.07454v1) [paper-pdf](http://arxiv.org/pdf/2405.07454v1)

**Authors**: Rimpi Borah, J. Harshan

**Abstract**: Analog Lagrange Coded Computing (ALCC) is a recently proposed coded computing paradigm wherein certain computations over analog datasets can be efficiently performed using distributed worker nodes through floating point implementation. While ALCC is known to preserve privacy of data from the workers, it is not resilient to adversarial workers that return erroneous computation results. Pointing at this security vulnerability, we focus on securing ALCC from a wide range of non-colluding and colluding adversarial workers. As a foundational step, we make use of error-correction algorithms for Discrete Fourier Transform (DFT) codes to build novel algorithms to nullify the erroneous computations returned from the adversaries. Furthermore, when such a robust ALCC is implemented in practical settings, we show that the presence of precision errors in the system can be exploited by the adversaries to propose novel colluding attacks to degrade the computation accuracy. As the main takeaway, we prove a counter-intuitive result that not all the adversaries should inject noise in their computations in order to optimally degrade the accuracy of the ALCC framework. This is the first work of its kind to address the vulnerability of ALCC against colluding adversaries.



## **19. Universal Coding for Shannon Ciphers under Side-Channel Attacks**

cs.IT

6 pages, 3 figures. arXiv admin note: substantial text overlap with  arXiv:1801.02563, arXiv:2201.11670, arXiv:1901.05940

**SubmitDate**: 2024-05-13    [abs](http://arxiv.org/abs/2302.01314v3) [paper-pdf](http://arxiv.org/pdf/2302.01314v3)

**Authors**: Yasutada Oohama, Bagus Santoso

**Abstract**: We study the universal coding under side-channel attacks posed and investigated by Oohama and Santoso (2022). They proposed a theoretical security model for Shannon cipher system under side-channel attacks, where the adversary is not only allowed to collect ciphertexts by eavesdropping the public communication channel, but is also allowed to collect the physical information leaked by the devices where the cipher system is implemented on such as running time, power consumption, electromagnetic radiation, etc. For any distributions of the plain text, any noisy channels through which the adversary observe the corrupted version of the key, and any measurement device used for collecting the physical information, we can derive an achievable rate region for reliability and security such that if we compress the ciphertext with rate within the achievable rate region, then: (1) anyone with secret key will be able to decrypt and decode the ciphertext correctly, but (2) any adversary who obtains the ciphertext and also the side physical information will not be able to obtain any information about the hidden source as long as the leaked physical information is encoded with a rate within the rate region.



## **20. The Janus Interface: How Fine-Tuning in Large Language Models Amplifies the Privacy Risks**

cs.CR

**SubmitDate**: 2024-05-12    [abs](http://arxiv.org/abs/2310.15469v2) [paper-pdf](http://arxiv.org/pdf/2310.15469v2)

**Authors**: Xiaoyi Chen, Siyuan Tang, Rui Zhu, Shijun Yan, Lei Jin, Zihao Wang, Liya Su, Zhikun Zhang, XiaoFeng Wang, Haixu Tang

**Abstract**: The rapid advancements of large language models (LLMs) have raised public concerns about the privacy leakage of personally identifiable information (PII) within their extensive training datasets. Recent studies have demonstrated that an adversary could extract highly sensitive privacy data from the training data of LLMs with carefully designed prompts. However, these attacks suffer from the model's tendency to hallucinate and catastrophic forgetting (CF) in the pre-training stage, rendering the veracity of divulged PIIs negligible. In our research, we propose a novel attack, Janus, which exploits the fine-tuning interface to recover forgotten PIIs from the pre-training data in LLMs. We formalize the privacy leakage problem in LLMs and explain why forgotten PIIs can be recovered through empirical analysis on open-source language models. Based upon these insights, we evaluate the performance of Janus on both open-source language models and two latest LLMs, i.e., GPT-3.5-Turbo and LLaMA-2-7b. Our experiment results show that Janus amplifies the privacy risks by over 10 times in comparison with the baseline and significantly outperforms the state-of-the-art privacy extraction attacks including prefix attacks and in-context learning (ICL). Furthermore, our analysis validates that existing fine-tuning APIs provided by OpenAI and Azure AI Studio are susceptible to our Janus attack, allowing an adversary to conduct such an attack at a low cost.



## **21. Synthesizing Iris Images using Generative Adversarial Networks: Survey and Comparative Analysis**

cs.CV

**SubmitDate**: 2024-05-11    [abs](http://arxiv.org/abs/2404.17105v2) [paper-pdf](http://arxiv.org/pdf/2404.17105v2)

**Authors**: Shivangi Yadav, Arun Ross

**Abstract**: Biometric systems based on iris recognition are currently being used in border control applications and mobile devices. However, research in iris recognition is stymied by various factors such as limited datasets of bonafide irides and presentation attack instruments; restricted intra-class variations; and privacy concerns. Some of these issues can be mitigated by the use of synthetic iris data. In this paper, we present a comprehensive review of state-of-the-art GAN-based synthetic iris image generation techniques, evaluating their strengths and limitations in producing realistic and useful iris images that can be used for both training and testing iris recognition systems and presentation attack detectors. In this regard, we first survey the various methods that have been used for synthetic iris generation and specifically consider generators based on StyleGAN, RaSGAN, CIT-GAN, iWarpGAN, StarGAN, etc. We then analyze the images generated by these models for realism, uniqueness, and biometric utility. This comprehensive analysis highlights the pros and cons of various GANs in the context of developing robust iris matchers and presentation attack detectors.



## **22. Tree Proof-of-Position Algorithms**

cs.DS

**SubmitDate**: 2024-05-10    [abs](http://arxiv.org/abs/2405.06761v1) [paper-pdf](http://arxiv.org/pdf/2405.06761v1)

**Authors**: Aida Manzano Kharman, Pietro Ferraro, Homayoun Hamedmoghadam, Robert Shorten

**Abstract**: We present a novel class of proof-of-position algorithms: Tree-Proof-of-Position (T-PoP). This algorithm is decentralised, collaborative and can be computed in a privacy preserving manner, such that agents do not need to reveal their position publicly. We make no assumptions of honest behaviour in the system, and consider varying ways in which agents may misbehave. Our algorithm is therefore resilient to highly adversarial scenarios. This makes it suitable for a wide class of applications, namely those in which trust in a centralised infrastructure may not be assumed, or high security risk scenarios. Our algorithm has a worst case quadratic runtime, making it suitable for hardware constrained IoT applications. We also provide a mathematical model that summarises T-PoP's performance for varying operating conditions. We then simulate T-PoP's behaviour with a large number of agent-based simulations, which are in complete agreement with our mathematical model, thus demonstrating its validity. T-PoP can achieve high levels of reliability and security by tuning its operating conditions, both in high and low density environments. Finally, we also present a mathematical model to probabilistically detect platooning attacks.



## **23. Certified $\ell_2$ Attribution Robustness via Uniformly Smoothed Attributions**

cs.LG

**SubmitDate**: 2024-05-10    [abs](http://arxiv.org/abs/2405.06361v1) [paper-pdf](http://arxiv.org/pdf/2405.06361v1)

**Authors**: Fan Wang, Adams Wai-Kin Kong

**Abstract**: Model attribution is a popular tool to explain the rationales behind model predictions. However, recent work suggests that the attributions are vulnerable to minute perturbations, which can be added to input samples to fool the attributions while maintaining the prediction outputs. Although empirical studies have shown positive performance via adversarial training, an effective certified defense method is eminently needed to understand the robustness of attributions. In this work, we propose to use uniform smoothing technique that augments the vanilla attributions by noises uniformly sampled from a certain space. It is proved that, for all perturbations within the attack region, the cosine similarity between uniformly smoothed attribution of perturbed sample and the unperturbed sample is guaranteed to be lower bounded. We also derive alternative formulations of the certification that is equivalent to the original one and provides the maximum size of perturbation or the minimum smoothing radius such that the attribution can not be perturbed. We evaluate the proposed method on three datasets and show that the proposed method can effectively protect the attributions from attacks, regardless of the architecture of networks, training schemes and the size of the datasets.



## **24. Evaluating Adversarial Robustness in the Spatial Frequency Domain**

cs.CV

14 pages

**SubmitDate**: 2024-05-10    [abs](http://arxiv.org/abs/2405.06345v1) [paper-pdf](http://arxiv.org/pdf/2405.06345v1)

**Authors**: Keng-Hsin Liao, Chin-Yuan Yeh, Hsi-Wen Chen, Ming-Syan Chen

**Abstract**: Convolutional Neural Networks (CNNs) have dominated the majority of computer vision tasks. However, CNNs' vulnerability to adversarial attacks has raised concerns about deploying these models to safety-critical applications. In contrast, the Human Visual System (HVS), which utilizes spatial frequency channels to process visual signals, is immune to adversarial attacks. As such, this paper presents an empirical study exploring the vulnerability of CNN models in the frequency domain. Specifically, we utilize the discrete cosine transform (DCT) to construct the Spatial-Frequency (SF) layer to produce a block-wise frequency spectrum of an input image and formulate Spatial Frequency CNNs (SF-CNNs) by replacing the initial feature extraction layers of widely-used CNN backbones with the SF layer. Through extensive experiments, we observe that SF-CNN models are more robust than their CNN counterparts under both white-box and black-box attacks. To further explain the robustness of SF-CNNs, we compare the SF layer with a trainable convolutional layer with identical kernel sizes using two mixing strategies to show that the lower frequency components contribute the most to the adversarial robustness of SF-CNNs. We believe our observations can guide the future design of robust CNN models.



## **25. Improving Transferable Targeted Adversarial Attack via Normalized Logit Calibration and Truncated Feature Mixing**

cs.CV

**SubmitDate**: 2024-05-10    [abs](http://arxiv.org/abs/2405.06340v1) [paper-pdf](http://arxiv.org/pdf/2405.06340v1)

**Authors**: Juanjuan Weng, Zhiming Luo, Shaozi Li

**Abstract**: This paper aims to enhance the transferability of adversarial samples in targeted attacks, where attack success rates remain comparatively low. To achieve this objective, we propose two distinct techniques for improving the targeted transferability from the loss and feature aspects. First, in previous approaches, logit calibrations used in targeted attacks primarily focus on the logit margin between the targeted class and the untargeted classes among samples, neglecting the standard deviation of the logit. In contrast, we introduce a new normalized logit calibration method that jointly considers the logit margin and the standard deviation of logits. This approach effectively calibrates the logits, enhancing the targeted transferability. Second, previous studies have demonstrated that mixing the features of clean samples during optimization can significantly increase transferability. Building upon this, we further investigate a truncated feature mixing method to reduce the impact of the source training model, resulting in additional improvements. The truncated feature is determined by removing the Rank-1 feature associated with the largest singular value decomposed from the high-level convolutional layers of the clean sample. Extensive experiments conducted on the ImageNet-Compatible and CIFAR-10 datasets demonstrate the individual and mutual benefits of our proposed two components, which outperform the state-of-the-art methods by a large margin in black-box targeted attacks.



## **26. PUMA: margin-based data pruning**

cs.LG

**SubmitDate**: 2024-05-10    [abs](http://arxiv.org/abs/2405.06298v1) [paper-pdf](http://arxiv.org/pdf/2405.06298v1)

**Authors**: Javier Maroto, Pascal Frossard

**Abstract**: Deep learning has been able to outperform humans in terms of classification accuracy in many tasks. However, to achieve robustness to adversarial perturbations, the best methodologies require to perform adversarial training on a much larger training set that has been typically augmented using generative models (e.g., diffusion models). Our main objective in this work, is to reduce these data requirements while achieving the same or better accuracy-robustness trade-offs. We focus on data pruning, where some training samples are removed based on the distance to the model classification boundary (i.e., margin). We find that the existing approaches that prune samples with low margin fails to increase robustness when we add a lot of synthetic data, and explain this situation with a perceptron learning task. Moreover, we find that pruning high margin samples for better accuracy increases the harmful impact of mislabeled perturbed data in adversarial training, hurting both robustness and accuracy. We thus propose PUMA, a new data pruning strategy that computes the margin using DeepFool, and prunes the training samples of highest margin without hurting performance by jointly adjusting the training attack norm on the samples of lowest margin. We show that PUMA can be used on top of the current state-of-the-art methodology in robustness, and it is able to significantly improve the model performance unlike the existing data pruning strategies. Not only PUMA achieves similar robustness with less data, but it also significantly increases the model accuracy, improving the performance trade-off.



## **27. Exploring the Interplay of Interpretability and Robustness in Deep Neural Networks: A Saliency-guided Approach**

cs.CV

**SubmitDate**: 2024-05-10    [abs](http://arxiv.org/abs/2405.06278v1) [paper-pdf](http://arxiv.org/pdf/2405.06278v1)

**Authors**: Amira Guesmi, Nishant Suresh Aswani, Muhammad Shafique

**Abstract**: Adversarial attacks pose a significant challenge to deploying deep learning models in safety-critical applications. Maintaining model robustness while ensuring interpretability is vital for fostering trust and comprehension in these models. This study investigates the impact of Saliency-guided Training (SGT) on model robustness, a technique aimed at improving the clarity of saliency maps to deepen understanding of the model's decision-making process. Experiments were conducted on standard benchmark datasets using various deep learning architectures trained with and without SGT. Findings demonstrate that SGT enhances both model robustness and interpretability. Additionally, we propose a novel approach combining SGT with standard adversarial training to achieve even greater robustness while preserving saliency map quality. Our strategy is grounded in the assumption that preserving salient features crucial for correctly classifying adversarial examples enhances model robustness, while masking non-relevant features improves interpretability. Our technique yields significant gains, achieving a 35\% and 20\% improvement in robustness against PGD attack with noise magnitudes of $0.2$ and $0.02$ for the MNIST and CIFAR-10 datasets, respectively, while producing high-quality saliency maps.



## **28. Disttack: Graph Adversarial Attacks Toward Distributed GNN Training**

cs.LG

Accepted by 30th International European Conference on Parallel and  Distributed Computing(Euro-Par 2024)

**SubmitDate**: 2024-05-10    [abs](http://arxiv.org/abs/2405.06247v1) [paper-pdf](http://arxiv.org/pdf/2405.06247v1)

**Authors**: Yuxiang Zhang, Xin Liu, Meng Wu, Wei Yan, Mingyu Yan, Xiaochun Ye, Dongrui Fan

**Abstract**: Graph Neural Networks (GNNs) have emerged as potent models for graph learning. Distributing the training process across multiple computing nodes is the most promising solution to address the challenges of ever-growing real-world graphs. However, current adversarial attack methods on GNNs neglect the characteristics and applications of the distributed scenario, leading to suboptimal performance and inefficiency in attacking distributed GNN training.   In this study, we introduce Disttack, the first framework of adversarial attacks for distributed GNN training that leverages the characteristics of frequent gradient updates in a distributed system. Specifically, Disttack corrupts distributed GNN training by injecting adversarial attacks into one single computing node. The attacked subgraphs are precisely perturbed to induce an abnormal gradient ascent in backpropagation, disrupting gradient synchronization between computing nodes and thus leading to a significant performance decline of the trained GNN. We evaluate Disttack on four large real-world graphs by attacking five widely adopted GNNs. Compared with the state-of-the-art attack method, experimental results demonstrate that Disttack amplifies the model accuracy degradation by 2.75$\times$ and achieves speedup by 17.33$\times$ on average while maintaining unnoticeability.



## **29. Concealing Backdoor Model Updates in Federated Learning by Trigger-Optimized Data Poisoning**

cs.CR

**SubmitDate**: 2024-05-10    [abs](http://arxiv.org/abs/2405.06206v1) [paper-pdf](http://arxiv.org/pdf/2405.06206v1)

**Authors**: Yujie Zhang, Neil Gong, Michael K. Reiter

**Abstract**: Federated Learning (FL) is a decentralized machine learning method that enables participants to collaboratively train a model without sharing their private data. Despite its privacy and scalability benefits, FL is susceptible to backdoor attacks, where adversaries poison the local training data of a subset of clients using a backdoor trigger, aiming to make the aggregated model produce malicious results when the same backdoor condition is met by an inference-time input. Existing backdoor attacks in FL suffer from common deficiencies: fixed trigger patterns and reliance on the assistance of model poisoning. State-of-the-art defenses based on Byzantine-robust aggregation exhibit a good defense performance on these attacks because of the significant divergence between malicious and benign model updates. To effectively conceal malicious model updates among benign ones, we propose DPOT, a backdoor attack strategy in FL that dynamically constructs backdoor objectives by optimizing a backdoor trigger, making backdoor data have minimal effect on model updates. We provide theoretical justifications for DPOT's attacking principle and display experimental results showing that DPOT, via only a data-poisoning attack, effectively undermines state-of-the-art defenses and outperforms existing backdoor attack techniques on various datasets.



## **30. Muting Whisper: A Universal Acoustic Adversarial Attack on Speech Foundation Models**

cs.CL

**SubmitDate**: 2024-05-09    [abs](http://arxiv.org/abs/2405.06134v1) [paper-pdf](http://arxiv.org/pdf/2405.06134v1)

**Authors**: Vyas Raina, Rao Ma, Charles McGhee, Kate Knill, Mark Gales

**Abstract**: Recent developments in large speech foundation models like Whisper have led to their widespread use in many automatic speech recognition (ASR) applications. These systems incorporate `special tokens' in their vocabulary, such as $\texttt{<endoftext>}$, to guide their language generation process. However, we demonstrate that these tokens can be exploited by adversarial attacks to manipulate the model's behavior. We propose a simple yet effective method to learn a universal acoustic realization of Whisper's $\texttt{<endoftext>}$ token, which, when prepended to any speech signal, encourages the model to ignore the speech and only transcribe the special token, effectively `muting' the model. Our experiments demonstrate that the same, universal 0.64-second adversarial audio segment can successfully mute a target Whisper ASR model for over 97\% of speech samples. Moreover, we find that this universal adversarial audio segment often transfers to new datasets and tasks. Overall this work demonstrates the vulnerability of Whisper models to `muting' adversarial attacks, where such attacks can pose both risks and potential benefits in real-world settings: for example the attack can be used to bypass speech moderation systems, or conversely the attack can also be used to protect private speech data.



## **31. Hard Work Does Not Always Pay Off: Poisoning Attacks on Neural Architecture Search**

cs.LG

**SubmitDate**: 2024-05-09    [abs](http://arxiv.org/abs/2405.06073v1) [paper-pdf](http://arxiv.org/pdf/2405.06073v1)

**Authors**: Zachary Coalson, Huazheng Wang, Qingyun Wu, Sanghyun Hong

**Abstract**: In this paper, we study the robustness of "data-centric" approaches to finding neural network architectures (known as neural architecture search) to data distribution shifts. To audit this robustness, we present a data poisoning attack, when injected to the training data used for architecture search that can prevent the victim algorithm from finding an architecture with optimal accuracy. We first define the attack objective for crafting poisoning samples that can induce the victim to generate sub-optimal architectures. To this end, we weaponize existing search algorithms to generate adversarial architectures that serve as our objectives. We also present techniques that the attacker can use to significantly reduce the computational costs of crafting poisoning samples. In an extensive evaluation of our poisoning attack on a representative architecture search algorithm, we show its surprising robustness. Because our attack employs clean-label poisoning, we also evaluate its robustness against label noise. We find that random label-flipping is more effective in generating sub-optimal architectures than our clean-label attack. Our results suggests that care must be taken for the data this emerging approach uses, and future work is needed to develop robust algorithms.



## **32. BB-Patch: BlackBox Adversarial Patch-Attack using Zeroth-Order Optimization**

cs.CV

**SubmitDate**: 2024-05-09    [abs](http://arxiv.org/abs/2405.06049v1) [paper-pdf](http://arxiv.org/pdf/2405.06049v1)

**Authors**: Satyadwyoom Kumar, Saurabh Gupta, Arun Balaji Buduru

**Abstract**: Deep Learning has become popular due to its vast applications in almost all domains. However, models trained using deep learning are prone to failure for adversarial samples and carry a considerable risk in sensitive applications. Most of these adversarial attack strategies assume that the adversary has access to the training data, the model parameters, and the input during deployment, hence, focus on perturbing the pixel level information present in the input image.   Adversarial Patches were introduced to the community which helped in bringing out the vulnerability of deep learning models in a much more pragmatic manner but here the attacker has a white-box access to the model parameters. Recently, there has been an attempt to develop these adversarial attacks using black-box techniques. However, certain assumptions such as availability large training data is not valid for a real-life scenarios. In a real-life scenario, the attacker can only assume the type of model architecture used from a select list of state-of-the-art architectures while having access to only a subset of input dataset. Hence, we propose an black-box adversarial attack strategy that produces adversarial patches which can be applied anywhere in the input image to perform an adversarial attack.



## **33. Trustworthy AI-Generative Content in Intelligent 6G Network: Adversarial, Privacy, and Fairness**

cs.CR

**SubmitDate**: 2024-05-09    [abs](http://arxiv.org/abs/2405.05930v1) [paper-pdf](http://arxiv.org/pdf/2405.05930v1)

**Authors**: Siyuan Li, Xi Lin, Yaju Liu, Jianhua Li

**Abstract**: AI-generated content (AIGC) models, represented by large language models (LLM), have brought revolutionary changes to the content generation fields. The high-speed and extensive 6G technology is an ideal platform for providing powerful AIGC mobile service applications, while future 6G mobile networks also need to support intelligent and personalized mobile generation services. However, the significant ethical and security issues of current AIGC models, such as adversarial attacks, privacy, and fairness, greatly affect the credibility of 6G intelligent networks, especially in ensuring secure, private, and fair AIGC applications. In this paper, we propose TrustGAIN, a novel paradigm for trustworthy AIGC in 6G networks, to ensure trustworthy large-scale AIGC services in future 6G networks. We first discuss the adversarial attacks and privacy threats faced by AIGC systems in 6G networks, as well as the corresponding protection issues. Subsequently, we emphasize the importance of ensuring the unbiasedness and fairness of the mobile generative service in future intelligent networks. In particular, we conduct a use case to demonstrate that TrustGAIN can effectively guide the resistance against malicious or generated false information. We believe that TrustGAIN is a necessary paradigm for intelligent and trustworthy 6G networks to support AIGC services, ensuring the security, privacy, and fairness of AIGC network services.



## **34. A Linear Reconstruction Approach for Attribute Inference Attacks against Synthetic Data**

cs.LG

**SubmitDate**: 2024-05-09    [abs](http://arxiv.org/abs/2301.10053v3) [paper-pdf](http://arxiv.org/pdf/2301.10053v3)

**Authors**: Meenatchi Sundaram Muthu Selva Annamalai, Andrea Gadotti, Luc Rocher

**Abstract**: Recent advances in synthetic data generation (SDG) have been hailed as a solution to the difficult problem of sharing sensitive data while protecting privacy. SDG aims to learn statistical properties of real data in order to generate "artificial" data that are structurally and statistically similar to sensitive data. However, prior research suggests that inference attacks on synthetic data can undermine privacy, but only for specific outlier records. In this work, we introduce a new attribute inference attack against synthetic data. The attack is based on linear reconstruction methods for aggregate statistics, which target all records in the dataset, not only outliers. We evaluate our attack on state-of-the-art SDG algorithms, including Probabilistic Graphical Models, Generative Adversarial Networks, and recent differentially private SDG mechanisms. By defining a formal privacy game, we show that our attack can be highly accurate even on arbitrary records, and that this is the result of individual information leakage (as opposed to population-level inference). We then systematically evaluate the tradeoff between protecting privacy and preserving statistical utility. Our findings suggest that current SDG methods cannot consistently provide sufficient privacy protection against inference attacks while retaining reasonable utility. The best method evaluated, a differentially private SDG mechanism, can provide both protection against inference attacks and reasonable utility, but only in very specific settings. Lastly, we show that releasing a larger number of synthetic records can improve utility but at the cost of making attacks far more effective.



## **35. Towards Robust Semantic Segmentation against Patch-based Attack via Attention Refinement**

cs.CV

Accepted by International Journal of Computer Vision (IJCV).34 pages,  5 figures, 16 tables

**SubmitDate**: 2024-05-09    [abs](http://arxiv.org/abs/2401.01750v2) [paper-pdf](http://arxiv.org/pdf/2401.01750v2)

**Authors**: Zheng Yuan, Jie Zhang, Yude Wang, Shiguang Shan, Xilin Chen

**Abstract**: The attention mechanism has been proven effective on various visual tasks in recent years. In the semantic segmentation task, the attention mechanism is applied in various methods, including the case of both Convolution Neural Networks (CNN) and Vision Transformer (ViT) as backbones. However, we observe that the attention mechanism is vulnerable to patch-based adversarial attacks. Through the analysis of the effective receptive field, we attribute it to the fact that the wide receptive field brought by global attention may lead to the spread of the adversarial patch. To address this issue, in this paper, we propose a Robust Attention Mechanism (RAM) to improve the robustness of the semantic segmentation model, which can notably relieve the vulnerability against patch-based attacks. Compared to the vallina attention mechanism, RAM introduces two novel modules called Max Attention Suppression and Random Attention Dropout, both of which aim to refine the attention matrix and limit the influence of a single adversarial patch on the semantic segmentation results of other positions. Extensive experiments demonstrate the effectiveness of our RAM to improve the robustness of semantic segmentation models against various patch-based attack methods under different attack settings.



## **36. TroLLoc: Logic Locking and Layout Hardening for IC Security Closure against Hardware Trojans**

cs.CR

**SubmitDate**: 2024-05-09    [abs](http://arxiv.org/abs/2405.05590v1) [paper-pdf](http://arxiv.org/pdf/2405.05590v1)

**Authors**: Fangzhou Wang, Qijing Wang, Lilas Alrahis, Bangqi Fu, Shui Jiang, Xiaopeng Zhang, Ozgur Sinanoglu, Tsung-Yi Ho, Evangeline F. Y. Young, Johann Knechtel

**Abstract**: Due to cost benefits, supply chains of integrated circuits (ICs) are largely outsourced nowadays. However, passing ICs through various third-party providers gives rise to many security threats, like piracy of IC intellectual property or insertion of hardware Trojans, i.e., malicious circuit modifications.   In this work, we proactively and systematically protect the physical layouts of ICs against post-design insertion of Trojans. Toward that end, we propose TroLLoc, a novel scheme for IC security closure that employs, for the first time, logic locking and layout hardening in unison. TroLLoc is fully integrated into a commercial-grade design flow, and TroLLoc is shown to be effective, efficient, and robust. Our work provides in-depth layout and security analysis considering the challenging benchmarks of the ISPD'22/23 contests for security closure. We show that TroLLoc successfully renders layouts resilient, with reasonable overheads, against (i) general prospects for Trojan insertion as in the ISPD'22 contest, (ii) actual Trojan insertion as in the ISPD'23 contest, and (iii) potential second-order attacks where adversaries would first (i.e., before Trojan insertion) try to bypass the locking defense, e.g., using advanced machine learning attacks. Finally, we release all our artifacts for independent verification [2].



## **37. Poisoning-based Backdoor Attacks for Arbitrary Target Label with Positive Triggers**

cs.CV

**SubmitDate**: 2024-05-09    [abs](http://arxiv.org/abs/2405.05573v1) [paper-pdf](http://arxiv.org/pdf/2405.05573v1)

**Authors**: Binxiao Huang, Jason Chun Lok, Chang Liu, Ngai Wong

**Abstract**: Poisoning-based backdoor attacks expose vulnerabilities in the data preparation stage of deep neural network (DNN) training. The DNNs trained on the poisoned dataset will be embedded with a backdoor, making them behave well on clean data while outputting malicious predictions whenever a trigger is applied. To exploit the abundant information contained in the input data to output label mapping, our scheme utilizes the network trained from the clean dataset as a trigger generator to produce poisons that significantly raise the success rate of backdoor attacks versus conventional approaches. Specifically, we provide a new categorization of triggers inspired by the adversarial technique and develop a multi-label and multi-payload Poisoning-based backdoor attack with Positive Triggers (PPT), which effectively moves the input closer to the target label on benign classifiers. After the classifier is trained on the poisoned dataset, we can generate an input-label-aware trigger to make the infected classifier predict any given input to any target label with a high possibility. Under both dirty- and clean-label settings, we show empirically that the proposed attack achieves a high attack success rate without sacrificing accuracy across various datasets, including SVHN, CIFAR10, GTSRB, and Tiny ImageNet. Furthermore, the PPT attack can elude a variety of classical backdoor defenses, proving its effectiveness.



## **38. Universal Adversarial Perturbations for Vision-Language Pre-trained Models**

cs.CV

9 pages, 5 figures

**SubmitDate**: 2024-05-09    [abs](http://arxiv.org/abs/2405.05524v1) [paper-pdf](http://arxiv.org/pdf/2405.05524v1)

**Authors**: Peng-Fei Zhang, Zi Huang, Guangdong Bai

**Abstract**: Vision-language pre-trained (VLP) models have been the foundation of numerous vision-language tasks. Given their prevalence, it becomes imperative to assess their adversarial robustness, especially when deploying them in security-crucial real-world applications. Traditionally, adversarial perturbations generated for this assessment target specific VLP models, datasets, and/or downstream tasks. This practice suffers from low transferability and additional computation costs when transitioning to new scenarios.   In this work, we thoroughly investigate whether VLP models are commonly sensitive to imperceptible perturbations of a specific pattern for the image modality. To this end, we propose a novel black-box method to generate Universal Adversarial Perturbations (UAPs), which is so called the Effective and T ransferable Universal Adversarial Attack (ETU), aiming to mislead a variety of existing VLP models in a range of downstream tasks. The ETU comprehensively takes into account the characteristics of UAPs and the intrinsic cross-modal interactions to generate effective UAPs. Under this regime, the ETU encourages both global and local utilities of UAPs. This benefits the overall utility while reducing interactions between UAP units, improving the transferability. To further enhance the effectiveness and transferability of UAPs, we also design a novel data augmentation method named ScMix. ScMix consists of self-mix and cross-mix data transformations, which can effectively increase the multi-modal data diversity while preserving the semantics of the original data. Through comprehensive experiments on various downstream tasks, VLP models, and datasets, we demonstrate that the proposed method is able to achieve effective and transferrable universal adversarial attacks.



## **39. Towards Accurate and Robust Architectures via Neural Architecture Search**

cs.CV

Accepted by CVPR2024. arXiv admin note: substantial text overlap with  arXiv:2212.14049

**SubmitDate**: 2024-05-09    [abs](http://arxiv.org/abs/2405.05502v1) [paper-pdf](http://arxiv.org/pdf/2405.05502v1)

**Authors**: Yuwei Ou, Yuqi Feng, Yanan Sun

**Abstract**: To defend deep neural networks from adversarial attacks, adversarial training has been drawing increasing attention for its effectiveness. However, the accuracy and robustness resulting from the adversarial training are limited by the architecture, because adversarial training improves accuracy and robustness by adjusting the weight connection affiliated to the architecture. In this work, we propose ARNAS to search for accurate and robust architectures for adversarial training. First we design an accurate and robust search space, in which the placement of the cells and the proportional relationship of the filter numbers are carefully determined. With the design, the architectures can obtain both accuracy and robustness by deploying accurate and robust structures to their sensitive positions, respectively. Then we propose a differentiable multi-objective search strategy, performing gradient descent towards directions that are beneficial for both natural loss and adversarial loss, thus the accuracy and robustness can be guaranteed at the same time. We conduct comprehensive experiments in terms of white-box attacks, black-box attacks, and transferability. Experimental results show that the searched architecture has the strongest robustness with the competitive accuracy, and breaks the traditional idea that NAS-based architectures cannot transfer well to complex tasks in robustness scenarios. By analyzing outstanding architectures searched, we also conclude that accurate and robust neural architectures tend to deploy different structures near the input and output, which has great practical significance on both hand-crafting and automatically designing of accurate and robust architectures.



## **40. Adversary-Guided Motion Retargeting for Skeleton Anonymization**

cs.CV

**SubmitDate**: 2024-05-08    [abs](http://arxiv.org/abs/2405.05428v1) [paper-pdf](http://arxiv.org/pdf/2405.05428v1)

**Authors**: Thomas Carr, Depeng Xu, Aidong Lu

**Abstract**: Skeleton-based motion visualization is a rising field in computer vision, especially in the case of virtual reality (VR). With further advancements in human-pose estimation and skeleton extracting sensors, more and more applications that utilize skeleton data have come about. These skeletons may appear to be anonymous but they contain embedded personally identifiable information (PII). In this paper we present a new anonymization technique that is based on motion retargeting, utilizing adversary classifiers to further remove PII embedded in the skeleton. Motion retargeting is effective in anonymization as it transfers the movement of the user onto the a dummy skeleton. In doing so, any PII linked to the skeleton will be based on the dummy skeleton instead of the user we are protecting. We propose a Privacy-centric Deep Motion Retargeting model (PMR) which aims to further clear the retargeted skeleton of PII through adversarial learning. In our experiments, PMR achieves motion retargeting utility performance on par with state of the art models while also reducing the performance of privacy attacks.



## **41. Air Gap: Protecting Privacy-Conscious Conversational Agents**

cs.CR

**SubmitDate**: 2024-05-08    [abs](http://arxiv.org/abs/2405.05175v1) [paper-pdf](http://arxiv.org/pdf/2405.05175v1)

**Authors**: Eugene Bagdasaryan, Ren Yi, Sahra Ghalebikesabi, Peter Kairouz, Marco Gruteser, Sewoong Oh, Borja Balle, Daniel Ramage

**Abstract**: The growing use of large language model (LLM)-based conversational agents to manage sensitive user data raises significant privacy concerns. While these agents excel at understanding and acting on context, this capability can be exploited by malicious actors. We introduce a novel threat model where adversarial third-party apps manipulate the context of interaction to trick LLM-based agents into revealing private information not relevant to the task at hand.   Grounded in the framework of contextual integrity, we introduce AirGapAgent, a privacy-conscious agent designed to prevent unintended data leakage by restricting the agent's access to only the data necessary for a specific task. Extensive experiments using Gemini, GPT, and Mistral models as agents validate our approach's effectiveness in mitigating this form of context hijacking while maintaining core agent functionality. For example, we show that a single-query context hijacking attack on a Gemini Ultra agent reduces its ability to protect user data from 94% to 45%, while an AirGapAgent achieves 97% protection, rendering the same attack ineffective.



## **42. Filtering and smoothing estimation algorithms from uncertain nonlinear observations with time-correlated additive noise and random deception attacks**

eess.SP

**SubmitDate**: 2024-05-08    [abs](http://arxiv.org/abs/2405.05157v1) [paper-pdf](http://arxiv.org/pdf/2405.05157v1)

**Authors**: R. Caballero-Águila, J. Hu, J. Linares-Pérez

**Abstract**: This paper discusses the problem of estimating a stochastic signal from nonlinear uncertain observations with time-correlated additive noise described by a first-order Markov process. Random deception attacks are assumed to be launched by an adversary, and both this phenomenon and the uncertainty in the observations are modelled by two sets of Bernoulli random variables. Under the assumption that the evolution model generating the signal to be estimated is unknown and only the mean and covariance functions of the processes involved in the observation equation are available, recursive algorithms based on linear approximations of the real observations are proposed for the least-squares filtering and fixed-point smoothing problems. Finally, the feasibility and effectiveness of the developed estimation algorithms are verified by a numerical simulation example, where the impact of uncertain observation and deception attack probabilities on estimation accuracy is evaluated.



## **43. Towards Efficient Training and Evaluation of Robust Models against $l_0$ Bounded Adversarial Perturbations**

cs.LG

**SubmitDate**: 2024-05-08    [abs](http://arxiv.org/abs/2405.05075v1) [paper-pdf](http://arxiv.org/pdf/2405.05075v1)

**Authors**: Xuyang Zhong, Yixiao Huang, Chen Liu

**Abstract**: This work studies sparse adversarial perturbations bounded by $l_0$ norm. We propose a white-box PGD-like attack method named sparse-PGD to effectively and efficiently generate such perturbations. Furthermore, we combine sparse-PGD with a black-box attack to comprehensively and more reliably evaluate the models' robustness against $l_0$ bounded adversarial perturbations. Moreover, the efficiency of sparse-PGD enables us to conduct adversarial training to build robust models against sparse perturbations. Extensive experiments demonstrate that our proposed attack algorithm exhibits strong performance in different scenarios. More importantly, compared with other robust models, our adversarially trained model demonstrates state-of-the-art robustness against various sparse attacks. Codes are available at https://github.com/CityU-MLO/sPGD.



## **44. Adversarial Threats to Automatic Modulation Open Set Recognition in Wireless Networks**

cs.CR

**SubmitDate**: 2024-05-08    [abs](http://arxiv.org/abs/2405.05022v1) [paper-pdf](http://arxiv.org/pdf/2405.05022v1)

**Authors**: Yandie Yang, Sicheng Zhang, Kuixian Li, Qiao Tian, Yun Lin

**Abstract**: Automatic Modulation Open Set Recognition (AMOSR) is a crucial technological approach for cognitive radio communications, wireless spectrum management, and interference monitoring within wireless networks. Numerous studies have shown that AMR is highly susceptible to minimal perturbations carefully designed by malicious attackers, leading to misclassification of signals. However, the adversarial security issue of AMOSR has not yet been explored. This paper adopts the perspective of attackers and proposes an Open Set Adversarial Attack (OSAttack), aiming at investigating the adversarial vulnerabilities of various AMOSR methods. Initially, an adversarial threat model for AMOSR scenarios is established. Subsequently, by analyzing the decision criteria of both discriminative and generative open set recognition, OSFGSM and OSPGD are proposed to reduce the performance of AMOSR. Finally, the influence of OSAttack on AMOSR is evaluated utilizing a range of qualitative and quantitative indicators. The results indicate that despite the increased resistance of AMOSR models to conventional interference signals, they remain vulnerable to attacks by adversarial examples.



## **45. Deep Reinforcement Learning with Spiking Q-learning**

cs.NE

15 pages, 7 figures

**SubmitDate**: 2024-05-08    [abs](http://arxiv.org/abs/2201.09754v3) [paper-pdf](http://arxiv.org/pdf/2201.09754v3)

**Authors**: Ding Chen, Peixi Peng, Tiejun Huang, Yonghong Tian

**Abstract**: With the help of special neuromorphic hardware, spiking neural networks (SNNs) are expected to realize artificial intelligence (AI) with less energy consumption. It provides a promising energy-efficient way for realistic control tasks by combining SNNs with deep reinforcement learning (RL). There are only a few existing SNN-based RL methods at present. Most of them either lack generalization ability or employ Artificial Neural Networks (ANNs) to estimate value function in training. The former needs to tune numerous hyper-parameters for each scenario, and the latter limits the application of different types of RL algorithm and ignores the large energy consumption in training. To develop a robust spike-based RL method, we draw inspiration from non-spiking interneurons found in insects and propose the deep spiking Q-network (DSQN), using the membrane voltage of non-spiking neurons as the representation of Q-value, which can directly learn robust policies from high-dimensional sensory inputs using end-to-end RL. Experiments conducted on 17 Atari games demonstrate the DSQN is effective and even outperforms the ANN-based deep Q-network (DQN) in most games. Moreover, the experiments show superior learning stability and robustness to adversarial attacks of DSQN.



## **46. Learning-Based Difficulty Calibration for Enhanced Membership Inference Attacks**

cs.CR

**SubmitDate**: 2024-05-08    [abs](http://arxiv.org/abs/2401.04929v2) [paper-pdf](http://arxiv.org/pdf/2401.04929v2)

**Authors**: Haonan Shi, Tu Ouyang, An Wang

**Abstract**: Machine learning models, in particular deep neural networks, are currently an integral part of various applications, from healthcare to finance. However, using sensitive data to train these models raises concerns about privacy and security. One method that has emerged to verify if the trained models are privacy-preserving is Membership Inference Attacks (MIA), which allows adversaries to determine whether a specific data point was part of a model's training dataset. While a series of MIAs have been proposed in the literature, only a few can achieve high True Positive Rates (TPR) in the low False Positive Rate (FPR) region (0.01%~1%). This is a crucial factor to consider for an MIA to be practically useful in real-world settings. In this paper, we present a novel approach to MIA that is aimed at significantly improving TPR at low FPRs. Our method, named learning-based difficulty calibration for MIA(LDC-MIA), characterizes data records by their hardness levels using a neural network classifier to determine membership. The experiment results show that LDC-MIA can improve TPR at low FPR by up to 4x compared to the other difficulty calibration based MIAs. It also has the highest Area Under ROC curve (AUC) across all datasets. Our method's cost is comparable with most of the existing MIAs, but is orders of magnitude more efficient than one of the state-of-the-art methods, LiRA, while achieving similar performance.



## **47. BiasKG: Adversarial Knowledge Graphs to Induce Bias in Large Language Models**

cs.CL

**SubmitDate**: 2024-05-08    [abs](http://arxiv.org/abs/2405.04756v1) [paper-pdf](http://arxiv.org/pdf/2405.04756v1)

**Authors**: Chu Fei Luo, Ahmad Ghawanmeh, Xiaodan Zhu, Faiza Khan Khattak

**Abstract**: Modern large language models (LLMs) have a significant amount of world knowledge, which enables strong performance in commonsense reasoning and knowledge-intensive tasks when harnessed properly. The language model can also learn social biases, which has a significant potential for societal harm. There have been many mitigation strategies proposed for LLM safety, but it is unclear how effective they are for eliminating social biases. In this work, we propose a new methodology for attacking language models with knowledge graph augmented generation. We refactor natural language stereotypes into a knowledge graph, and use adversarial attacking strategies to induce biased responses from several open- and closed-source language models. We find our method increases bias in all models, even those trained with safety guardrails. This demonstrates the need for further research in AI safety, and further work in this new adversarial space.



## **48. Demonstration of an Adversarial Attack Against a Multimodal Vision Language Model for Pathology Imaging**

eess.IV

**SubmitDate**: 2024-05-07    [abs](http://arxiv.org/abs/2401.02565v3) [paper-pdf](http://arxiv.org/pdf/2401.02565v3)

**Authors**: Poojitha Thota, Jai Prakash Veerla, Partha Sai Guttikonda, Mohammad S. Nasr, Shirin Nilizadeh, Jacob M. Luber

**Abstract**: In the context of medical artificial intelligence, this study explores the vulnerabilities of the Pathology Language-Image Pretraining (PLIP) model, a Vision Language Foundation model, under targeted attacks. Leveraging the Kather Colon dataset with 7,180 H&E images across nine tissue types, our investigation employs Projected Gradient Descent (PGD) adversarial perturbation attacks to induce misclassifications intentionally. The outcomes reveal a 100% success rate in manipulating PLIP's predictions, underscoring its susceptibility to adversarial perturbations. The qualitative analysis of adversarial examples delves into the interpretability challenges, shedding light on nuanced changes in predictions induced by adversarial manipulations. These findings contribute crucial insights into the interpretability, domain adaptation, and trustworthiness of Vision Language Models in medical imaging. The study emphasizes the pressing need for robust defenses to ensure the reliability of AI models. The source codes for this experiment can be found at https://github.com/jaiprakash1824/VLM_Adv_Attack.



## **49. Fully Automated Selfish Mining Analysis in Efficient Proof Systems Blockchains**

cs.CR

**SubmitDate**: 2024-05-07    [abs](http://arxiv.org/abs/2405.04420v1) [paper-pdf](http://arxiv.org/pdf/2405.04420v1)

**Authors**: Krishnendu Chatterjee, Amirali Ebrahimzadeh, Mehrdad Karrabi, Krzysztof Pietrzak, Michelle Yeo, Đorđe Žikelić

**Abstract**: We study selfish mining attacks in longest-chain blockchains like Bitcoin, but where the proof of work is replaced with efficient proof systems -- like proofs of stake or proofs of space -- and consider the problem of computing an optimal selfish mining attack which maximizes expected relative revenue of the adversary, thus minimizing the chain quality. To this end, we propose a novel selfish mining attack that aims to maximize this objective and formally model the attack as a Markov decision process (MDP). We then present a formal analysis procedure which computes an $\epsilon$-tight lower bound on the optimal expected relative revenue in the MDP and a strategy that achieves this $\epsilon$-tight lower bound, where $\epsilon>0$ may be any specified precision. Our analysis is fully automated and provides formal guarantees on the correctness. We evaluate our selfish mining attack and observe that it achieves superior expected relative revenue compared to two considered baselines.   In concurrent work [Sarenche FC'24] does an automated analysis on selfish mining in predictable longest-chain blockchains based on efficient proof systems. Predictable means the randomness for the challenges is fixed for many blocks (as used e.g., in Ouroboros), while we consider unpredictable (Bitcoin-like) chains where the challenge is derived from the previous block.



## **50. NeuroIDBench: An Open-Source Benchmark Framework for the Standardization of Methodology in Brainwave-based Authentication Research**

cs.CR

21 pages, 5 Figures, 3 tables, Submitted to the Journal of  Information Security and Applications

**SubmitDate**: 2024-05-07    [abs](http://arxiv.org/abs/2402.08656v4) [paper-pdf](http://arxiv.org/pdf/2402.08656v4)

**Authors**: Avinash Kumar Chaurasia, Matin Fallahi, Thorsten Strufe, Philipp Terhörst, Patricia Arias Cabarcos

**Abstract**: Biometric systems based on brain activity have been proposed as an alternative to passwords or to complement current authentication techniques. By leveraging the unique brainwave patterns of individuals, these systems offer the possibility of creating authentication solutions that are resistant to theft, hands-free, accessible, and potentially even revocable. However, despite the growing stream of research in this area, faster advance is hindered by reproducibility problems. Issues such as the lack of standard reporting schemes for performance results and system configuration, or the absence of common evaluation benchmarks, make comparability and proper assessment of different biometric solutions challenging. Further, barriers are erected to future work when, as so often, source code is not published open access. To bridge this gap, we introduce NeuroIDBench, a flexible open source tool to benchmark brainwave-based authentication models. It incorporates nine diverse datasets, implements a comprehensive set of pre-processing parameters and machine learning algorithms, enables testing under two common adversary models (known vs unknown attacker), and allows researchers to generate full performance reports and visualizations. We use NeuroIDBench to investigate the shallow classifiers and deep learning-based approaches proposed in the literature, and to test robustness across multiple sessions. We observe a 37.6% reduction in Equal Error Rate (EER) for unknown attacker scenarios (typically not tested in the literature), and we highlight the importance of session variability to brainwave authentication. All in all, our results demonstrate the viability and relevance of NeuroIDBench in streamlining fair comparisons of algorithms, thereby furthering the advancement of brainwave-based authentication through robust methodological practices.



