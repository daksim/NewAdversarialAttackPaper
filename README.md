# Latest Adversarial Attack Papers
**update at 2024-04-25 09:46:58**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. OffRAMPS: An FPGA-based Intermediary for Analysis and Modification of Additive Manufacturing Control Systems**

cs.CR

**SubmitDate**: 2024-04-23    [abs](http://arxiv.org/abs/2404.15446v1) [paper-pdf](http://arxiv.org/pdf/2404.15446v1)

**Authors**: Jason Blocklove, Md Raz, Prithwish Basu Roy, Hammond Pearce, Prashanth Krishnamurthy, Farshad Khorrami, Ramesh Karri

**Abstract**: Cybersecurity threats in Additive Manufacturing (AM) are an increasing concern as AM adoption continues to grow. AM is now being used for parts in the aerospace, transportation, and medical domains. Threat vectors which allow for part compromise are particularly concerning, as any failure in these domains would have life-threatening consequences. A major challenge to investigation of AM part-compromises comes from the difficulty in evaluating and benchmarking both identified threat vectors as well as methods for detecting adversarial actions. In this work, we introduce a generalized platform for systematic analysis of attacks against and defenses for 3D printers. Our "OFFRAMPS" platform is based on the open-source 3D printer control board "RAMPS." OFFRAMPS allows analysis, recording, and modification of all control signals and I/O for a 3D printer. We show the efficacy of OFFRAMPS by presenting a series of case studies based on several Trojans, including ones identified in the literature, and show that OFFRAMPS can both emulate and detect these attacks, i.e., it can both change and detect arbitrary changes to the g-code print commands.



## **2. Beyond Text: Utilizing Vocal Cues to Improve Decision Making in LLMs for Robot Navigation Tasks**

cs.AI

28 pages, 7 figures

**SubmitDate**: 2024-04-23    [abs](http://arxiv.org/abs/2402.03494v2) [paper-pdf](http://arxiv.org/pdf/2402.03494v2)

**Authors**: Xingpeng Sun, Haoming Meng, Souradip Chakraborty, Amrit Singh Bedi, Aniket Bera

**Abstract**: While LLMs excel in processing text in these human conversations, they struggle with the nuances of verbal instructions in scenarios like social navigation, where ambiguity and uncertainty can erode trust in robotic and other AI systems. We can address this shortcoming by moving beyond text and additionally focusing on the paralinguistic features of these audio responses. These features are the aspects of spoken communication that do not involve the literal wording (lexical content) but convey meaning and nuance through how something is said. We present \emph{Beyond Text}; an approach that improves LLM decision-making by integrating audio transcription along with a subsection of these features, which focus on the affect and more relevant in human-robot conversations.This approach not only achieves a 70.26\% winning rate, outperforming existing LLMs by 22.16\% to 48.30\% (gemini-1.5-pro and gpt-3.5 respectively), but also enhances robustness against token manipulation adversarial attacks, highlighted by a 22.44\% less decrease ratio than the text-only language model in winning rate. ``\textit{Beyond Text}'' marks an advancement in social robot navigation and broader Human-Robot interactions, seamlessly integrating text-based guidance with human-audio-informed language models.



## **3. JailbreakBench: An Open Robustness Benchmark for Jailbreaking Large Language Models**

cs.CR

**SubmitDate**: 2024-04-23    [abs](http://arxiv.org/abs/2404.01318v2) [paper-pdf](http://arxiv.org/pdf/2404.01318v2)

**Authors**: Patrick Chao, Edoardo Debenedetti, Alexander Robey, Maksym Andriushchenko, Francesco Croce, Vikash Sehwag, Edgar Dobriban, Nicolas Flammarion, George J. Pappas, Florian Tramer, Hamed Hassani, Eric Wong

**Abstract**: Jailbreak attacks cause large language models (LLMs) to generate harmful, unethical, or otherwise objectionable content. Evaluating these attacks presents a number of challenges, which the current collection of benchmarks and evaluation techniques do not adequately address. First, there is no clear standard of practice regarding jailbreaking evaluation. Second, existing works compute costs and success rates in incomparable ways. And third, numerous works are not reproducible, as they withhold adversarial prompts, involve closed-source code, or rely on evolving proprietary APIs. To address these challenges, we introduce JailbreakBench, an open-sourced benchmark with the following components: (1) an evolving repository of state-of-the-art adversarial prompts, which we refer to as jailbreak artifacts; (2) a jailbreaking dataset comprising 100 behaviors -- both original and sourced from prior work -- which align with OpenAI's usage policies; (3) a standardized evaluation framework that includes a clearly defined threat model, system prompts, chat templates, and scoring functions; and (4) a leaderboard that tracks the performance of attacks and defenses for various LLMs. We have carefully considered the potential ethical implications of releasing this benchmark, and believe that it will be a net positive for the community. Over time, we will expand and adapt the benchmark to reflect technical and methodological advances in the research community.



## **4. Differentially-Private Data Synthetisation for Efficient Re-Identification Risk Control**

cs.LG

21 pages, 6 figures and 2 tables

**SubmitDate**: 2024-04-23    [abs](http://arxiv.org/abs/2212.00484v3) [paper-pdf](http://arxiv.org/pdf/2212.00484v3)

**Authors**: Tânia Carvalho, Nuno Moniz, Luís Antunes, Nitesh Chawla

**Abstract**: Protecting user data privacy can be achieved via many methods, from statistical transformations to generative models. However, all of them have critical drawbacks. For example, creating a transformed data set using traditional techniques is highly time-consuming. Also, recent deep learning-based solutions require significant computational resources in addition to long training phases, and differentially private-based solutions may undermine data utility. In this paper, we propose $\epsilon$-PrivateSMOTE, a technique designed for safeguarding against re-identification and linkage attacks, particularly addressing cases with a high \sloppy re-identification risk. Our proposal combines synthetic data generation via noise-induced interpolation with differential privacy principles to obfuscate high-risk cases. We demonstrate how $\epsilon$-PrivateSMOTE is capable of achieving competitive results in privacy risk and better predictive performance when compared to multiple traditional and state-of-the-art privacy-preservation methods, including generative adversarial networks, variational autoencoders, and differential privacy baselines. We also show how our method improves time requirements by at least a factor of 9 and is a resource-efficient solution that ensures high performance without specialised hardware.



## **5. Problem space structural adversarial attacks for Network Intrusion Detection Systems based on Graph Neural Networks**

cs.CR

preprint submitted to IEEE TIFS, under review

**SubmitDate**: 2024-04-23    [abs](http://arxiv.org/abs/2403.11830v2) [paper-pdf](http://arxiv.org/pdf/2403.11830v2)

**Authors**: Andrea Venturi, Dario Stabili, Mirco Marchetti

**Abstract**: Machine Learning (ML) algorithms have become increasingly popular for supporting Network Intrusion Detection Systems (NIDS). Nevertheless, extensive research has shown their vulnerability to adversarial attacks, which involve subtle perturbations to the inputs of the models aimed at compromising their performance. Recent proposals have effectively leveraged Graph Neural Networks (GNN) to produce predictions based also on the structural patterns exhibited by intrusions to enhance the detection robustness. However, the adoption of GNN-based NIDS introduces new types of risks. In this paper, we propose the first formalization of adversarial attacks specifically tailored for GNN in network intrusion detection. Moreover, we outline and model the problem space constraints that attackers need to consider to carry out feasible structural attacks in real-world scenarios. As a final contribution, we conduct an extensive experimental campaign in which we launch the proposed attacks against state-of-the-art GNN-based NIDS. Our findings demonstrate the increased robustness of the models against classical feature-based adversarial attacks, while highlighting their susceptibility to structure-based attacks.



## **6. ALI-DPFL: Differentially Private Federated Learning with Adaptive Local Iterations**

cs.LG

**SubmitDate**: 2024-04-23    [abs](http://arxiv.org/abs/2308.10457v6) [paper-pdf](http://arxiv.org/pdf/2308.10457v6)

**Authors**: Xinpeng Ling, Jie Fu, Kuncan Wang, Haitao Liu, Zhili Chen

**Abstract**: Federated Learning (FL) is a distributed machine learning technique that allows model training among multiple devices or organizations by sharing training parameters instead of raw data. However, adversaries can still infer individual information through inference attacks (e.g. differential attacks) on these training parameters. As a result, Differential Privacy (DP) has been widely used in FL to prevent such attacks.   We consider differentially private federated learning in a resource-constrained scenario, where both privacy budget and communication rounds are constrained. By theoretically analyzing the convergence, we can find the optimal number of local DPSGD iterations for clients between any two sequential global updates. Based on this, we design an algorithm of Differentially Private Federated Learning with Adaptive Local Iterations (ALI-DPFL). We experiment our algorithm on the MNIST, FashionMNIST and Cifar10 datasets, and demonstrate significantly better performances than previous work in the resource-constraint scenario. Code is available at https://github.com/KnightWan/ALI-DPFL.



## **7. Perturbing Attention Gives You More Bang for the Buck: Subtle Imaging Perturbations That Efficiently Fool Customized Diffusion Models**

cs.CV

Published at CVPR 2024

**SubmitDate**: 2024-04-23    [abs](http://arxiv.org/abs/2404.15081v1) [paper-pdf](http://arxiv.org/pdf/2404.15081v1)

**Authors**: Jingyao Xu, Yuetong Lu, Yandong Li, Siyang Lu, Dongdong Wang, Xiang Wei

**Abstract**: Diffusion models (DMs) embark a new era of generative modeling and offer more opportunities for efficient generating high-quality and realistic data samples. However, their widespread use has also brought forth new challenges in model security, which motivates the creation of more effective adversarial attackers on DMs to understand its vulnerability. We propose CAAT, a simple but generic and efficient approach that does not require costly training to effectively fool latent diffusion models (LDMs). The approach is based on the observation that cross-attention layers exhibits higher sensitivity to gradient change, allowing for leveraging subtle perturbations on published images to significantly corrupt the generated images. We show that a subtle perturbation on an image can significantly impact the cross-attention layers, thus changing the mapping between text and image during the fine-tuning of customized diffusion models. Extensive experiments demonstrate that CAAT is compatible with diverse diffusion models and outperforms baseline attack methods in a more effective (more noise) and efficient (twice as fast as Anti-DreamBooth and Mist) manner.



## **8. Formal Verification of Graph Convolutional Networks with Uncertain Node Features and Uncertain Graph Structure**

cs.LG

under review

**SubmitDate**: 2024-04-23    [abs](http://arxiv.org/abs/2404.15065v1) [paper-pdf](http://arxiv.org/pdf/2404.15065v1)

**Authors**: Tobias Ladner, Michael Eichelbeck, Matthias Althoff

**Abstract**: Graph neural networks are becoming increasingly popular in the field of machine learning due to their unique ability to process data structured in graphs. They have also been applied in safety-critical environments where perturbations inherently occur. However, these perturbations require us to formally verify neural networks before their deployment in safety-critical environments as neural networks are prone to adversarial attacks. While there exists research on the formal verification of neural networks, there is no work verifying the robustness of generic graph convolutional network architectures with uncertainty in the node features and in the graph structure over multiple message-passing steps. This work addresses this research gap by explicitly preserving the non-convex dependencies of all elements in the underlying computations through reachability analysis with (matrix) polynomial zonotopes. We demonstrate our approach on three popular benchmark datasets.



## **9. Leverage Variational Graph Representation For Model Poisoning on Federated Learning**

cs.CR

12 pages, 8 figures, 2 tables

**SubmitDate**: 2024-04-23    [abs](http://arxiv.org/abs/2404.15042v1) [paper-pdf](http://arxiv.org/pdf/2404.15042v1)

**Authors**: Kai Li, Xin Yuan, Jingjing Zheng, Wei Ni, Falko Dressler, Abbas Jamalipour

**Abstract**: This paper puts forth a new training data-untethered model poisoning (MP) attack on federated learning (FL). The new MP attack extends an adversarial variational graph autoencoder (VGAE) to create malicious local models based solely on the benign local models overheard without any access to the training data of FL. Such an advancement leads to the VGAE-MP attack that is not only efficacious but also remains elusive to detection. VGAE-MP attack extracts graph structural correlations among the benign local models and the training data features, adversarially regenerates the graph structure, and generates malicious local models using the adversarial graph structure and benign models' features. Moreover, a new attacking algorithm is presented to train the malicious local models using VGAE and sub-gradient descent, while enabling an optimal selection of the benign local models for training the VGAE. Experiments demonstrate a gradual drop in FL accuracy under the proposed VGAE-MP attack and the ineffectiveness of existing defense mechanisms in detecting the attack, posing a severe threat to FL.



## **10. Manipulating Recommender Systems: A Survey of Poisoning Attacks and Countermeasures**

cs.CR

**SubmitDate**: 2024-04-23    [abs](http://arxiv.org/abs/2404.14942v1) [paper-pdf](http://arxiv.org/pdf/2404.14942v1)

**Authors**: Thanh Toan Nguyen, Quoc Viet Hung Nguyen, Thanh Tam Nguyen, Thanh Trung Huynh, Thanh Thi Nguyen, Matthias Weidlich, Hongzhi Yin

**Abstract**: Recommender systems have become an integral part of online services to help users locate specific information in a sea of data. However, existing studies show that some recommender systems are vulnerable to poisoning attacks, particularly those that involve learning schemes. A poisoning attack is where an adversary injects carefully crafted data into the process of training a model, with the goal of manipulating the system's final recommendations. Based on recent advancements in artificial intelligence, such attacks have gained importance recently. While numerous countermeasures to poisoning attacks have been developed, they have not yet been systematically linked to the properties of the attacks. Consequently, assessing the respective risks and potential success of mitigation strategies is difficult, if not impossible. This survey aims to fill this gap by primarily focusing on poisoning attacks and their countermeasures. This is in contrast to prior surveys that mainly focus on attacks and their detection methods. Through an exhaustive literature review, we provide a novel taxonomy for poisoning attacks, formalise its dimensions, and accordingly organise 30+ attacks described in the literature. Further, we review 40+ countermeasures to detect and/or prevent poisoning attacks, evaluating their effectiveness against specific types of attacks. This comprehensive survey should serve as a point of reference for protecting recommender systems against poisoning attacks. The article concludes with a discussion on open issues in the field and impactful directions for future research. A rich repository of resources associated with poisoning attacks is available at https://github.com/tamlhp/awesome-recsys-poisoning.



## **11. Adaptive Hybrid Masking Strategy for Privacy-Preserving Face Recognition Against Model Inversion Attack**

cs.CV

**SubmitDate**: 2024-04-23    [abs](http://arxiv.org/abs/2403.10558v2) [paper-pdf](http://arxiv.org/pdf/2403.10558v2)

**Authors**: Yinggui Wang, Yuanqing Huang, Jianshu Li, Le Yang, Kai Song, Lei Wang

**Abstract**: The utilization of personal sensitive data in training face recognition (FR) models poses significant privacy concerns, as adversaries can employ model inversion attacks (MIA) to infer the original training data. Existing defense methods, such as data augmentation and differential privacy, have been employed to mitigate this issue. However, these methods often fail to strike an optimal balance between privacy and accuracy. To address this limitation, this paper introduces an adaptive hybrid masking algorithm against MIA. Specifically, face images are masked in the frequency domain using an adaptive MixUp strategy. Unlike the traditional MixUp algorithm, which is predominantly used for data augmentation, our modified approach incorporates frequency domain mixing. Previous studies have shown that increasing the number of images mixed in MixUp can enhance privacy preservation but at the expense of reduced face recognition accuracy. To overcome this trade-off, we develop an enhanced adaptive MixUp strategy based on reinforcement learning, which enables us to mix a larger number of images while maintaining satisfactory recognition accuracy. To optimize privacy protection, we propose maximizing the reward function (i.e., the loss function of the FR system) during the training of the strategy network. While the loss function of the FR network is minimized in the phase of training the FR network. The strategy network and the face recognition network can be viewed as antagonistic entities in the training process, ultimately reaching a more balanced trade-off. Experimental results demonstrate that our proposed hybrid masking scheme outperforms existing defense algorithms in terms of privacy preservation and recognition accuracy against MIA.



## **12. Double Privacy Guard: Robust Traceable Adversarial Watermarking against Face Recognition**

cs.CR

**SubmitDate**: 2024-04-23    [abs](http://arxiv.org/abs/2404.14693v1) [paper-pdf](http://arxiv.org/pdf/2404.14693v1)

**Authors**: Yunming Zhang, Dengpan Ye, Sipeng Shen, Caiyun Xie, Ziyi Liu, Jiacheng Deng, Long Tang

**Abstract**: The wide deployment of Face Recognition (FR) systems poses risks of privacy leakage. One countermeasure to address this issue is adversarial attacks, which deceive malicious FR searches but simultaneously interfere the normal identity verification of trusted authorizers. In this paper, we propose the first Double Privacy Guard (DPG) scheme based on traceable adversarial watermarking. DPG employs a one-time watermark embedding to deceive unauthorized FR models and allows authorizers to perform identity verification by extracting the watermark. Specifically, we propose an information-guided adversarial attack against FR models. The encoder embeds an identity-specific watermark into the deep feature space of the carrier, guiding recognizable features of the image to deviate from the source identity. We further adopt a collaborative meta-optimization strategy compatible with sub-tasks, which regularizes the joint optimization direction of the encoder and decoder. This strategy enhances the representation of universal carrier features, mitigating multi-objective optimization conflicts in watermarking. Experiments confirm that DPG achieves significant attack success rates and traceability accuracy on state-of-the-art FR models, exhibiting remarkable robustness that outperforms the existing privacy protection methods using adversarial attacks and deep watermarking, or simple combinations of the two. Our work potentially opens up new insights into proactive protection for FR privacy.



## **13. Pseudorandom Permutations from Random Reversible Circuits**

cs.CC

**SubmitDate**: 2024-04-23    [abs](http://arxiv.org/abs/2404.14648v1) [paper-pdf](http://arxiv.org/pdf/2404.14648v1)

**Authors**: William He, Ryan O'Donnell

**Abstract**: We study pseudorandomness properties of permutations on $\{0,1\}^n$ computed by random circuits made from reversible $3$-bit gates (permutations on $\{0,1\}^3$). Our main result is that a random circuit of depth $n \cdot \tilde{O}(k^2)$, with each layer consisting of $\approx n/3$ random gates in a fixed nearest-neighbor architecture, yields almost $k$-wise independent permutations. The main technical component is showing that the Markov chain on $k$-tuples of $n$-bit strings induced by a single random $3$-bit nearest-neighbor gate has spectral gap at least $1/n \cdot \tilde{O}(k)$. This improves on the original work of Gowers [Gowers96], who showed a gap of $1/\mathrm{poly}(n,k)$ for one random gate (with non-neighboring inputs); and, on subsequent work [HMMR05,BH08] improving the gap to $\Omega(1/n^2k)$ in the same setting.   From the perspective of cryptography, our result can be seen as a particularly simple/practical block cipher construction that gives provable statistical security against attackers with access to $k$~input-output pairs within few rounds. We also show that the Luby--Rackoff construction of pseudorandom permutations from pseudorandom functions can be implemented with reversible circuits. From this, we make progress on the complexity of the Minimum Reversible Circuit Size Problem (MRCSP), showing that block ciphers of fixed polynomial size are computationally secure against arbitrary polynomial-time adversaries, assuming the existence of one-way functions (OWFs).



## **14. RETVec: Resilient and Efficient Text Vectorizer**

cs.CL

37th Conference on Neural Information Processing Systems (NeurIPS  2023)

**SubmitDate**: 2024-04-23    [abs](http://arxiv.org/abs/2302.09207v3) [paper-pdf](http://arxiv.org/pdf/2302.09207v3)

**Authors**: Elie Bursztein, Marina Zhang, Owen Vallis, Xinyu Jia, Alexey Kurakin

**Abstract**: This paper describes RETVec, an efficient, resilient, and multilingual text vectorizer designed for neural-based text processing. RETVec combines a novel character encoding with an optional small embedding model to embed words into a 256-dimensional vector space. The RETVec embedding model is pre-trained using pair-wise metric learning to be robust against typos and character-level adversarial attacks. In this paper, we evaluate and compare RETVec to state-of-the-art vectorizers and word embeddings on popular model architectures and datasets. These comparisons demonstrate that RETVec leads to competitive, multilingual models that are significantly more resilient to typos and adversarial text attacks. RETVec is available under the Apache 2 license at https://github.com/google-research/retvec.



## **15. Image Hijacks: Adversarial Images can Control Generative Models at Runtime**

cs.LG

Project page at https://image-hijacks.github.io

**SubmitDate**: 2024-04-22    [abs](http://arxiv.org/abs/2309.00236v3) [paper-pdf](http://arxiv.org/pdf/2309.00236v3)

**Authors**: Luke Bailey, Euan Ong, Stuart Russell, Scott Emmons

**Abstract**: Are foundation models secure against malicious actors? In this work, we focus on the image input to a vision-language model (VLM). We discover image hijacks, adversarial images that control the behaviour of VLMs at inference time, and introduce the general Behaviour Matching algorithm for training image hijacks. From this, we derive the Prompt Matching method, allowing us to train hijacks matching the behaviour of an arbitrary user-defined text prompt (e.g. 'the Eiffel Tower is now located in Rome') using a generic, off-the-shelf dataset unrelated to our choice of prompt. We use Behaviour Matching to craft hijacks for four types of attack, forcing VLMs to generate outputs of the adversary's choice, leak information from their context window, override their safety training, and believe false statements. We study these attacks against LLaVA, a state-of-the-art VLM based on CLIP and LLaMA-2, and find that all attack types achieve a success rate of over 80%. Moreover, our attacks are automated and require only small image perturbations.



## **16. An Adversarial Approach to Evaluating the Robustness of Event Identification Models**

eess.SY

**SubmitDate**: 2024-04-22    [abs](http://arxiv.org/abs/2402.12338v2) [paper-pdf](http://arxiv.org/pdf/2402.12338v2)

**Authors**: Obai Bahwal, Oliver Kosut, Lalitha Sankar

**Abstract**: Intelligent machine learning approaches are finding active use for event detection and identification that allow real-time situational awareness. Yet, such machine learning algorithms have been shown to be susceptible to adversarial attacks on the incoming telemetry data. This paper considers a physics-based modal decomposition method to extract features for event classification and focuses on interpretable classifiers including logistic regression and gradient boosting to distinguish two types of events: load loss and generation loss. The resulting classifiers are then tested against an adversarial algorithm to evaluate their robustness. The adversarial attack is tested in two settings: the white box setting, wherein the attacker knows exactly the classification model; and the gray box setting, wherein the attacker has access to historical data from the same network as was used to train the classifier, but does not know the classification model. Thorough experiments on the synthetic South Carolina 500-bus system highlight that a relatively simpler model such as logistic regression is more susceptible to adversarial attacks than gradient boosting.



## **17. Automatic Discovery of Visual Circuits**

cs.CV

14 pages, 11 figures

**SubmitDate**: 2024-04-22    [abs](http://arxiv.org/abs/2404.14349v1) [paper-pdf](http://arxiv.org/pdf/2404.14349v1)

**Authors**: Achyuta Rajaram, Neil Chowdhury, Antonio Torralba, Jacob Andreas, Sarah Schwettmann

**Abstract**: To date, most discoveries of network subcomponents that implement human-interpretable computations in deep vision models have involved close study of single units and large amounts of human labor. We explore scalable methods for extracting the subgraph of a vision model's computational graph that underlies recognition of a specific visual concept. We introduce a new method for identifying these subgraphs: specifying a visual concept using a few examples, and then tracing the interdependence of neuron activations across layers, or their functional connectivity. We find that our approach extracts circuits that causally affect model output, and that editing these circuits can defend large pretrained models from adversarial attacks.



## **18. Towards Better Adversarial Purification via Adversarial Denoising Diffusion Training**

cs.CV

**SubmitDate**: 2024-04-22    [abs](http://arxiv.org/abs/2404.14309v1) [paper-pdf](http://arxiv.org/pdf/2404.14309v1)

**Authors**: Yiming Liu, Kezhao Liu, Yao Xiao, Ziyi Dong, Xiaogang Xu, Pengxu Wei, Liang Lin

**Abstract**: Recently, diffusion-based purification (DBP) has emerged as a promising approach for defending against adversarial attacks. However, previous studies have used questionable methods to evaluate the robustness of DBP models, their explanations of DBP robustness also lack experimental support. We re-examine DBP robustness using precise gradient, and discuss the impact of stochasticity on DBP robustness. To better explain DBP robustness, we assess DBP robustness under a novel attack setting, Deterministic White-box, and pinpoint stochasticity as the main factor in DBP robustness. Our results suggest that DBP models rely on stochasticity to evade the most effective attack direction, rather than directly countering adversarial perturbations. To improve the robustness of DBP models, we propose Adversarial Denoising Diffusion Training (ADDT). This technique uses Classifier-Guided Perturbation Optimization (CGPO) to generate adversarial perturbation through guidance from a pre-trained classifier, and uses Rank-Based Gaussian Mapping (RBGM) to convert adversarial pertubation into a normal Gaussian distribution. Empirical results show that ADDT improves the robustness of DBP models. Further experiments confirm that ADDT equips DBP models with the ability to directly counter adversarial perturbations.



## **19. Frosty: Bringing strong liveness guarantees to the Snow family of consensus protocols**

cs.DC

**SubmitDate**: 2024-04-22    [abs](http://arxiv.org/abs/2404.14250v1) [paper-pdf](http://arxiv.org/pdf/2404.14250v1)

**Authors**: Aaron Buchwald, Stephen Buttolph, Andrew Lewis-Pye, Patrick O'Grady, Kevin Sekniqi

**Abstract**: Snowman is the consensus protocol implemented by the Avalanche blockchain and is part of the Snow family of protocols, first introduced through the original Avalanche leaderless consensus protocol. A major advantage of Snowman is that each consensus decision only requires an expected constant communication overhead per processor in the `common' case that the protocol is not under substantial Byzantine attack, i.e. it provides a solution to the scalability problem which ensures that the expected communication overhead per processor is independent of the total number of processors $n$ during normal operation. This is the key property that would enable a consensus protocol to scale to 10,000 or more independent validators (i.e. processors). On the other hand, the two following concerns have remained:   (1) Providing formal proofs of consistency for Snowman has presented a formidable challenge.   (2) Liveness attacks exist in the case that a Byzantine adversary controls more than $O(\sqrt{n})$ processors, slowing termination to more than a logarithmic number of steps.   In this paper, we address the two issues above. We consider a Byzantine adversary that controls at most $f<n/5$ processors. First, we provide a simple proof of consistency for Snowman. Then we supplement Snowman with a `liveness module' that can be triggered in the case that a substantial adversary launches a liveness attack, and which guarantees liveness in this event by temporarily forgoing the communication complexity advantages of Snowman, but without sacrificing these low communication complexity advantages during normal operation.



## **20. Robustness and Visual Explanation for Black Box Image, Video, and ECG Signal Classification with Reinforcement Learning**

cs.LG

AAAI Proceedings reference:  https://ojs.aaai.org/index.php/AAAI/article/view/30579

**SubmitDate**: 2024-04-22    [abs](http://arxiv.org/abs/2403.18985v2) [paper-pdf](http://arxiv.org/pdf/2403.18985v2)

**Authors**: Soumyendu Sarkar, Ashwin Ramesh Babu, Sajad Mousavi, Vineet Gundecha, Avisek Naug, Sahand Ghorbanpour

**Abstract**: We present a generic Reinforcement Learning (RL) framework optimized for crafting adversarial attacks on different model types spanning from ECG signal analysis (1D), image classification (2D), and video classification (3D). The framework focuses on identifying sensitive regions and inducing misclassifications with minimal distortions and various distortion types. The novel RL method outperforms state-of-the-art methods for all three applications, proving its efficiency. Our RL approach produces superior localization masks, enhancing interpretability for image classification and ECG analysis models. For applications such as ECG analysis, our platform highlights critical ECG segments for clinicians while ensuring resilience against prevalent distortions. This comprehensive tool aims to bolster both resilience with adversarial training and transparency across varied applications and data types.



## **21. Secure compilation of rich smart contracts on poor UTXO blockchains**

cs.CR

**SubmitDate**: 2024-04-22    [abs](http://arxiv.org/abs/2305.09545v3) [paper-pdf](http://arxiv.org/pdf/2305.09545v3)

**Authors**: Massimo Bartoletti, Riccardo Marchesin, Roberto Zunino

**Abstract**: Most blockchain platforms from Ethereum onwards render smart contracts as stateful reactive objects that update their state and transfer crypto-assets in response to transactions. A drawback of this design is that when users submit a transaction, they cannot predict in which state it will be executed. This exposes them to transaction-ordering attacks, a widespread class of attacks where adversaries with the power to construct blocks of transactions can extract value from smart contracts (the so-called MEV attacks). The UTXO model is an alternative blockchain design that thwarts these attacks by requiring new transactions to spend past ones: since transactions have unique identifiers, reordering attacks are ineffective. Currently, the blockchains following the UTXO model either provide contracts with limited expressiveness (Bitcoin), or require complex run-time environments (Cardano). We present ILLUM , an Intermediate-Level Language for the UTXO Model. ILLUM can express real-world smart contracts, e.g. those found in Decentralized Finance. We define a compiler from ILLUM to a bare-bone UTXO blockchain with loop-free scripts. Our compilation target only requires minimal extensions to Bitcoin Script: in particular, we exploit covenants, a mechanism for preserving scripts along chains of transactions. We prove the security of our compiler: namely, any attack targeting the compiled contract is also observable at the ILLUM level. Hence, the compiler does not introduce new vulnerabilities that were not already present in the source ILLUM contract. We evaluate the practicality of ILLUM as a compilation target for higher-level languages. To this purpose, we implement a compiler from a contract language inspired by Solidity to ILLUM, and we apply it to a benchmark or real-world smart contracts.



## **22. Protecting Your LLMs with Information Bottleneck**

cs.CL

**SubmitDate**: 2024-04-22    [abs](http://arxiv.org/abs/2404.13968v1) [paper-pdf](http://arxiv.org/pdf/2404.13968v1)

**Authors**: Zichuan Liu, Zefan Wang, Linjie Xu, Jinyu Wang, Lei Song, Tianchun Wang, Chunlin Chen, Wei Cheng, Jiang Bian

**Abstract**: The advent of large language models (LLMs) has revolutionized the field of natural language processing, yet they might be attacked to produce harmful content. Despite efforts to ethically align LLMs, these are often fragile and can be circumvented by jailbreaking attacks through optimized or manual adversarial prompts. To address this, we introduce the Information Bottleneck Protector (IBProtector), a defense mechanism grounded in the information bottleneck principle, and we modify the objective to avoid trivial solutions. The IBProtector selectively compresses and perturbs prompts, facilitated by a lightweight and trainable extractor, preserving only essential information for the target LLMs to respond with the expected answer. Moreover, we further consider a situation where the gradient is not visible to be compatible with any LLM. Our empirical evaluations show that IBProtector outperforms current defense methods in mitigating jailbreak attempts, without overly affecting response quality or inference speed. Its effectiveness and adaptability across various attack methods and target LLMs underscore the potential of IBProtector as a novel, transferable defense that bolsters the security of LLMs without requiring modifications to the underlying models.



## **23. Audio Anti-Spoofing Detection: A Survey**

cs.SD

submitted to ACM Computing Surveys

**SubmitDate**: 2024-04-22    [abs](http://arxiv.org/abs/2404.13914v1) [paper-pdf](http://arxiv.org/pdf/2404.13914v1)

**Authors**: Menglu Li, Yasaman Ahmadiadli, Xiao-Ping Zhang

**Abstract**: The availability of smart devices leads to an exponential increase in multimedia content. However, the rapid advancements in deep learning have given rise to sophisticated algorithms capable of manipulating or creating multimedia fake content, known as Deepfake. Audio Deepfakes pose a significant threat by producing highly realistic voices, thus facilitating the spread of misinformation. To address this issue, numerous audio anti-spoofing detection challenges have been organized to foster the development of anti-spoofing countermeasures. This survey paper presents a comprehensive review of every component within the detection pipeline, including algorithm architectures, optimization techniques, application generalizability, evaluation metrics, performance comparisons, available datasets, and open-source availability. For each aspect, we conduct a systematic evaluation of the recent advancements, along with discussions on existing challenges. Additionally, we also explore emerging research topics on audio anti-spoofing, including partial spoofing detection, cross-dataset evaluation, and adversarial attack defence, while proposing some promising research directions for future work. This survey paper not only identifies the current state-of-the-art to establish strong baselines for future experiments but also guides future researchers on a clear path for understanding and enhancing the audio anti-spoofing detection mechanisms.



## **24. Competition Report: Finding Universal Jailbreak Backdoors in Aligned LLMs**

cs.CL

Competition Report

**SubmitDate**: 2024-04-22    [abs](http://arxiv.org/abs/2404.14461v1) [paper-pdf](http://arxiv.org/pdf/2404.14461v1)

**Authors**: Javier Rando, Francesco Croce, Kryštof Mitka, Stepan Shabalin, Maksym Andriushchenko, Nicolas Flammarion, Florian Tramèr

**Abstract**: Large language models are aligned to be safe, preventing users from generating harmful content like misinformation or instructions for illegal activities. However, previous work has shown that the alignment process is vulnerable to poisoning attacks. Adversaries can manipulate the safety training data to inject backdoors that act like a universal sudo command: adding the backdoor string to any prompt enables harmful responses from models that, otherwise, behave safely. Our competition, co-located at IEEE SaTML 2024, challenged participants to find universal backdoors in several large language models. This report summarizes the key findings and promising ideas for future research.



## **25. Distributional Black-Box Model Inversion Attack with Multi-Agent Reinforcement Learning**

cs.LG

**SubmitDate**: 2024-04-22    [abs](http://arxiv.org/abs/2404.13860v1) [paper-pdf](http://arxiv.org/pdf/2404.13860v1)

**Authors**: Huan Bao, Kaimin Wei, Yongdong Wu, Jin Qian, Robert H. Deng

**Abstract**: A Model Inversion (MI) attack based on Generative Adversarial Networks (GAN) aims to recover the private training data from complex deep learning models by searching codes in the latent space. However, they merely search a deterministic latent space such that the found latent code is usually suboptimal. In addition, the existing distributional MI schemes assume that an attacker can access the structures and parameters of the target model, which is not always viable in practice. To overcome the above shortcomings, this paper proposes a novel Distributional Black-Box Model Inversion (DBB-MI) attack by constructing the probabilistic latent space for searching the target privacy data. Specifically, DBB-MI does not need the target model parameters or specialized GAN training. Instead, it finds the latent probability distribution by combining the output of the target model with multi-agent reinforcement learning techniques. Then, it randomly chooses latent codes from the latent probability distribution for recovering the private data. As the latent probability distribution closely aligns with the target privacy data in latent space, the recovered data will leak the privacy of training samples of the target model significantly. Abundant experiments conducted on diverse datasets and networks show that the present DBB-MI has better performance than state-of-the-art in attack accuracy, K-nearest neighbor feature distance, and Peak Signal-to-Noise Ratio.



## **26. Concept Arithmetics for Circumventing Concept Inhibition in Diffusion Models**

cs.CV

**SubmitDate**: 2024-04-21    [abs](http://arxiv.org/abs/2404.13706v1) [paper-pdf](http://arxiv.org/pdf/2404.13706v1)

**Authors**: Vitali Petsiuk, Kate Saenko

**Abstract**: Motivated by ethical and legal concerns, the scientific community is actively developing methods to limit the misuse of Text-to-Image diffusion models for reproducing copyrighted, violent, explicit, or personal information in the generated images. Simultaneously, researchers put these newly developed safety measures to the test by assuming the role of an adversary to find vulnerabilities and backdoors in them. We use compositional property of diffusion models, which allows to leverage multiple prompts in a single image generation. This property allows us to combine other concepts, that should not have been affected by the inhibition, to reconstruct the vector, responsible for target concept generation, even though the direct computation of this vector is no longer accessible. We provide theoretical and empirical evidence why the proposed attacks are possible and discuss the implications of these findings for safe model deployment. We argue that it is essential to consider all possible approaches to image generation with diffusion models that can be employed by an adversary. Our work opens up the discussion about the implications of concept arithmetics and compositional inference for safety mechanisms in diffusion models.   Content Advisory: This paper contains discussions and model-generated content that may be considered offensive. Reader discretion is advised.   Project page: https://cs-people.bu.edu/vpetsiuk/arc



## **27. Large Language Models for Blockchain Security: A Systematic Literature Review**

cs.CR

**SubmitDate**: 2024-04-21    [abs](http://arxiv.org/abs/2403.14280v3) [paper-pdf](http://arxiv.org/pdf/2403.14280v3)

**Authors**: Zheyuan He, Zihao Li, Sen Yang

**Abstract**: Large Language Models (LLMs) have emerged as powerful tools in various domains involving blockchain security (BS). Several recent studies are exploring LLMs applied to BS. However, there remains a gap in our understanding regarding the full scope of applications, impacts, and potential constraints of LLMs on blockchain security. To fill this gap, we conduct a literature review on LLM4BS.   As the first review of LLM's application on blockchain security, our study aims to comprehensively analyze existing research and elucidate how LLMs contribute to enhancing the security of blockchain systems. Through a thorough examination of scholarly works, we delve into the integration of LLMs into various aspects of blockchain security. We explore the mechanisms through which LLMs can bolster blockchain security, including their applications in smart contract auditing, identity verification, anomaly detection, vulnerable repair, and so on. Furthermore, we critically assess the challenges and limitations associated with leveraging LLMs for blockchain security, considering factors such as scalability, privacy concerns, and adversarial attacks. Our review sheds light on the opportunities and potential risks inherent in this convergence, providing valuable insights for researchers, practitioners, and policymakers alike.



## **28. Attack on Scene Flow using Point Clouds**

cs.CV

**SubmitDate**: 2024-04-21    [abs](http://arxiv.org/abs/2404.13621v1) [paper-pdf](http://arxiv.org/pdf/2404.13621v1)

**Authors**: Haniyeh Ehsani Oskouie, Mohammad-Shahram Moin, Shohreh Kasaei

**Abstract**: Deep neural networks have made significant advancements in accurately estimating scene flow using point clouds, which is vital for many applications like video analysis, action recognition, and navigation. Robustness of these techniques, however, remains a concern, particularly in the face of adversarial attacks that have been proven to deceive state-of-the-art deep neural networks in many domains. Surprisingly, the robustness of scene flow networks against such attacks has not been thoroughly investigated. To address this problem, the proposed approach aims to bridge this gap by introducing adversarial white-box attacks specifically tailored for scene flow networks. Experimental results show that the generated adversarial examples obtain up to 33.7 relative degradation in average end-point error on the KITTI and FlyingThings3D datasets. The study also reveals the significant impact that attacks targeting point clouds in only one dimension or color channel have on average end-point error. Analyzing the success and failure of these attacks on the scene flow networks and their 2D optical flow network variants show a higher vulnerability for the optical flow networks.



## **29. Robust EEG-based Emotion Recognition Using an Inception and Two-sided Perturbation Model**

eess.SP

**SubmitDate**: 2024-04-21    [abs](http://arxiv.org/abs/2404.15373v1) [paper-pdf](http://arxiv.org/pdf/2404.15373v1)

**Authors**: Shadi Sartipi, Mujdat Cetin

**Abstract**: Automated emotion recognition using electroencephalogram (EEG) signals has gained substantial attention. Although deep learning approaches exhibit strong performance, they often suffer from vulnerabilities to various perturbations, like environmental noise and adversarial attacks. In this paper, we propose an Inception feature generator and two-sided perturbation (INC-TSP) approach to enhance emotion recognition in brain-computer interfaces. INC-TSP integrates the Inception module for EEG data analysis and employs two-sided perturbation (TSP) as a defensive mechanism against input perturbations. TSP introduces worst-case perturbations to the model's weights and inputs, reinforcing the model's elasticity against adversarial attacks. The proposed approach addresses the challenge of maintaining accurate emotion recognition in the presence of input uncertainties. We validate INC-TSP in a subject-independent three-class emotion recognition scenario, demonstrating robust performance.



## **30. How to Evaluate Semantic Communications for Images with ViTScore Metric?**

cs.CV

**SubmitDate**: 2024-04-21    [abs](http://arxiv.org/abs/2309.04891v2) [paper-pdf](http://arxiv.org/pdf/2309.04891v2)

**Authors**: Tingting Zhu, Bo Peng, Jifan Liang, Tingchen Han, Hai Wan, Jingqiao Fu, Junjie Chen

**Abstract**: Semantic communications (SC) have been expected to be a new paradigm shifting to catalyze the next generation communication, whose main concerns shift from accurate bit transmission to effective semantic information exchange in communications. However, the previous and widely-used metrics for images are not applicable to evaluate the image semantic similarity in SC. Classical metrics to measure the similarity between two images usually rely on the pixel level or the structural level, such as the PSNR and the MS-SSIM. Straightforwardly using some tailored metrics based on deep-learning methods in CV community, such as the LPIPS, is infeasible for SC. To tackle this, inspired by BERTScore in NLP community, we propose a novel metric for evaluating image semantic similarity, named Vision Transformer Score (ViTScore). We prove theoretically that ViTScore has 3 important properties, including symmetry, boundedness, and normalization, which make ViTScore convenient and intuitive for image measurement. To evaluate the performance of ViTScore, we compare ViTScore with 3 typical metrics (PSNR, MS-SSIM, and LPIPS) through 4 classes of experiments: (i) correlation with BERTScore through evaluation of image caption downstream CV task, (ii) evaluation in classical image communications, (iii) evaluation in image semantic communication systems, and (iv) evaluation in image semantic communication systems with semantic attack. Experimental results demonstrate that ViTScore is robust and efficient in evaluating the semantic similarity of images. Particularly, ViTScore outperforms the other 3 typical metrics in evaluating the image semantic changes by semantic attack, such as image inverse with Generative Adversarial Networks (GANs). This indicates that ViTScore is an effective performance metric when deployed in SC scenarios.



## **31. Reliable Model Watermarking: Defending Against Theft without Compromising on Evasion**

cs.CR

**SubmitDate**: 2024-04-21    [abs](http://arxiv.org/abs/2404.13518v1) [paper-pdf](http://arxiv.org/pdf/2404.13518v1)

**Authors**: Hongyu Zhu, Sichu Liang, Wentao Hu, Fangqi Li, Ju Jia, Shilin Wang

**Abstract**: With the rise of Machine Learning as a Service (MLaaS) platforms,safeguarding the intellectual property of deep learning models is becoming paramount. Among various protective measures, trigger set watermarking has emerged as a flexible and effective strategy for preventing unauthorized model distribution. However, this paper identifies an inherent flaw in the current paradigm of trigger set watermarking: evasion adversaries can readily exploit the shortcuts created by models memorizing watermark samples that deviate from the main task distribution, significantly impairing their generalization in adversarial settings. To counteract this, we leverage diffusion models to synthesize unrestricted adversarial examples as trigger sets. By learning the model to accurately recognize them, unique watermark behaviors are promoted through knowledge injection rather than error memorization, thus avoiding exploitable shortcuts. Furthermore, we uncover that the resistance of current trigger set watermarking against removal attacks primarily relies on significantly damaging the decision boundaries during embedding, intertwining unremovability with adverse impacts. By optimizing the knowledge transfer properties of protected models, our approach conveys watermark behaviors to extraction surrogates without aggressively decision boundary perturbation. Experimental results on CIFAR-10/100 and Imagenette datasets demonstrate the effectiveness of our method, showing not only improved robustness against evasion adversaries but also superior resistance to watermark removal attacks compared to state-of-the-art solutions.



## **32. Machine Learning Robustness: A Primer**

cs.LG

**SubmitDate**: 2024-04-21    [abs](http://arxiv.org/abs/2404.00897v2) [paper-pdf](http://arxiv.org/pdf/2404.00897v2)

**Authors**: Houssem Ben Braiek, Foutse Khomh

**Abstract**: This chapter explores the foundational concept of robustness in Machine Learning (ML) and its integral role in establishing trustworthiness in Artificial Intelligence (AI) systems. The discussion begins with a detailed definition of robustness, portraying it as the ability of ML models to maintain stable performance across varied and unexpected environmental conditions. ML robustness is dissected through several lenses: its complementarity with generalizability; its status as a requirement for trustworthy AI; its adversarial vs non-adversarial aspects; its quantitative metrics; and its indicators such as reproducibility and explainability. The chapter delves into the factors that impede robustness, such as data bias, model complexity, and the pitfalls of underspecified ML pipelines. It surveys key techniques for robustness assessment from a broad perspective, including adversarial attacks, encompassing both digital and physical realms. It covers non-adversarial data shifts and nuances of Deep Learning (DL) software testing methodologies. The discussion progresses to explore amelioration strategies for bolstering robustness, starting with data-centric approaches like debiasing and augmentation. Further examination includes a variety of model-centric methods such as transfer learning, adversarial training, and randomized smoothing. Lastly, post-training methods are discussed, including ensemble techniques, pruning, and model repairs, emerging as cost-effective strategies to make models more resilient against the unpredictable. This chapter underscores the ongoing challenges and limitations in estimating and achieving ML robustness by existing approaches. It offers insights and directions for future research on this crucial concept, as a prerequisite for trustworthy AI systems.



## **33. Deepfake Generation and Detection: A Benchmark and Survey**

cs.CV

We closely follow the latest developments in  https://github.com/flyingby/Awesome-Deepfake-Generation-and-Detection

**SubmitDate**: 2024-04-20    [abs](http://arxiv.org/abs/2403.17881v3) [paper-pdf](http://arxiv.org/pdf/2403.17881v3)

**Authors**: Gan Pei, Jiangning Zhang, Menghan Hu, Zhenyu Zhang, Chengjie Wang, Yunsheng Wu, Guangtao Zhai, Jian Yang, Chunhua Shen, Dacheng Tao

**Abstract**: Deepfake is a technology dedicated to creating highly realistic facial images and videos under specific conditions, which has significant application potential in fields such as entertainment, movie production, digital human creation, to name a few. With the advancements in deep learning, techniques primarily represented by Variational Autoencoders and Generative Adversarial Networks have achieved impressive generation results. More recently, the emergence of diffusion models with powerful generation capabilities has sparked a renewed wave of research. In addition to deepfake generation, corresponding detection technologies continuously evolve to regulate the potential misuse of deepfakes, such as for privacy invasion and phishing attacks. This survey comprehensively reviews the latest developments in deepfake generation and detection, summarizing and analyzing current state-of-the-arts in this rapidly evolving field. We first unify task definitions, comprehensively introduce datasets and metrics, and discuss developing technologies. Then, we discuss the development of several related sub-fields and focus on researching four representative deepfake fields: face swapping, face reenactment, talking face generation, and facial attribute editing, as well as forgery detection. Subsequently, we comprehensively benchmark representative methods on popular datasets for each field, fully evaluating the latest and influential published works. Finally, we analyze challenges and future research directions of the discussed fields.



## **34. Pixel is a Barrier: Diffusion Models Are More Adversarially Robust Than We Think**

cs.CV

**SubmitDate**: 2024-04-20    [abs](http://arxiv.org/abs/2404.13320v1) [paper-pdf](http://arxiv.org/pdf/2404.13320v1)

**Authors**: Haotian Xue, Yongxin Chen

**Abstract**: Adversarial examples for diffusion models are widely used as solutions for safety concerns. By adding adversarial perturbations to personal images, attackers can not edit or imitate them easily. However, it is essential to note that all these protections target the latent diffusion model (LDMs), the adversarial examples for diffusion models in the pixel space (PDMs) are largely overlooked. This may mislead us to think that the diffusion models are vulnerable to adversarial attacks like most deep models. In this paper, we show novel findings that: even though gradient-based white-box attacks can be used to attack the LDMs, they fail to attack PDMs. This finding is supported by extensive experiments of almost a wide range of attacking methods on various PDMs and LDMs with different model structures, which means diffusion models are indeed much more robust against adversarial attacks. We also find that PDMs can be used as an off-the-shelf purifier to effectively remove the adversarial patterns that were generated on LDMs to protect the images, which means that most protection methods nowadays, to some extent, cannot protect our images from malicious attacks. We hope that our insights will inspire the community to rethink the adversarial samples for diffusion models as protection methods and move forward to more effective protection. Codes are available in https://github.com/xavihart/PDM-Pure.



## **35. Backdoor Attacks and Defenses on Semantic-Symbol Reconstruction in Semantic Communications**

cs.CR

This paper has been accepted by IEEE ICC 2024

**SubmitDate**: 2024-04-20    [abs](http://arxiv.org/abs/2404.13279v1) [paper-pdf](http://arxiv.org/pdf/2404.13279v1)

**Authors**: Yuan Zhou, Rose Qingyang Hu, Yi Qian

**Abstract**: Semantic communication is of crucial importance for the next-generation wireless communication networks. The existing works have developed semantic communication frameworks based on deep learning. However, systems powered by deep learning are vulnerable to threats such as backdoor attacks and adversarial attacks. This paper delves into backdoor attacks targeting deep learning-enabled semantic communication systems. Since current works on backdoor attacks are not tailored for semantic communication scenarios, a new backdoor attack paradigm on semantic symbols (BASS) is introduced, based on which the corresponding defense measures are designed. Specifically, a training framework is proposed to prevent BASS. Additionally, reverse engineering-based and pruning-based defense strategies are designed to protect against backdoor attacks in semantic communication. Simulation results demonstrate the effectiveness of both the proposed attack paradigm and the defense strategies.



## **36. Beyond Score Changes: Adversarial Attack on No-Reference Image Quality Assessment from Two Perspectives**

eess.IV

Submitted to a conference

**SubmitDate**: 2024-04-20    [abs](http://arxiv.org/abs/2404.13277v1) [paper-pdf](http://arxiv.org/pdf/2404.13277v1)

**Authors**: Chenxi Yang, Yujia Liu, Dingquan Li, Yan Zhong, Tingting Jiang

**Abstract**: Deep neural networks have demonstrated impressive success in No-Reference Image Quality Assessment (NR-IQA). However, recent researches highlight the vulnerability of NR-IQA models to subtle adversarial perturbations, leading to inconsistencies between model predictions and subjective ratings. Current adversarial attacks, however, focus on perturbing predicted scores of individual images, neglecting the crucial aspect of inter-score correlation relationships within an entire image set. Meanwhile, it is important to note that the correlation, like ranking correlation, plays a significant role in NR-IQA tasks. To comprehensively explore the robustness of NR-IQA models, we introduce a new framework of correlation-error-based attacks that perturb both the correlation within an image set and score changes on individual images. Our research primarily focuses on ranking-related correlation metrics like Spearman's Rank-Order Correlation Coefficient (SROCC) and prediction error-related metrics like Mean Squared Error (MSE). As an instantiation, we propose a practical two-stage SROCC-MSE-Attack (SMA) that initially optimizes target attack scores for the entire image set and then generates adversarial examples guided by these scores. Experimental results demonstrate that our SMA method not only significantly disrupts the SROCC to negative values but also maintains a considerable change in the scores of individual images. Meanwhile, it exhibits state-of-the-art performance across metrics with different categories. Our method provides a new perspective on the robustness of NR-IQA models.



## **37. The Instruction Hierarchy: Training LLMs to Prioritize Privileged Instructions**

cs.CR

**SubmitDate**: 2024-04-19    [abs](http://arxiv.org/abs/2404.13208v1) [paper-pdf](http://arxiv.org/pdf/2404.13208v1)

**Authors**: Eric Wallace, Kai Xiao, Reimar Leike, Lilian Weng, Johannes Heidecke, Alex Beutel

**Abstract**: Today's LLMs are susceptible to prompt injections, jailbreaks, and other attacks that allow adversaries to overwrite a model's original instructions with their own malicious prompts. In this work, we argue that one of the primary vulnerabilities underlying these attacks is that LLMs often consider system prompts (e.g., text from an application developer) to be the same priority as text from untrusted users and third parties. To address this, we propose an instruction hierarchy that explicitly defines how models should behave when instructions of different priorities conflict. We then propose a data generation method to demonstrate this hierarchical instruction following behavior, which teaches LLMs to selectively ignore lower-privileged instructions. We apply this method to GPT-3.5, showing that it drastically increases robustness -- even for attack types not seen during training -- while imposing minimal degradations on standard capabilities.



## **38. Artwork Protection Against Neural Style Transfer Using Locally Adaptive Adversarial Color Attack**

cs.CV

9 pages, 5 figures, 4 tables

**SubmitDate**: 2024-04-19    [abs](http://arxiv.org/abs/2401.09673v2) [paper-pdf](http://arxiv.org/pdf/2401.09673v2)

**Authors**: Zhongliang Guo, Junhao Dong, Yifei Qian, Kaixuan Wang, Weiye Li, Ziheng Guo, Yuheng Wang, Yanli Li, Ognjen Arandjelović, Lei Fang

**Abstract**: Neural style transfer (NST) generates new images by combining the style of one image with the content of another. However, unauthorized NST can exploit artwork, raising concerns about artists' rights and motivating the development of proactive protection methods. We propose Locally Adaptive Adversarial Color Attack (LAACA), empowering artists to protect their artwork from unauthorized style transfer by processing before public release. By delving into the intricacies of human visual perception and the role of different frequency components, our method strategically introduces frequency-adaptive perturbations in the image. These perturbations significantly degrade the generation quality of NST while maintaining an acceptable level of visual change in the original image, ensuring that potential infringers are discouraged from using the protected artworks, because of its bad NST generation quality. Additionally, existing metrics often overlook the importance of color fidelity in evaluating color-mattered tasks, such as the quality of NST-generated images, which is crucial in the context of artistic works. To comprehensively assess the color-mattered tasks, we propose the Adversarial Color Distance Metric (ACDM), designed to quantify the color difference of images pre- and post-manipulations. Experimental results confirm that attacking NST using LAACA results in visually inferior style transfer, and the ACDM can efficiently measure color-mattered tasks. By providing artists with a tool to safeguard their intellectual property, our work relieves the socio-technical challenges posed by the misuse of NST in the art community.



## **39. Set-Based Training for Neural Network Verification**

cs.LG

**SubmitDate**: 2024-04-19    [abs](http://arxiv.org/abs/2401.14961v2) [paper-pdf](http://arxiv.org/pdf/2401.14961v2)

**Authors**: Lukas Koller, Tobias Ladner, Matthias Althoff

**Abstract**: Neural networks are vulnerable to adversarial attacks, i.e., small input perturbations can significantly affect the outputs of a neural network. In safety-critical environments, the inputs often contain noisy sensor data; hence, in this case, neural networks that are robust against input perturbations are required. To ensure safety, the robustness of a neural network must be formally verified. However, training and formally verifying robust neural networks is challenging. We address both of these challenges by employing, for the first time, an end-to-end set-based training procedure that trains robust neural networks for formal verification. Our training procedure trains neural networks, which can be easily verified using simple polynomial-time verification algorithms. Moreover, our extensive evaluation demonstrates that our set-based training procedure effectively trains robust neural networks, which are easier to verify. Set-based trained neural networks consistently match or outperform those trained with state-of-the-art robust training approaches.



## **40. Predominant Aspects on Security for Quantum Machine Learning: Literature Review**

quant-ph

**SubmitDate**: 2024-04-19    [abs](http://arxiv.org/abs/2401.07774v2) [paper-pdf](http://arxiv.org/pdf/2401.07774v2)

**Authors**: Nicola Franco, Alona Sakhnenko, Leon Stolpmann, Daniel Thuerck, Fabian Petsch, Annika Rüll, Jeanette Miriam Lorenz

**Abstract**: Quantum Machine Learning (QML) has emerged as a promising intersection of quantum computing and classical machine learning, anticipated to drive breakthroughs in computational tasks. This paper discusses the question which security concerns and strengths are connected to QML by means of a systematic literature review. We categorize and review the security of QML models, their vulnerabilities inherent to quantum architectures, and the mitigation strategies proposed. The survey reveals that while QML possesses unique strengths, it also introduces novel attack vectors not seen in classical systems. We point out specific risks, such as cross-talk in superconducting systems and forced repeated shuttle operations in ion-trap systems, which threaten QML's reliability. However, approaches like adversarial training, quantum noise exploitation, and quantum differential privacy have shown potential in enhancing QML robustness. Our review discuss the need for continued and rigorous research to ensure the secure deployment of QML in real-world applications. This work serves as a foundational reference for researchers and practitioners aiming to navigate the security aspects of QML.



## **41. A Proactive Decoy Selection Scheme for Cyber Deception using MITRE ATT&CK**

cs.CR

**SubmitDate**: 2024-04-19    [abs](http://arxiv.org/abs/2404.12783v1) [paper-pdf](http://arxiv.org/pdf/2404.12783v1)

**Authors**: Marco Zambianco, Claudio Facchinetti, Domenico Siracusa

**Abstract**: Cyber deception allows compensating the late response of defenders countermeasures to the ever evolving tactics, techniques, and procedures (TTPs) of attackers. This proactive defense strategy employs decoys resembling legitimate system components to lure stealthy attackers within the defender environment, slowing and/or denying the accomplishment of their goals. In this regard, the selection of decoys that can expose the techniques used by malicious users plays a central role to incentivize their engagement. However, this is a difficult task to achieve in practice, since it requires an accurate and realistic modeling of the attacker capabilities and his possible targets. In this work, we tackle this challenge and we design a decoy selection scheme that is supported by an adversarial modeling based on empirical observation of real-world attackers. We take advantage of a domain-specific threat modelling language using MITRE ATT&CK framework as source of attacker TTPs targeting enterprise systems. In detail, we extract the information about the execution preconditions of each technique as well as its possible effects on the environment to generate attack graphs modeling the adversary capabilities. Based on this, we formulate a graph partition problem that minimizes the number of decoys detecting a corresponding number of techniques employed in various attack paths directed to specific targets. We compare our optimization-based decoy selection approach against several benchmark schemes that ignore the preconditions between the various attack steps. Results reveal that the proposed scheme provides the highest interception rate of attack paths using the lowest amount of decoys.



## **42. Defending against Data Poisoning Attacks in Federated Learning via User Elimination**

cs.CR

To be submitted in AISEC 2024

**SubmitDate**: 2024-04-19    [abs](http://arxiv.org/abs/2404.12778v1) [paper-pdf](http://arxiv.org/pdf/2404.12778v1)

**Authors**: Nick Galanis

**Abstract**: In the evolving landscape of Federated Learning (FL), a new type of attacks concerns the research community, namely Data Poisoning Attacks, which threaten the model integrity by maliciously altering training data. This paper introduces a novel defensive framework focused on the strategic elimination of adversarial users within a federated model. We detect those anomalies in the aggregation phase of the Federated Algorithm, by integrating metadata gathered by the local training instances with Differential Privacy techniques, to ensure that no data leakage is possible. To our knowledge, this is the first proposal in the field of FL that leverages metadata other than the model's gradients in order to ensure honesty in the reported local models. Our extensive experiments demonstrate the efficacy of our methods, significantly mitigating the risk of data poisoning while maintaining user privacy and model performance. Our findings suggest that this new user elimination approach serves us with a great balance between privacy and utility, thus contributing to the arsenal of arguments in favor of the safe adoption of FL in safe domains, both in academic setting and in the industry.



## **43. A Clean-graph Backdoor Attack against Graph Convolutional Networks with Poisoned Label Only**

cs.AI

**SubmitDate**: 2024-04-19    [abs](http://arxiv.org/abs/2404.12704v1) [paper-pdf](http://arxiv.org/pdf/2404.12704v1)

**Authors**: Jiazhu Dai, Haoyu Sun

**Abstract**: Graph Convolutional Networks (GCNs) have shown excellent performance in dealing with various graph structures such as node classification, graph classification and other tasks. However,recent studies have shown that GCNs are vulnerable to a novel threat known as backdoor attacks. However, all existing backdoor attacks in the graph domain require modifying the training samples to accomplish the backdoor injection, which may not be practical in many realistic scenarios where adversaries have no access to modify the training samples and may leads to the backdoor attack being detected easily. In order to explore the backdoor vulnerability of GCNs and create a more practical and stealthy backdoor attack method, this paper proposes a clean-graph backdoor attack against GCNs (CBAG) in the node classification task,which only poisons the training labels without any modification to the training samples, revealing that GCNs have this security vulnerability. Specifically, CBAG designs a new trigger exploration method to find important feature dimensions as the trigger patterns to improve the attack performance. By poisoning the training labels, a hidden backdoor is injected into the GCNs model. Experimental results show that our clean graph backdoor can achieve 99% attack success rate while maintaining the functionality of the GCNs model on benign samples.



## **44. MLSD-GAN -- Generating Strong High Quality Face Morphing Attacks using Latent Semantic Disentanglement**

cs.CV

**SubmitDate**: 2024-04-19    [abs](http://arxiv.org/abs/2404.12679v1) [paper-pdf](http://arxiv.org/pdf/2404.12679v1)

**Authors**: Aravinda Reddy PN, Raghavendra Ramachandra, Krothapalli Sreenivasa Rao, Pabitra Mitra

**Abstract**: Face-morphing attacks are a growing concern for biometric researchers, as they can be used to fool face recognition systems (FRS). These attacks can be generated at the image level (supervised) or representation level (unsupervised). Previous unsupervised morphing attacks have relied on generative adversarial networks (GANs). More recently, researchers have used linear interpolation of StyleGAN-encoded images to generate morphing attacks. In this paper, we propose a new method for generating high-quality morphing attacks using StyleGAN disentanglement. Our approach, called MLSD-GAN, spherically interpolates the disentangled latents to produce realistic and diverse morphing attacks. We evaluate the vulnerability of MLSD-GAN on two deep-learning-based FRS techniques. The results show that MLSD-GAN poses a significant threat to FRS, as it can generate morphing attacks that are highly effective at fooling these systems.



## **45. How Real Is Real? A Human Evaluation Framework for Unrestricted Adversarial Examples**

cs.AI

3 pages, 3 figures, AAAI 2024 Spring Symposium on User-Aligned  Assessment of Adaptive AI Systems

**SubmitDate**: 2024-04-19    [abs](http://arxiv.org/abs/2404.12653v1) [paper-pdf](http://arxiv.org/pdf/2404.12653v1)

**Authors**: Dren Fazlija, Arkadij Orlov, Johanna Schrader, Monty-Maximilian Zühlke, Michael Rohs, Daniel Kudenko

**Abstract**: With an ever-increasing reliance on machine learning (ML) models in the real world, adversarial examples threaten the safety of AI-based systems such as autonomous vehicles. In the image domain, they represent maliciously perturbed data points that look benign to humans (i.e., the image modification is not noticeable) but greatly mislead state-of-the-art ML models. Previously, researchers ensured the imperceptibility of their altered data points by restricting perturbations via $\ell_p$ norms. However, recent publications claim that creating natural-looking adversarial examples without such restrictions is also possible. With much more freedom to instill malicious information into data, these unrestricted adversarial examples can potentially overcome traditional defense strategies as they are not constrained by the limitations or patterns these defenses typically recognize and mitigate. This allows attackers to operate outside of expected threat models. However, surveying existing image-based methods, we noticed a need for more human evaluations of the proposed image modifications. Based on existing human-assessment frameworks for image generation quality, we propose SCOOTER - an evaluation framework for unrestricted image-based attacks. It provides researchers with guidelines for conducting statistically significant human experiments, standardized questions, and a ready-to-use implementation. We propose a framework that allows researchers to analyze how imperceptible their unrestricted attacks truly are.



## **46. AED-PADA:Improving Generalizability of Adversarial Example Detection via Principal Adversarial Domain Adaptation**

cs.CV

**SubmitDate**: 2024-04-19    [abs](http://arxiv.org/abs/2404.12635v1) [paper-pdf](http://arxiv.org/pdf/2404.12635v1)

**Authors**: Heqi Peng, Yunhong Wang, Ruijie Yang, Beichen Li, Rui Wang, Yuanfang Guo

**Abstract**: Adversarial example detection, which can be conveniently applied in many scenarios, is important in the area of adversarial defense. Unfortunately, existing detection methods suffer from poor generalization performance, because their training process usually relies on the examples generated from a single known adversarial attack and there exists a large discrepancy between the training and unseen testing adversarial examples. To address this issue, we propose a novel method, named Adversarial Example Detection via Principal Adversarial Domain Adaptation (AED-PADA). Specifically, our approach identifies the Principal Adversarial Domains (PADs), i.e., a combination of features of the adversarial examples from different attacks, which possesses large coverage of the entire adversarial feature space. Then, we pioneer to exploit multi-source domain adaptation in adversarial example detection with PADs as source domains. Experiments demonstrate the superior generalization ability of our proposed AED-PADA. Note that this superiority is particularly achieved in challenging scenarios characterized by employing the minimal magnitude constraint for the perturbations.



## **47. Watermark-embedded Adversarial Examples for Copyright Protection against Diffusion Models**

cs.CV

updated references

**SubmitDate**: 2024-04-19    [abs](http://arxiv.org/abs/2404.09401v2) [paper-pdf](http://arxiv.org/pdf/2404.09401v2)

**Authors**: Peifei Zhu, Tsubasa Takahashi, Hirokatsu Kataoka

**Abstract**: Diffusion Models (DMs) have shown remarkable capabilities in various image-generation tasks. However, there are growing concerns that DMs could be used to imitate unauthorized creations and thus raise copyright issues. To address this issue, we propose a novel framework that embeds personal watermarks in the generation of adversarial examples. Such examples can force DMs to generate images with visible watermarks and prevent DMs from imitating unauthorized images. We construct a generator based on conditional adversarial networks and design three losses (adversarial loss, GAN loss, and perturbation loss) to generate adversarial examples that have subtle perturbation but can effectively attack DMs to prevent copyright violations. Training a generator for a personal watermark by our method only requires 5-10 samples within 2-3 minutes, and once the generator is trained, it can generate adversarial examples with that watermark significantly fast (0.2s per image). We conduct extensive experiments in various conditional image-generation scenarios. Compared to existing methods that generate images with chaotic textures, our method adds visible watermarks on the generated images, which is a more straightforward way to indicate copyright violations. We also observe that our adversarial examples exhibit good transferability across unknown generative models. Therefore, this work provides a simple yet powerful way to protect copyright from DM-based imitation.



## **48. SA-Attack: Speed-adaptive stealthy adversarial attack on trajectory prediction**

cs.LG

This work is published in IEEE IV Symposium

**SubmitDate**: 2024-04-19    [abs](http://arxiv.org/abs/2404.12612v1) [paper-pdf](http://arxiv.org/pdf/2404.12612v1)

**Authors**: Huilin Yin, Jiaxiang Li, Pengju Zhen, Jun Yan

**Abstract**: Trajectory prediction is critical for the safe planning and navigation of automated vehicles. The trajectory prediction models based on the neural networks are vulnerable to adversarial attacks. Previous attack methods have achieved high attack success rates but overlook the adaptability to realistic scenarios and the concealment of the deceits. To address this problem, we propose a speed-adaptive stealthy adversarial attack method named SA-Attack. This method searches the sensitive region of trajectory prediction models and generates the adversarial trajectories by using the vehicle-following method and incorporating information about forthcoming trajectories. Our method has the ability to adapt to different speed scenarios by reconstructing the trajectory from scratch. Fusing future trajectory trends and curvature constraints can guarantee the smoothness of adversarial trajectories, further ensuring the stealthiness of attacks. The empirical study on the datasets of nuScenes and Apolloscape demonstrates the attack performance of our proposed method. Finally, we also demonstrate the adaptability and stealthiness of SA-Attack for different speed scenarios. Our code is available at the repository: https://github.com/eclipse-bot/SA-Attack.



## **49. Proteus: Preserving Model Confidentiality during Graph Optimizations**

cs.CR

**SubmitDate**: 2024-04-18    [abs](http://arxiv.org/abs/2404.12512v1) [paper-pdf](http://arxiv.org/pdf/2404.12512v1)

**Authors**: Yubo Gao, Maryam Haghifam, Christina Giannoula, Renbo Tu, Gennady Pekhimenko, Nandita Vijaykumar

**Abstract**: Deep learning (DL) models have revolutionized numerous domains, yet optimizing them for computational efficiency remains a challenging endeavor. Development of new DL models typically involves two parties: the model developers and performance optimizers. The collaboration between the parties often necessitates the model developers exposing the model architecture and computational graph to the optimizers. However, this exposure is undesirable since the model architecture is an important intellectual property, and its innovations require significant investments and expertise. During the exchange, the model is also vulnerable to adversarial attacks via model stealing.   This paper presents Proteus, a novel mechanism that enables model optimization by an independent party while preserving the confidentiality of the model architecture. Proteus obfuscates the protected model by partitioning its computational graph into subgraphs and concealing each subgraph within a large pool of generated realistic subgraphs that cannot be easily distinguished from the original. We evaluate Proteus on a range of DNNs, demonstrating its efficacy in preserving confidentiality without compromising performance optimization opportunities. Proteus effectively hides the model as one alternative among up to $10^{32}$ possible model architectures, and is resilient against attacks with a learning-based adversary. We also demonstrate that heuristic based and manual approaches are ineffective in identifying the protected model. To our knowledge, Proteus is the first work that tackles the challenge of model confidentiality during performance optimization. Proteus will be open-sourced for direct use and experimentation, with easy integration with compilers such as ONNXRuntime.



## **50. KDk: A Defense Mechanism Against Label Inference Attacks in Vertical Federated Learning**

cs.LG

**SubmitDate**: 2024-04-18    [abs](http://arxiv.org/abs/2404.12369v1) [paper-pdf](http://arxiv.org/pdf/2404.12369v1)

**Authors**: Marco Arazzi, Serena Nicolazzo, Antonino Nocera

**Abstract**: Vertical Federated Learning (VFL) is a category of Federated Learning in which models are trained collaboratively among parties with vertically partitioned data. Typically, in a VFL scenario, the labels of the samples are kept private from all the parties except for the aggregating server, that is the label owner. Nevertheless, recent works discovered that by exploiting gradient information returned by the server to bottom models, with the knowledge of only a small set of auxiliary labels on a very limited subset of training data points, an adversary can infer the private labels. These attacks are known as label inference attacks in VFL. In our work, we propose a novel framework called KDk, that combines Knowledge Distillation and k-anonymity to provide a defense mechanism against potential label inference attacks in a VFL scenario. Through an exhaustive experimental campaign we demonstrate that by applying our approach, the performance of the analyzed label inference attacks decreases consistently, even by more than 60%, maintaining the accuracy of the whole VFL almost unaltered.



