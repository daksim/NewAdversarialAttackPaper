# Latest Adversarial Attack Papers
**update at 2024-02-04 13:14:52**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. ALISON: Fast and Effective Stylometric Authorship Obfuscation**

cs.CL

10 pages, 6 figures, 4 tables. To be published in the Proceedings of  the 38th Annual AAAI Conference on Artificial Intelligence (AAAI-24)

**SubmitDate**: 2024-02-01    [abs](http://arxiv.org/abs/2402.00835v1) [paper-pdf](http://arxiv.org/pdf/2402.00835v1)

**Authors**: Eric Xing, Saranya Venkatraman, Thai Le, Dongwon Lee

**Abstract**: Authorship Attribution (AA) and Authorship Obfuscation (AO) are two competing tasks of increasing importance in privacy research. Modern AA leverages an author's consistent writing style to match a text to its author using an AA classifier. AO is the corresponding adversarial task, aiming to modify a text in such a way that its semantics are preserved, yet an AA model cannot correctly infer its authorship. To address privacy concerns raised by state-of-the-art (SOTA) AA methods, new AO methods have been proposed but remain largely impractical to use due to their prohibitively slow training and obfuscation speed, often taking hours. To this challenge, we propose a practical AO method, ALISON, that (1) dramatically reduces training/obfuscation time, demonstrating more than 10x faster obfuscation than SOTA AO methods, (2) achieves better obfuscation success through attacking three transformer-based AA methods on two benchmark datasets, typically performing 15% better than competing methods, (3) does not require direct signals from a target AA classifier during obfuscation, and (4) utilizes unique stylometric features, allowing sound model interpretation for explainable obfuscation. We also demonstrate that ALISON can effectively prevent four SOTA AA methods from accurately determining the authorship of ChatGPT-generated texts, all while minimally changing the original text semantics. To ensure the reproducibility of our findings, our code and data are available at: https://github.com/EricX003/ALISON.



## **2. Tropical Decision Boundaries for Neural Networks Are Robust Against Adversarial Attacks**

cs.LG

**SubmitDate**: 2024-02-01    [abs](http://arxiv.org/abs/2402.00576v1) [paper-pdf](http://arxiv.org/pdf/2402.00576v1)

**Authors**: Kurt Pasque, Christopher Teska, Ruriko Yoshida, Keiji Miura, Jefferson Huang

**Abstract**: We introduce a simple, easy to implement, and computationally efficient tropical convolutional neural network architecture that is robust against adversarial attacks. We exploit the tropical nature of piece-wise linear neural networks by embedding the data in the tropical projective torus in a single hidden layer which can be added to any model. We study the geometry of its decision boundary theoretically and show its robustness against adversarial attacks on image datasets using computational experiments.



## **3. Short: Benchmarking transferable adversarial attacks**

cs.CV

Accepted by NDSS 2024 Workshop

**SubmitDate**: 2024-02-01    [abs](http://arxiv.org/abs/2402.00418v1) [paper-pdf](http://arxiv.org/pdf/2402.00418v1)

**Authors**: Zhibo Jin, Jiayu Zhang, Zhiyu Zhu, Huaming Chen

**Abstract**: The robustness of deep learning models against adversarial attacks remains a pivotal concern. This study presents, for the first time, an exhaustive review of the transferability aspect of adversarial attacks. It systematically categorizes and critically evaluates various methodologies developed to augment the transferability of adversarial attacks. This study encompasses a spectrum of techniques, including Generative Structure, Semantic Similarity, Gradient Editing, Target Modification, and Ensemble Approach. Concurrently, this paper introduces a benchmark framework \textit{TAA-Bench}, integrating ten leading methodologies for adversarial attack transferability, thereby providing a standardized and systematic platform for comparative analysis across diverse model architectures. Through comprehensive scrutiny, we delineate the efficacy and constraints of each method, shedding light on their underlying operational principles and practical utility. This review endeavors to be a quintessential resource for both scholars and practitioners in the field, charting the complex terrain of adversarial transferability and setting a foundation for future explorations in this vital sector. The associated codebase is accessible at: https://github.com/KxPlaug/TAA-Bench



## **4. Hidding the Ghostwriters: An Adversarial Evaluation of AI-Generated Student Essay Detection**

cs.CL

Accepted by EMNLP 2023 Main conference, Oral Presentation

**SubmitDate**: 2024-02-01    [abs](http://arxiv.org/abs/2402.00412v1) [paper-pdf](http://arxiv.org/pdf/2402.00412v1)

**Authors**: Xinlin Peng, Ying Zhou, Ben He, Le Sun, Yingfei Sun

**Abstract**: Large language models (LLMs) have exhibited remarkable capabilities in text generation tasks. However, the utilization of these models carries inherent risks, including but not limited to plagiarism, the dissemination of fake news, and issues in educational exercises. Although several detectors have been proposed to address these concerns, their effectiveness against adversarial perturbations, specifically in the context of student essay writing, remains largely unexplored. This paper aims to bridge this gap by constructing AIG-ASAP, an AI-generated student essay dataset, employing a range of text perturbation methods that are expected to generate high-quality essays while evading detection. Through empirical experiments, we assess the performance of current AIGC detectors on the AIG-ASAP dataset. The results reveal that the existing detectors can be easily circumvented using straightforward automatic adversarial attacks. Specifically, we explore word substitution and sentence substitution perturbation methods that effectively evade detection while maintaining the quality of the generated essays. This highlights the urgent need for more accurate and robust methods to detect AI-generated student essays in the education domain.



## **5. Comparing Spectral Bias and Robustness For Two-Layer Neural Networks: SGD vs Adaptive Random Fourier Features**

cs.LG

6 Pages, 4 Figures; Accepted in the International Conference on  Scientific Computing and Machine Learning

**SubmitDate**: 2024-02-01    [abs](http://arxiv.org/abs/2402.00332v1) [paper-pdf](http://arxiv.org/pdf/2402.00332v1)

**Authors**: Aku Kammonen, Lisi Liang, Anamika Pandey, Raúl Tempone

**Abstract**: We present experimental results highlighting two key differences resulting from the choice of training algorithm for two-layer neural networks. The spectral bias of neural networks is well known, while the spectral bias dependence on the choice of training algorithm is less studied. Our experiments demonstrate that an adaptive random Fourier features algorithm (ARFF) can yield a spectral bias closer to zero compared to the stochastic gradient descent optimizer (SGD). Additionally, we train two identically structured classifiers, employing SGD and ARFF, to the same accuracy levels and empirically assess their robustness against adversarial noise attacks.



## **6. Invariance-powered Trustworthy Defense via Remove Then Restore**

cs.CV

**SubmitDate**: 2024-02-01    [abs](http://arxiv.org/abs/2402.00304v1) [paper-pdf](http://arxiv.org/pdf/2402.00304v1)

**Authors**: Xiaowei Fu, Yuhang Zhou, Lina Ma, Lei Zhang

**Abstract**: Adversarial attacks pose a challenge to the deployment of deep neural networks (DNNs), while previous defense models overlook the generalization to various attacks. Inspired by targeted therapies for cancer, we view adversarial samples as local lesions of natural benign samples, because a key finding is that salient attack in an adversarial sample dominates the attacking process, while trivial attack unexpectedly provides trustworthy evidence for obtaining generalizable robustness. Based on this finding, a Pixel Surgery and Semantic Regeneration (PSSR) model following the targeted therapy mechanism is developed, which has three merits: 1) To remove the salient attack, a score-based Pixel Surgery module is proposed, which retains the trivial attack as a kind of invariance information. 2) To restore the discriminative content, a Semantic Regeneration module based on a conditional alignment extrapolator is proposed, which achieves pixel and semantic consistency. 3) To further harmonize robustness and accuracy, an intractable problem, a self-augmentation regularizer with adversarial R-drop is designed. Experiments on numerous benchmarks show the superiority of PSSR.



## **7. Adversarial Quantum Machine Learning: An Information-Theoretic Generalization Analysis**

quant-ph

10 pages, 2 figures

**SubmitDate**: 2024-01-31    [abs](http://arxiv.org/abs/2402.00176v1) [paper-pdf](http://arxiv.org/pdf/2402.00176v1)

**Authors**: Petros Georgiou, Sharu Theresa Jose, Osvaldo Simeone

**Abstract**: In a manner analogous to their classical counterparts, quantum classifiers are vulnerable to adversarial attacks that perturb their inputs. A promising countermeasure is to train the quantum classifier by adopting an attack-aware, or adversarial, loss function. This paper studies the generalization properties of quantum classifiers that are adversarially trained against bounded-norm white-box attacks. Specifically, a quantum adversary maximizes the classifier's loss by transforming an input state $\rho(x)$ into a state $\lambda$ that is $\epsilon$-close to the original state $\rho(x)$ in $p$-Schatten distance. Under suitable assumptions on the quantum embedding $\rho(x)$, we derive novel information-theoretic upper bounds on the generalization error of adversarially trained quantum classifiers for $p = 1$ and $p = \infty$. The derived upper bounds consist of two terms: the first is an exponential function of the 2-R\'enyi mutual information between classical data and quantum embedding, while the second term scales linearly with the adversarial perturbation size $\epsilon$. Both terms are shown to decrease as $1/\sqrt{T}$ over the training set size $T$ . An extension is also considered in which the adversary assumed during training has different parameters $p$ and $\epsilon$ as compared to the adversary affecting the test inputs. Finally, we validate our theoretical findings with numerical experiments for a synthetic setting.



## **8. Privacy Risks Analysis and Mitigation in Federated Learning for Medical Images**

cs.LG

V1

**SubmitDate**: 2024-01-31    [abs](http://arxiv.org/abs/2311.06643v2) [paper-pdf](http://arxiv.org/pdf/2311.06643v2)

**Authors**: Badhan Chandra Das, M. Hadi Amini, Yanzhao Wu

**Abstract**: Federated learning (FL) is gaining increasing popularity in the medical domain for analyzing medical images, which is considered an effective technique to safeguard sensitive patient data and comply with privacy regulations. However, several recent studies have revealed that the default settings of FL may leak private training data under privacy attacks. Thus, it is still unclear whether and to what extent such privacy risks of FL exist in the medical domain, and if so, "how to mitigate such risks?". In this paper, first, we propose a holistic framework for Medical data Privacy risk analysis and mitigation in Federated Learning (MedPFL) to analyze privacy risks and develop effective mitigation strategies in FL for protecting private medical data. Second, we demonstrate the substantial privacy risks of using FL to process medical images, where adversaries can easily perform privacy attacks to reconstruct private medical images accurately. Third, we show that the defense approach of adding random noises may not always work effectively to protect medical images against privacy attacks in FL, which poses unique and pressing challenges associated with medical data for privacy protection.



## **9. Elephants Do Not Forget: Differential Privacy with State Continuity for Privacy Budget**

cs.CR

**SubmitDate**: 2024-01-31    [abs](http://arxiv.org/abs/2401.17628v1) [paper-pdf](http://arxiv.org/pdf/2401.17628v1)

**Authors**: Jiankai Jin, Chitchanok Chuengsatiansup, Toby Murray, Benjamin I. P. Rubinstein, Yuval Yarom, Olga Ohrimenko

**Abstract**: Current implementations of differentially-private (DP) systems either lack support to track the global privacy budget consumed on a dataset, or fail to faithfully maintain the state continuity of this budget. We show that failure to maintain a privacy budget enables an adversary to mount replay, rollback and fork attacks - obtaining answers to many more queries than what a secure system would allow. As a result the attacker can reconstruct secret data that DP aims to protect - even if DP code runs in a Trusted Execution Environment (TEE). We propose ElephantDP, a system that aims to provide the same guarantees as a trusted curator in the global DP model would, albeit set in an untrusted environment. Our system relies on a state continuity module to provide protection for the privacy budget and a TEE to faithfully execute DP code and update the budget. To provide security, our protocol makes several design choices including the content of the persistent state and the order between budget updates and query answers. We prove that ElephantDP provides liveness (i.e., the protocol can restart from a correct state and respond to queries as long as the budget is not exceeded) and DP confidentiality (i.e., an attacker learns about a dataset as much as it would from interacting with a trusted curator). Our implementation and evaluation of the protocol use Intel SGX as a TEE to run the DP code and a network of TEEs to maintain state continuity. Compared to an insecure baseline, we observe only 1.1-2$\times$ overheads and lower relative overheads for larger datasets and complex DP queries.



## **10. Game-Theoretic Unlearnable Example Generator**

cs.LG

**SubmitDate**: 2024-01-31    [abs](http://arxiv.org/abs/2401.17523v1) [paper-pdf](http://arxiv.org/pdf/2401.17523v1)

**Authors**: Shuang Liu, Yihan Wang, Xiao-Shan Gao

**Abstract**: Unlearnable example attacks are data poisoning attacks aiming to degrade the clean test accuracy of deep learning by adding imperceptible perturbations to the training samples, which can be formulated as a bi-level optimization problem. However, directly solving this optimization problem is intractable for deep neural networks. In this paper, we investigate unlearnable example attacks from a game-theoretic perspective, by formulating the attack as a nonzero sum Stackelberg game. First, the existence of game equilibria is proved under the normal setting and the adversarial training setting. It is shown that the game equilibrium gives the most powerful poison attack in that the victim has the lowest test accuracy among all networks within the same hypothesis space, when certain loss functions are used. Second, we propose a novel attack method, called the Game Unlearnable Example (GUE), which has three main gradients. (1) The poisons are obtained by directly solving the equilibrium of the Stackelberg game with a first-order algorithm. (2) We employ an autoencoder-like generative network model as the poison attacker. (3) A novel payoff function is introduced to evaluate the performance of the poison. Comprehensive experiments demonstrate that GUE can effectively poison the model in various scenarios. Furthermore, the GUE still works by using a relatively small percentage of the training data to train the generator, and the poison generator can generalize to unseen data well. Our implementation code can be found at https://github.com/hong-xian/gue.



## **11. AdvGPS: Adversarial GPS for Multi-Agent Perception Attack**

cs.CV

Accepted by the 2024 IEEE International Conference on Robotics and  Automation (ICRA)

**SubmitDate**: 2024-01-30    [abs](http://arxiv.org/abs/2401.17499v1) [paper-pdf](http://arxiv.org/pdf/2401.17499v1)

**Authors**: Jinlong Li, Baolu Li, Xinyu Liu, Jianwu Fang, Felix Juefei-Xu, Qing Guo, Hongkai Yu

**Abstract**: The multi-agent perception system collects visual data from sensors located on various agents and leverages their relative poses determined by GPS signals to effectively fuse information, mitigating the limitations of single-agent sensing, such as occlusion. However, the precision of GPS signals can be influenced by a range of factors, including wireless transmission and obstructions like buildings. Given the pivotal role of GPS signals in perception fusion and the potential for various interference, it becomes imperative to investigate whether specific GPS signals can easily mislead the multi-agent perception system. To address this concern, we frame the task as an adversarial attack challenge and introduce \textsc{AdvGPS}, a method capable of generating adversarial GPS signals which are also stealthy for individual agents within the system, significantly reducing object detection accuracy. To enhance the success rates of these attacks in a black-box scenario, we introduce three types of statistically sensitive natural discrepancies: appearance-based discrepancy, distribution-based discrepancy, and task-aware discrepancy. Our extensive experiments on the OPV2V dataset demonstrate that these attacks substantially undermine the performance of state-of-the-art methods, showcasing remarkable transferability across different point cloud based 3D detection systems. This alarming revelation underscores the pressing need to address security implications within multi-agent perception systems, thereby underscoring a critical area of research.



## **12. Camouflage Adversarial Attacks on Multiple Agent Systems**

cs.MA

arXiv admin note: text overlap with arXiv:2311.00859

**SubmitDate**: 2024-01-30    [abs](http://arxiv.org/abs/2401.17405v1) [paper-pdf](http://arxiv.org/pdf/2401.17405v1)

**Authors**: Ziqing Lu, Guanlin Liu, Lifeng Lai, Weiyu Xu

**Abstract**: The multi-agent reinforcement learning systems (MARL) based on the Markov decision process (MDP) have emerged in many critical applications. To improve the robustness/defense of MARL systems against adversarial attacks, the study of various adversarial attacks on reinforcement learning systems is very important. Previous works on adversarial attacks considered some possible features to attack in MDP, such as the action poisoning attacks, the reward poisoning attacks, and the state perception attacks. In this paper, we propose a brand-new form of attack called the camouflage attack in the MARL systems. In the camouflage attack, the attackers change the appearances of some objects without changing the actual objects themselves; and the camouflaged appearances may look the same to all the targeted recipient (victim) agents. The camouflaged appearances can mislead the recipient agents to misguided actions. We design algorithms that give the optimal camouflage attacks minimizing the rewards of recipient agents. Our numerical and theoretical results show that camouflage attacks can rival the more conventional, but likely more difficult state perception attacks. We also investigate cost-constrained camouflage attacks and showed numerically how cost budgets affect the attack performance.



## **13. Robust Prompt Optimization for Defending Language Models Against Jailbreaking Attacks**

cs.LG

code available at https://github.com/andyz245/rpo

**SubmitDate**: 2024-01-30    [abs](http://arxiv.org/abs/2401.17263v1) [paper-pdf](http://arxiv.org/pdf/2401.17263v1)

**Authors**: Andy Zhou, Bo Li, Haohan Wang

**Abstract**: Despite advances in AI alignment, language models (LM) remain vulnerable to adversarial attacks or jailbreaking, in which adversaries modify input prompts to induce harmful behavior. While some defenses have been proposed, they focus on narrow threat models and fall short of a strong defense, which we posit should be effective, universal, and practical. To achieve this, we propose the first adversarial objective for defending LMs against jailbreaking attacks and an algorithm, robust prompt optimization (RPO), that uses gradient-based token optimization to enforce harmless outputs. This results in an easily accessible suffix that significantly improves robustness to both jailbreaks seen during optimization and unknown, held-out jailbreaks, reducing the attack success rate on Starling-7B from 84% to 8.66% across 20 jailbreaks. In addition, we find that RPO has a minor effect on normal LM use, is successful under adaptive attacks, and can transfer to black-box models, reducing the success rate of the strongest attack on GPT-4 from 92% to 6%.



## **14. Weak-to-Strong Jailbreaking on Large Language Models**

cs.CL

**SubmitDate**: 2024-01-30    [abs](http://arxiv.org/abs/2401.17256v1) [paper-pdf](http://arxiv.org/pdf/2401.17256v1)

**Authors**: Xuandong Zhao, Xianjun Yang, Tianyu Pang, Chao Du, Lei Li, Yu-Xiang Wang, William Yang Wang

**Abstract**: Although significant efforts have been dedicated to aligning large language models (LLMs), red-teaming reports suggest that these carefully aligned LLMs could still be jailbroken through adversarial prompts, tuning, or decoding. Upon examining the jailbreaking vulnerability of aligned LLMs, we observe that the decoding distributions of jailbroken and aligned models differ only in the initial generations. This observation motivates us to propose the weak-to-strong jailbreaking attack, where adversaries can utilize smaller unsafe/aligned LLMs (e.g., 7B) to guide jailbreaking against significantly larger aligned LLMs (e.g., 70B). To jailbreak, one only needs to additionally decode two smaller LLMs once, which involves minimal computation and latency compared to decoding the larger LLMs. The efficacy of this attack is demonstrated through experiments conducted on five models from three different organizations. Our study reveals a previously unnoticed yet efficient way of jailbreaking, exposing an urgent safety issue that needs to be considered when aligning LLMs. As an initial attempt, we propose a defense strategy to protect against such attacks, but creating more advanced defenses remains challenging. The code for replicating the method is available at https://github.com/XuandongZhao/weak-to-strong



## **15. Single Word Change is All You Need: Designing Attacks and Defenses for Text Classifiers**

cs.CL

**SubmitDate**: 2024-01-30    [abs](http://arxiv.org/abs/2401.17196v1) [paper-pdf](http://arxiv.org/pdf/2401.17196v1)

**Authors**: Lei Xu, Sarah Alnegheimish, Laure Berti-Equille, Alfredo Cuesta-Infante, Kalyan Veeramachaneni

**Abstract**: In text classification, creating an adversarial example means subtly perturbing a few words in a sentence without changing its meaning, causing it to be misclassified by a classifier. A concerning observation is that a significant portion of adversarial examples generated by existing methods change only one word. This single-word perturbation vulnerability represents a significant weakness in classifiers, which malicious users can exploit to efficiently create a multitude of adversarial examples. This paper studies this problem and makes the following key contributions: (1) We introduce a novel metric \r{ho} to quantitatively assess a classifier's robustness against single-word perturbation. (2) We present the SP-Attack, designed to exploit the single-word perturbation vulnerability, achieving a higher attack success rate, better preserving sentence meaning, while reducing computation costs compared to state-of-the-art adversarial methods. (3) We propose SP-Defense, which aims to improve \r{ho} by applying data augmentation in learning. Experimental results on 4 datasets and BERT and distilBERT classifiers show that SP-Defense improves \r{ho} by 14.6% and 13.9% and decreases the attack success rate of SP-Attack by 30.4% and 21.2% on two classifiers respectively, and decreases the attack success rate of existing attack methods that involve multiple-word perturbations.



## **16. Adversarial Machine Learning in Latent Representations of Neural Networks**

cs.LG

**SubmitDate**: 2024-01-30    [abs](http://arxiv.org/abs/2309.17401v3) [paper-pdf](http://arxiv.org/pdf/2309.17401v3)

**Authors**: Milin Zhang, Mohammad Abdi, Francesco Restuccia

**Abstract**: Distributed deep neural networks (DNNs) have been shown to reduce the computational burden of mobile devices and decrease the end-to-end inference latency in edge computing scenarios. While distributed DNNs have been studied, to the best of our knowledge the resilience of distributed DNNs to adversarial action still remains an open problem. In this paper, we fill the existing research gap by rigorously analyzing the robustness of distributed DNNs against adversarial action. We cast this problem in the context of information theory and introduce two new measurements for distortion and robustness. Our theoretical findings indicate that (i) assuming the same level of information distortion, latent features are always more robust than input representations; (ii) the adversarial robustness is jointly determined by the feature dimension and the generalization capability of the DNN. To test our theoretical findings, we perform extensive experimental analysis by considering 6 different DNN architectures, 6 different approaches for distributed DNN and 10 different adversarial attacks to the ImageNet-1K dataset. Our experimental results support our theoretical findings by showing that the compressed latent representations can reduce the success rate of adversarial attacks by 88% in the best case and by 57% on the average compared to attacks to the input space.



## **17. Systematically Assessing the Security Risks of AI/ML-enabled Connected Healthcare Systems**

cs.CR

13 pages, 5 figures, 3 tables

**SubmitDate**: 2024-01-30    [abs](http://arxiv.org/abs/2401.17136v1) [paper-pdf](http://arxiv.org/pdf/2401.17136v1)

**Authors**: Mohammed Elnawawy, Mohammadreza Hallajiyan, Gargi Mitra, Shahrear Iqbal, Karthik Pattabiraman

**Abstract**: The adoption of machine-learning-enabled systems in the healthcare domain is on the rise. While the use of ML in healthcare has several benefits, it also expands the threat surface of medical systems. We show that the use of ML in medical systems, particularly connected systems that involve interfacing the ML engine with multiple peripheral devices, has security risks that might cause life-threatening damage to a patient's health in case of adversarial interventions. These new risks arise due to security vulnerabilities in the peripheral devices and communication channels. We present a case study where we demonstrate an attack on an ML-enabled blood glucose monitoring system by introducing adversarial data points during inference. We show that an adversary can achieve this by exploiting a known vulnerability in the Bluetooth communication channel connecting the glucose meter with the ML-enabled app. We further show that state-of-the-art risk assessment techniques are not adequate for identifying and assessing these new risks. Our study highlights the need for novel risk analysis methods for analyzing the security of AI-enabled connected health devices.



## **18. What can Information Guess? Guessing Advantage vs. Rényi Entropy for Small Leakages**

cs.IT

**SubmitDate**: 2024-01-30    [abs](http://arxiv.org/abs/2401.17057v1) [paper-pdf](http://arxiv.org/pdf/2401.17057v1)

**Authors**: Julien Béguinot, Olivier Rioul

**Abstract**: We leverage the Gibbs inequality and its natural generalization to R\'enyi entropies to derive closed-form parametric expressions of the optimal lower bounds of $\rho$th-order guessing entropy (guessing moment) of a secret taking values on a finite set, in terms of the R\'enyi-Arimoto $\alpha$-entropy. This is carried out in an non-asymptotic regime when side information may be available. The resulting bounds yield a theoretical solution to a fundamental problem in side-channel analysis: Ensure that an adversary will not gain much guessing advantage when the leakage information is sufficiently weakened by proper countermeasures in a given cryptographic implementation. Practical evaluation for classical leakage models show that the proposed bounds greatly improve previous ones for analyzing the capability of an adversary to perform side-channel attacks.



## **19. Towards Assessing the Synthetic-to-Measured Adversarial Vulnerability of SAR ATR**

cs.CV

**SubmitDate**: 2024-01-30    [abs](http://arxiv.org/abs/2401.17038v1) [paper-pdf](http://arxiv.org/pdf/2401.17038v1)

**Authors**: Bowen Peng, Bo Peng, Jingyuan Xia, Tianpeng Liu, Yongxiang Liu, Li Liu

**Abstract**: Recently, there has been increasing concern about the vulnerability of deep neural network (DNN)-based synthetic aperture radar (SAR) automatic target recognition (ATR) to adversarial attacks, where a DNN could be easily deceived by clean input with imperceptible but aggressive perturbations. This paper studies the synthetic-to-measured (S2M) transfer setting, where an attacker generates adversarial perturbation based solely on synthetic data and transfers it against victim models trained with measured data. Compared with the current measured-to-measured (M2M) transfer setting, our approach does not need direct access to the victim model or the measured SAR data. We also propose the transferability estimation attack (TEA) to uncover the adversarial risks in this more challenging and practical scenario. The TEA makes full use of the limited similarity between the synthetic and measured data pairs for blind estimation and optimization of S2M transferability, leading to feasible surrogate model enhancement without mastering the victim model and data. Comprehensive evaluations based on the publicly available synthetic and measured paired labeled experiment (SAMPLE) dataset demonstrate that the TEA outperforms state-of-the-art methods and can significantly enhance various attack algorithms in computer vision and remote sensing applications. Codes and data are available at https://github.com/scenarri/S2M-TEA.



## **20. Quantum Transfer Learning with Adversarial Robustness for Classification of High-Resolution Image Datasets**

quant-ph

**SubmitDate**: 2024-01-30    [abs](http://arxiv.org/abs/2401.17009v1) [paper-pdf](http://arxiv.org/pdf/2401.17009v1)

**Authors**: Amena Khatun, Muhammad Usman

**Abstract**: The application of quantum machine learning to large-scale high-resolution image datasets is not yet possible due to the limited number of qubits and relatively high level of noise in the current generation of quantum devices. In this work, we address this challenge by proposing a quantum transfer learning (QTL) architecture that integrates quantum variational circuits with a classical machine learning network pre-trained on ImageNet dataset. Through a systematic set of simulations over a variety of image datasets such as Ants & Bees, CIFAR-10, and Road Sign Detection, we demonstrate the superior performance of our QTL approach over classical and quantum machine learning without involving transfer learning. Furthermore, we evaluate the adversarial robustness of QTL architecture with and without adversarial training, confirming that our QTL method is adversarially robust against data manipulation attacks and outperforms classical methods.



## **21. Provably Robust Multi-bit Watermarking for AI-generated Text via Error Correction Code**

cs.CR

**SubmitDate**: 2024-01-30    [abs](http://arxiv.org/abs/2401.16820v1) [paper-pdf](http://arxiv.org/pdf/2401.16820v1)

**Authors**: Wenjie Qu, Dong Yin, Zixin He, Wei Zou, Tianyang Tao, Jinyuan Jia, Jiaheng Zhang

**Abstract**: Large Language Models (LLMs) have been widely deployed for their remarkable capability to generate texts resembling human language. However, they could be misused by criminals to create deceptive content, such as fake news and phishing emails, which raises ethical concerns. Watermarking is a key technique to mitigate the misuse of LLMs, which embeds a watermark (e.g., a bit string) into a text generated by a LLM. Consequently, this enables the detection of texts generated by a LLM as well as the tracing of generated texts to a specific user. The major limitation of existing watermark techniques is that they cannot accurately or efficiently extract the watermark from a text, especially when the watermark is a long bit string. This key limitation impedes their deployment for real-world applications, e.g., tracing generated texts to a specific user.   This work introduces a novel watermarking method for LLM-generated text grounded in \textbf{error-correction codes} to address this challenge. We provide strong theoretical analysis, demonstrating that under bounded adversarial word/token edits (insertion, deletion, and substitution), our method can correctly extract watermarks, offering a provable robustness guarantee. This breakthrough is also evidenced by our extensive experimental results. The experiments show that our method substantially outperforms existing baselines in both accuracy and robustness on benchmark datasets. For instance, when embedding a bit string of length 12 into a 200-token generated text, our approach attains an impressive match rate of $98.4\%$, surpassing the performance of Yoo et al. (state-of-the-art baseline) at $85.6\%$. When subjected to a copy-paste attack involving the injection of 50 tokens to generated texts with 200 words, our method maintains a substantial match rate of $90.8\%$, while the match rate of Yoo et al. diminishes to below $65\%$.



## **22. Machine-learned Adversarial Attacks against Fault Prediction Systems in Smart Electrical Grids**

cs.CR

Accepted in AdvML@KDD'22

**SubmitDate**: 2024-01-30    [abs](http://arxiv.org/abs/2303.18136v2) [paper-pdf](http://arxiv.org/pdf/2303.18136v2)

**Authors**: Carmelo Ardito, Yashar Deldjoo, Tommaso Di Noia, Eugenio Di Sciascio, Fatemeh Nazary, Giovanni Servedio

**Abstract**: In smart electrical grids, fault detection tasks may have a high impact on society due to their economic and critical implications. In the recent years, numerous smart grid applications, such as defect detection and load forecasting, have embraced data-driven methodologies. The purpose of this study is to investigate the challenges associated with the security of machine learning (ML) applications in the smart grid scenario. Indeed, the robustness and security of these data-driven algorithms have not been extensively studied in relation to all power grid applications. We demonstrate first that the deep neural network method used in the smart grid is susceptible to adversarial perturbation. Then, we highlight how studies on fault localization and type classification illustrate the weaknesses of present ML algorithms in smart grids to various adversarial attacks



## **23. GE-AdvGAN: Improving the transferability of adversarial samples by gradient editing-based adversarial generative model**

cs.CV

Accepted by SIAM International Conference on Data Mining (SDM24)

**SubmitDate**: 2024-01-30    [abs](http://arxiv.org/abs/2401.06031v2) [paper-pdf](http://arxiv.org/pdf/2401.06031v2)

**Authors**: Zhiyu Zhu, Huaming Chen, Xinyi Wang, Jiayu Zhang, Zhibo Jin, Kim-Kwang Raymond Choo, Jun Shen, Dong Yuan

**Abstract**: Adversarial generative models, such as Generative Adversarial Networks (GANs), are widely applied for generating various types of data, i.e., images, text, and audio. Accordingly, its promising performance has led to the GAN-based adversarial attack methods in the white-box and black-box attack scenarios. The importance of transferable black-box attacks lies in their ability to be effective across different models and settings, more closely aligning with real-world applications. However, it remains challenging to retain the performance in terms of transferable adversarial examples for such methods. Meanwhile, we observe that some enhanced gradient-based transferable adversarial attack algorithms require prolonged time for adversarial sample generation. Thus, in this work, we propose a novel algorithm named GE-AdvGAN to enhance the transferability of adversarial samples whilst improving the algorithm's efficiency. The main approach is via optimising the training process of the generator parameters. With the functional and characteristic similarity analysis, we introduce a novel gradient editing (GE) mechanism and verify its feasibility in generating transferable samples on various models. Moreover, by exploring the frequency domain information to determine the gradient editing direction, GE-AdvGAN can generate highly transferable adversarial samples while minimizing the execution time in comparison to the state-of-the-art transferable adversarial attack algorithms. The performance of GE-AdvGAN is comprehensively evaluated by large-scale experiments on different datasets, which results demonstrate the superiority of our algorithm. The code for our algorithm is available at: https://github.com/LMBTough/GE-advGAN



## **24. Adversarial Training on Purification (AToP): Advancing Both Robustness and Generalization**

cs.CV

**SubmitDate**: 2024-01-29    [abs](http://arxiv.org/abs/2401.16352v1) [paper-pdf](http://arxiv.org/pdf/2401.16352v1)

**Authors**: Guang Lin, Chao Li, Jianhai Zhang, Toshihisa Tanaka, Qibin Zhao

**Abstract**: The deep neural networks are known to be vulnerable to well-designed adversarial attacks. The most successful defense technique based on adversarial training (AT) can achieve optimal robustness against particular attacks but cannot generalize well to unseen attacks. Another effective defense technique based on adversarial purification (AP) can enhance generalization but cannot achieve optimal robustness. Meanwhile, both methods share one common limitation on the degraded standard accuracy. To mitigate these issues, we propose a novel framework called Adversarial Training on Purification (AToP), which comprises two components: perturbation destruction by random transforms (RT) and purifier model fine-tuned (FT) by adversarial loss. RT is essential to avoid overlearning to known attacks resulting in the robustness generalization to unseen attacks and FT is essential for the improvement of robustness. To evaluate our method in an efficient and scalable way, we conduct extensive experiments on CIFAR-10, CIFAR-100, and ImageNette to demonstrate that our method achieves state-of-the-art results and exhibits generalization ability against unseen attacks.



## **25. Tradeoffs Between Alignment and Helpfulness in Language Models**

cs.CL

**SubmitDate**: 2024-01-29    [abs](http://arxiv.org/abs/2401.16332v1) [paper-pdf](http://arxiv.org/pdf/2401.16332v1)

**Authors**: Yotam Wolf, Noam Wies, Dorin Shteyman, Binyamin Rothberg, Yoav Levine, Amnon Shashua

**Abstract**: Language model alignment has become an important component of AI safety, allowing safe interactions between humans and language models, by enhancing desired behaviors and inhibiting undesired ones. It is often done by tuning the model or inserting preset aligning prompts. Recently, representation engineering, a method which alters the model's behavior via changing its representations post-training, was shown to be effective in aligning LLMs (Zou et al., 2023a). Representation engineering yields gains in alignment oriented tasks such as resistance to adversarial attacks and reduction of social biases, but was also shown to cause a decrease in the ability of the model to perform basic tasks. In this paper we study the tradeoff between the increase in alignment and decrease in helpfulness of the model. We propose a theoretical framework which provides bounds for these two quantities, and demonstrate their relevance empirically. Interestingly, we find that while the helpfulness generally decreases, it does so quadratically with the norm of the representation engineering vector, while the alignment increases linearly with it, indicating a regime in which it is efficient to use representation engineering. We validate our findings empirically, and chart the boundaries to the usefulness of representation engineering for alignment.



## **26. Understanding Adversarial Robustness from Feature Maps of Convolutional Layers**

cs.CV

14pages

**SubmitDate**: 2024-01-29    [abs](http://arxiv.org/abs/2202.12435v2) [paper-pdf](http://arxiv.org/pdf/2202.12435v2)

**Authors**: Cong Xu, Wei Zhang, Jun Wang, Min Yang

**Abstract**: The adversarial robustness of a neural network mainly relies on two factors: model capacity and anti-perturbation ability. In this paper, we study the anti-perturbation ability of the network from the feature maps of convolutional layers. Our theoretical analysis discovers that larger convolutional feature maps before average pooling can contribute to better resistance to perturbations, but the conclusion is not true for max pooling. It brings new inspiration to the design of robust neural networks and urges us to apply these findings to improve existing architectures. The proposed modifications are very simple and only require upsampling the inputs or slightly modifying the stride configurations of downsampling operators. We verify our approaches on several benchmark neural network architectures, including AlexNet, VGG, RestNet18, and PreActResNet18. Non-trivial improvements in terms of both natural accuracy and adversarial robustness can be achieved under various attack and defense mechanisms. The code is available at \url{https://github.com/MTandHJ/rcm}.



## **27. LESSON: Multi-Label Adversarial False Data Injection Attack for Deep Learning Locational Detection**

cs.CR

Accepted by TDSC

**SubmitDate**: 2024-01-29    [abs](http://arxiv.org/abs/2401.16001v1) [paper-pdf](http://arxiv.org/pdf/2401.16001v1)

**Authors**: Jiwei Tian, Chao Shen, Buhong Wang, Xiaofang Xia, Meng Zhang, Chenhao Lin, Qian Li

**Abstract**: Deep learning methods can not only detect false data injection attacks (FDIA) but also locate attacks of FDIA. Although adversarial false data injection attacks (AFDIA) based on deep learning vulnerabilities have been studied in the field of single-label FDIA detection, the adversarial attack and defense against multi-label FDIA locational detection are still not involved. To bridge this gap, this paper first explores the multi-label adversarial example attacks against multi-label FDIA locational detectors and proposes a general multi-label adversarial attack framework, namely muLti-labEl adverSarial falSe data injectiON attack (LESSON). The proposed LESSON attack framework includes three key designs, namely Perturbing State Variables, Tailored Loss Function Design, and Change of Variables, which can help find suitable multi-label adversarial perturbations within the physical constraints to circumvent both Bad Data Detection (BDD) and Neural Attack Location (NAL). Four typical LESSON attacks based on the proposed framework and two dimensions of attack objectives are examined, and the experimental results demonstrate the effectiveness of the proposed attack framework, posing serious and pressing security concerns in smart grids.



## **28. A Two-Layer Blockchain Sharding Protocol Leveraging Safety and Liveness for Enhanced Performance**

cs.CR

The paper has been accepted to Network and Distributed System  Security (NDSS) Symposium 2024

**SubmitDate**: 2024-01-29    [abs](http://arxiv.org/abs/2310.11373v2) [paper-pdf](http://arxiv.org/pdf/2310.11373v2)

**Authors**: Yibin Xu, Jingyi Zheng, Boris Düdder, Tijs Slaats, Yongluan Zhou

**Abstract**: Sharding is essential for improving blockchain scalability. Existing protocols overlook diverse adversarial attacks, limiting transaction throughput. This paper presents Reticulum, a groundbreaking sharding protocol addressing this issue, boosting blockchain scalability.   Reticulum employs a two-phase approach, adapting transaction throughput based on runtime adversarial attacks. It comprises "control" and "process" shards in two layers. Process shards contain at least one trustworthy node, while control shards have a majority of trusted nodes. In the first phase, transactions are written to blocks and voted on by nodes in process shards. Unanimously accepted blocks are confirmed. In the second phase, blocks without unanimous acceptance are voted on by control shards. Blocks are accepted if the majority votes in favor, eliminating first-phase opponents and silent voters. Reticulum uses unanimous voting in the first phase, involving fewer nodes, enabling more parallel process shards. Control shards finalize decisions and resolve disputes.   Experiments confirm Reticulum's innovative design, providing high transaction throughput and robustness against various network attacks, outperforming existing sharding protocols for blockchain networks.



## **29. Mitigation of Channel Tampering Attacks in Continuous-Variable Quantum Key Distribution**

quant-ph

10 pages, 5 figures

**SubmitDate**: 2024-01-29    [abs](http://arxiv.org/abs/2401.15898v1) [paper-pdf](http://arxiv.org/pdf/2401.15898v1)

**Authors**: Sebastian P. Kish, Chandra Thapa, Mikhael Sayat, Hajime Suzuki, Josef Pieprzyk, Seyit Camtepe

**Abstract**: Despite significant advancements in continuous-variable quantum key distribution (CV-QKD), practical CV-QKD systems can be compromised by various attacks. Consequently, identifying new attack vectors and countermeasures for CV-QKD implementations is important for the continued robustness of CV-QKD. In particular, as CV-QKD relies on a public quantum channel, vulnerability to communication disruption persists from potential adversaries employing Denial-of-Service (DoS) attacks. Inspired by DoS attacks, this paper introduces a novel threat in CV-QKD called the Channel Amplification (CA) attack, wherein Eve manipulates the communication channel through amplification. We specifically model this attack in a CV-QKD optical fiber setup. To counter this threat, we propose a detection and mitigation strategy. Detection involves a machine learning (ML) model based on a decision tree classifier, classifying various channel tampering attacks, including CA and DoS attacks. For mitigation, Bob, post-selects quadrature data by classifying the attack type and frequency. Our ML model exhibits high accuracy in distinguishing and categorizing these attacks. The CA attack's impact on the secret key rate (SKR) is explored concerning Eve's location and the relative intensity noise of the local oscillator (LO). The proposed mitigation strategy improves the attacked SKR for CA attacks and, in some cases, for hybrid CA-DoS attacks. Our study marks a novel application of both ML classification and post-selection in this context. These findings are important for enhancing the robustness of CV-QKD systems against emerging threats on the channel.



## **30. TransTroj: Transferable Backdoor Attacks to Pre-trained Models via Embedding Indistinguishability**

cs.CR

13 pages, 16 figures, 5 tables

**SubmitDate**: 2024-01-29    [abs](http://arxiv.org/abs/2401.15883v1) [paper-pdf](http://arxiv.org/pdf/2401.15883v1)

**Authors**: Hao Wang, Tao Xiang, Shangwei Guo, Jialing He, Hangcheng Liu, Tianwei Zhang

**Abstract**: Pre-trained models (PTMs) are extensively utilized in various downstream tasks. Adopting untrusted PTMs may suffer from backdoor attacks, where the adversary can compromise the downstream models by injecting backdoors into the PTM. However, existing backdoor attacks to PTMs can only achieve partially task-agnostic and the embedded backdoors are easily erased during the fine-tuning process. In this paper, we propose a novel transferable backdoor attack, TransTroj, to simultaneously meet functionality-preserving, durable, and task-agnostic. In particular, we first formalize transferable backdoor attacks as the indistinguishability problem between poisoned and clean samples in the embedding space. We decompose the embedding indistinguishability into pre- and post-indistinguishability, representing the similarity of the poisoned and reference embeddings before and after the attack. Then, we propose a two-stage optimization that separately optimizes triggers and victim PTMs to achieve embedding indistinguishability. We evaluate TransTroj on four PTMs and six downstream tasks. Experimental results show that TransTroj significantly outperforms SOTA task-agnostic backdoor attacks (18%$\sim$99%, 68% on average) and exhibits superior performance under various system settings. The code is available at https://github.com/haowang-cqu/TransTroj .



## **31. Adversarial Attacks and Defenses in 6G Network-Assisted IoT Systems**

cs.IT

17 pages, 5 figures, and 4 tables. Submitted for publications

**SubmitDate**: 2024-01-29    [abs](http://arxiv.org/abs/2401.14780v2) [paper-pdf](http://arxiv.org/pdf/2401.14780v2)

**Authors**: Bui Duc Son, Nguyen Tien Hoa, Trinh Van Chien, Waqas Khalid, Mohamed Amine Ferrag, Wan Choi, Merouane Debbah

**Abstract**: The Internet of Things (IoT) and massive IoT systems are key to sixth-generation (6G) networks due to dense connectivity, ultra-reliability, low latency, and high throughput. Artificial intelligence, including deep learning and machine learning, offers solutions for optimizing and deploying cutting-edge technologies for future radio communications. However, these techniques are vulnerable to adversarial attacks, leading to degraded performance and erroneous predictions, outcomes unacceptable for ubiquitous networks. This survey extensively addresses adversarial attacks and defense methods in 6G network-assisted IoT systems. The theoretical background and up-to-date research on adversarial attacks and defenses are discussed. Furthermore, we provide Monte Carlo simulations to validate the effectiveness of adversarial attacks compared to jamming attacks. Additionally, we examine the vulnerability of 6G IoT systems by demonstrating attack strategies applicable to key technologies, including reconfigurable intelligent surfaces, massive multiple-input multiple-output (MIMO)/cell-free massive MIMO, satellites, the metaverse, and semantic communications. Finally, we outline the challenges and future developments associated with adversarial attacks and defenses in 6G IoT systems.



## **32. Transparency Attacks: How Imperceptible Image Layers Can Fool AI Perception**

cs.CV

**SubmitDate**: 2024-01-29    [abs](http://arxiv.org/abs/2401.15817v1) [paper-pdf](http://arxiv.org/pdf/2401.15817v1)

**Authors**: Forrest McKee, David Noever

**Abstract**: This paper investigates a novel algorithmic vulnerability when imperceptible image layers confound multiple vision models into arbitrary label assignments and captions. We explore image preprocessing methods to introduce stealth transparency, which triggers AI misinterpretation of what the human eye perceives. The research compiles a broad attack surface to investigate the consequences ranging from traditional watermarking, steganography, and background-foreground miscues. We demonstrate dataset poisoning using the attack to mislabel a collection of grayscale landscapes and logos using either a single attack layer or randomly selected poisoning classes. For example, a military tank to the human eye is a mislabeled bridge to object classifiers based on convolutional networks (YOLO, etc.) and vision transformers (ViT, GPT-Vision, etc.). A notable attack limitation stems from its dependency on the background (hidden) layer in grayscale as a rough match to the transparent foreground image that the human eye perceives. This dependency limits the practical success rate without manual tuning and exposes the hidden layers when placed on the opposite display theme (e.g., light background, light transparent foreground visible, works best against a light theme image viewer or browser). The stealth transparency confounds established vision systems, including evading facial recognition and surveillance, digital watermarking, content filtering, dataset curating, automotive and drone autonomy, forensic evidence tampering, and retail product misclassifying. This method stands in contrast to traditional adversarial attacks that typically focus on modifying pixel values in ways that are either slightly perceptible or entirely imperceptible for both humans and machines.



## **33. Improving Transformation-based Defenses against Adversarial Examples with First-order Perturbations**

cs.CV

This paper has technical errors

**SubmitDate**: 2024-01-28    [abs](http://arxiv.org/abs/2103.04565v3) [paper-pdf](http://arxiv.org/pdf/2103.04565v3)

**Authors**: Haimin Zhang, Min Xu

**Abstract**: Deep neural networks have been successfully applied in various machine learning tasks. However, studies show that neural networks are susceptible to adversarial attacks. This exposes a potential threat to neural network-based intelligent systems. We observe that the probability of the correct result outputted by the neural network increases by applying small first-order perturbations generated for non-predicted class labels to adversarial examples. Based on this observation, we propose a method for counteracting adversarial perturbations to improve adversarial robustness. In the proposed method, we randomly select a number of class labels and generate small first-order perturbations for these selected labels. The generated perturbations are added together and then clamped onto a specified space. The obtained perturbation is finally added to the adversarial example to counteract the adversarial perturbation contained in the example. The proposed method is applied at inference time and does not require retraining or finetuning the model. We experimentally validate the proposed method on CIFAR-10 and CIFAR-100. The results demonstrate that our method effectively improves the defense performance of several transformation-based defense methods, especially against strong adversarial examples generated using more iterations.



## **34. Adversarial Attacks on Graph Neural Networks via Meta Learning**

cs.LG

ICLR submission

**SubmitDate**: 2024-01-28    [abs](http://arxiv.org/abs/1902.08412v2) [paper-pdf](http://arxiv.org/pdf/1902.08412v2)

**Authors**: Daniel Zügner, Stephan Günnemann

**Abstract**: Deep learning models for graphs have advanced the state of the art on many tasks. Despite their recent success, little is known about their robustness. We investigate training time attacks on graph neural networks for node classification that perturb the discrete graph structure. Our core principle is to use meta-gradients to solve the bilevel problem underlying training-time attacks, essentially treating the graph as a hyperparameter to optimize. Our experiments show that small graph perturbations consistently lead to a strong decrease in performance for graph convolutional networks, and even transfer to unsupervised embeddings. Remarkably, the perturbations created by our algorithm can misguide the graph neural networks such that they perform worse than a simple baseline that ignores all relational information. Our attacks do not assume any knowledge about or access to the target classifiers.



## **35. Passive Inference Attacks on Split Learning via Adversarial Regularization**

cs.CR

19 pages, 20 figures

**SubmitDate**: 2024-01-28    [abs](http://arxiv.org/abs/2310.10483v4) [paper-pdf](http://arxiv.org/pdf/2310.10483v4)

**Authors**: Xiaochen Zhu, Xinjian Luo, Yuncheng Wu, Yangfan Jiang, Xiaokui Xiao, Beng Chin Ooi

**Abstract**: Split Learning (SL) has emerged as a practical and efficient alternative to traditional federated learning. While previous attempts to attack SL have often relied on overly strong assumptions or targeted easily exploitable models, we seek to develop more practical attacks. We introduce SDAR, a novel attack framework against SL with an honest-but-curious server. SDAR leverages auxiliary data and adversarial regularization to learn a decodable simulator of the client's private model, which can effectively infer the client's private features under the vanilla SL, and both features and labels under the U-shaped SL. We perform extensive experiments in both configurations to validate the effectiveness of our proposed attacks. Notably, in challenging but practical scenarios where existing passive attacks struggle to reconstruct the client's private data effectively, SDAR consistently achieves attack performance comparable to active attacks. On CIFAR-10, at the deep split level of 7, SDAR achieves private feature reconstruction with less than 0.025 mean squared error in both the vanilla and the U-shaped SL, and attains a label inference accuracy of over 98% in the U-shaped setting, while existing attacks fail to produce non-trivial results.



## **36. Masked Language Model Based Textual Adversarial Example Detection**

cs.CR

13 pages,3 figures

**SubmitDate**: 2024-01-28    [abs](http://arxiv.org/abs/2304.08767v3) [paper-pdf](http://arxiv.org/pdf/2304.08767v3)

**Authors**: Xiaomei Zhang, Zhaoxi Zhang, Qi Zhong, Xufei Zheng, Yanjun Zhang, Shengshan Hu, Leo Yu Zhang

**Abstract**: Adversarial attacks are a serious threat to the reliable deployment of machine learning models in safety-critical applications. They can misguide current models to predict incorrectly by slightly modifying the inputs. Recently, substantial work has shown that adversarial examples tend to deviate from the underlying data manifold of normal examples, whereas pre-trained masked language models can fit the manifold of normal NLP data. To explore how to use the masked language model in adversarial detection, we propose a novel textual adversarial example detection method, namely Masked Language Model-based Detection (MLMD), which can produce clearly distinguishable signals between normal examples and adversarial examples by exploring the changes in manifolds induced by the masked language model. MLMD features a plug and play usage (i.e., no need to retrain the victim model) for adversarial defense and it is agnostic to classification tasks, victim model's architectures, and to-be-defended attack methods. We evaluate MLMD on various benchmark textual datasets, widely studied machine learning models, and state-of-the-art (SOTA) adversarial attacks (in total $3*4*4 = 48$ settings). Experimental results show that MLMD can achieve strong performance, with detection accuracy up to 0.984, 0.967, and 0.901 on AG-NEWS, IMDB, and SST-2 datasets, respectively. Additionally, MLMD is superior, or at least comparable to, the SOTA detection defenses in detection accuracy and F1 score. Among many defenses based on the off-manifold assumption of adversarial examples, this work offers a new angle for capturing the manifold change. The code for this work is openly accessible at \url{https://github.com/mlmddetection/MLMDdetection}.



## **37. Addressing Noise and Efficiency Issues in Graph-Based Machine Learning Models From the Perspective of Adversarial Attack**

cs.LG

**SubmitDate**: 2024-01-28    [abs](http://arxiv.org/abs/2401.15615v1) [paper-pdf](http://arxiv.org/pdf/2401.15615v1)

**Authors**: Yongyu Wang

**Abstract**: Given that no existing graph construction method can generate a perfect graph for a given dataset, graph-based algorithms are invariably affected by the plethora of redundant and erroneous edges present within the constructed graphs. In this paper, we propose treating these noisy edges as adversarial attack and use a spectral adversarial robustness evaluation method to diminish the impact of noisy edges on the performance of graph algorithms. Our method identifies those points that are less vulnerable to noisy edges and leverages only these robust points to perform graph-based algorithms. Our experiments with spectral clustering, one of the most representative and widely utilized graph algorithms, reveal that our methodology not only substantially elevates the precision of the algorithm but also greatly accelerates its computational efficiency by leveraging only a select number of robust data points.



## **38. Generalizing Speaker Verification for Spoof Awareness in the Embedding Space**

cs.CR

Published in IEEE/ACM Transactions on Audio, Speech, and Language  Processing (doi updated)

**SubmitDate**: 2024-01-28    [abs](http://arxiv.org/abs/2401.11156v2) [paper-pdf](http://arxiv.org/pdf/2401.11156v2)

**Authors**: Xuechen Liu, Md Sahidullah, Kong Aik Lee, Tomi Kinnunen

**Abstract**: It is now well-known that automatic speaker verification (ASV) systems can be spoofed using various types of adversaries. The usual approach to counteract ASV systems against such attacks is to develop a separate spoofing countermeasure (CM) module to classify speech input either as a bonafide, or a spoofed utterance. Nevertheless, such a design requires additional computation and utilization efforts at the authentication stage. An alternative strategy involves a single monolithic ASV system designed to handle both zero-effort imposter (non-targets) and spoofing attacks. Such spoof-aware ASV systems have the potential to provide stronger protections and more economic computations. To this end, we propose to generalize the standalone ASV (G-SASV) against spoofing attacks, where we leverage limited training data from CM to enhance a simple backend in the embedding space, without the involvement of a separate CM module during the test (authentication) phase. We propose a novel yet simple backend classifier based on deep neural networks and conduct the study via domain adaptation and multi-task integration of spoof embeddings at the training stage. Experiments are conducted on the ASVspoof 2019 logical access dataset, where we improve the performance of statistical ASV backends on the joint (bonafide and spoofed) and spoofed conditions by a maximum of 36.2% and 49.8% in terms of equal error rates, respectively.



## **39. Style-News: Incorporating Stylized News Generation and Adversarial Verification for Neural Fake News Detection**

cs.CL

EACL 2024 Main Track

**SubmitDate**: 2024-01-27    [abs](http://arxiv.org/abs/2401.15509v1) [paper-pdf](http://arxiv.org/pdf/2401.15509v1)

**Authors**: Wei-Yao Wang, Yu-Chieh Chang, Wen-Chih Peng

**Abstract**: With the improvements in generative models, the issues of producing hallucinations in various domains (e.g., law, writing) have been brought to people's attention due to concerns about misinformation. In this paper, we focus on neural fake news, which refers to content generated by neural networks aiming to mimic the style of real news to deceive people. To prevent harmful disinformation spreading fallaciously from malicious social media (e.g., content farms), we propose a novel verification framework, Style-News, using publisher metadata to imply a publisher's template with the corresponding text types, political stance, and credibility. Based on threat modeling aspects, a style-aware neural news generator is introduced as an adversary for generating news content conditioning for a specific publisher, and style and source discriminators are trained to defend against this attack by identifying which publisher the style corresponds with, and discriminating whether the source of the given news is human-written or machine-generated. To evaluate the quality of the generated content, we integrate various dimensional metrics (language fluency, content preservation, and style adherence) and demonstrate that Style-News significantly outperforms the previous approaches by a margin of 0.35 for fluency, 15.24 for content, and 0.38 for style at most. Moreover, our discriminative model outperforms state-of-the-art baselines in terms of publisher prediction (up to 4.64%) and neural fake news detection (+6.94% $\sim$ 31.72%).



## **40. No-Box Attacks on 3D Point Cloud Classification**

cs.CV

10 pages, 6 figures

**SubmitDate**: 2024-01-27    [abs](http://arxiv.org/abs/2210.14164v3) [paper-pdf](http://arxiv.org/pdf/2210.14164v3)

**Authors**: Hanieh Naderi, Chinthaka Dinesh, Ivan V. Bajic, Shohreh Kasaei

**Abstract**: Adversarial attacks pose serious challenges for deep neural network (DNN)-based analysis of various input signals. In the case of 3D point clouds, methods have been developed to identify points that play a key role in network decision, and these become crucial in generating existing adversarial attacks. For example, a saliency map approach is a popular method for identifying adversarial drop points, whose removal would significantly impact the network decision. Generally, methods for identifying adversarial points rely on the access to the DNN model itself to determine which points are critically important for the model's decision. This paper aims to provide a novel viewpoint on this problem, where adversarial points can be predicted without access to the target DNN model, which is referred to as a ``no-box'' attack. To this end, we define 14 point cloud features and use multiple linear regression to examine whether these features can be used for adversarial point prediction, and which combination of features is best suited for this purpose. Experiments show that a suitable combination of features is able to predict adversarial points of four different networks -- PointNet, PointNet++, DGCNN, and PointConv -- significantly better than a random guess and comparable to white-box attacks. Additionally, we show that no-box attack is transferable to unseen models. The results also provide further insight into DNNs for point cloud classification, by showing which features play key roles in their decision-making process.



## **41. L-AutoDA: Leveraging Large Language Models for Automated Decision-based Adversarial Attacks**

cs.CR

Under Review of IJCNN 2024

**SubmitDate**: 2024-01-27    [abs](http://arxiv.org/abs/2401.15335v1) [paper-pdf](http://arxiv.org/pdf/2401.15335v1)

**Authors**: Ping Guo, Fei Liu, Xi Lin, Qingchuan Zhao, Qingfu Zhang

**Abstract**: In the rapidly evolving field of machine learning, adversarial attacks present a significant challenge to model robustness and security. Decision-based attacks, which only require feedback on the decision of a model rather than detailed probabilities or scores, are particularly insidious and difficult to defend against. This work introduces L-AutoDA (Large Language Model-based Automated Decision-based Adversarial Attacks), a novel approach leveraging the generative capabilities of Large Language Models (LLMs) to automate the design of these attacks. By iteratively interacting with LLMs in an evolutionary framework, L-AutoDA automatically designs competitive attack algorithms efficiently without much human effort. We demonstrate the efficacy of L-AutoDA on CIFAR-10 dataset, showing significant improvements over baseline methods in both success rate and computational efficiency. Our findings underscore the potential of language models as tools for adversarial attack generation and highlight new avenues for the development of robust AI systems.



## **42. Multi-Trigger Backdoor Attacks: More Triggers, More Threats**

cs.LG

**SubmitDate**: 2024-01-27    [abs](http://arxiv.org/abs/2401.15295v1) [paper-pdf](http://arxiv.org/pdf/2401.15295v1)

**Authors**: Yige Li, Xingjun Ma, Jiabo He, Hanxun Huang, Yu-Gang Jiang

**Abstract**: Backdoor attacks have emerged as a primary threat to (pre-)training and deployment of deep neural networks (DNNs). While backdoor attacks have been extensively studied in a body of works, most of them were focused on single-trigger attacks that poison a dataset using a single type of trigger. Arguably, real-world backdoor attacks can be much more complex, e.g., the existence of multiple adversaries for the same dataset if it is of high value. In this work, we investigate the practical threat of backdoor attacks under the setting of \textbf{multi-trigger attacks} where multiple adversaries leverage different types of triggers to poison the same dataset. By proposing and investigating three types of multi-trigger attacks, including parallel, sequential, and hybrid attacks, we provide a set of important understandings of the coexisting, overwriting, and cross-activating effects between different triggers on the same dataset. Moreover, we show that single-trigger attacks tend to cause overly optimistic views of the security of current defense techniques, as all examined defense methods struggle to defend against multi-trigger attacks. Finally, we create a multi-trigger backdoor poisoning dataset to help future evaluation of backdoor attacks and defenses. Although our work is purely empirical, we hope it can help steer backdoor research toward more realistic settings.



## **43. Unraveling Attacks in Machine Learning-based IoT Ecosystems: A Survey and the Open Libraries Behind Them**

cs.CR

**SubmitDate**: 2024-01-27    [abs](http://arxiv.org/abs/2401.11723v2) [paper-pdf](http://arxiv.org/pdf/2401.11723v2)

**Authors**: Chao Liu, Boxi Chen, Wei Shao, Chris Zhang, Kelvin Wong, Yi Zhang

**Abstract**: The advent of the Internet of Things (IoT) has brought forth an era of unprecedented connectivity, with an estimated 80 billion smart devices expected to be in operation by the end of 2025. These devices facilitate a multitude of smart applications, enhancing the quality of life and efficiency across various domains. Machine Learning (ML) serves as a crucial technology, not only for analyzing IoT-generated data but also for diverse applications within the IoT ecosystem. For instance, ML finds utility in IoT device recognition, anomaly detection, and even in uncovering malicious activities. This paper embarks on a comprehensive exploration of the security threats arising from ML's integration into various facets of IoT, spanning various attack types including membership inference, adversarial evasion, reconstruction, property inference, model extraction, and poisoning attacks. Unlike previous studies, our work offers a holistic perspective, categorizing threats based on criteria such as adversary models, attack targets, and key security attributes (confidentiality, availability, and integrity). We delve into the underlying techniques of ML attacks in IoT environment, providing a critical evaluation of their mechanisms and impacts. Furthermore, our research thoroughly assesses 65 libraries, both author-contributed and third-party, evaluating their role in safeguarding model and data privacy. We emphasize the availability and usability of these libraries, aiming to arm the community with the necessary tools to bolster their defenses against the evolving threat landscape. Through our comprehensive review and analysis, this paper seeks to contribute to the ongoing discourse on ML-based IoT security, offering valuable insights and practical solutions to secure ML models and data in the rapidly expanding field of artificial intelligence in IoT.



## **44. Asymptotic Behavior of Adversarial Training Estimator under $\ell_\infty$-Perturbation**

math.ST

**SubmitDate**: 2024-01-27    [abs](http://arxiv.org/abs/2401.15262v1) [paper-pdf](http://arxiv.org/pdf/2401.15262v1)

**Authors**: Yiling Xie, Xiaoming Huo

**Abstract**: Adversarial training has been proposed to hedge against adversarial attacks in machine learning and statistical models. This paper focuses on adversarial training under $\ell_\infty$-perturbation, which has recently attracted much research attention. The asymptotic behavior of the adversarial training estimator is investigated in the generalized linear model. The results imply that the limiting distribution of the adversarial training estimator under $\ell_\infty$-perturbation could put a positive probability mass at $0$ when the true parameter is $0$, providing a theoretical guarantee of the associated sparsity-recovery ability. Alternatively, a two-step procedure is proposed -- adaptive adversarial training, which could further improve the performance of adversarial training under $\ell_\infty$-perturbation. Specifically, the proposed procedure could achieve asymptotic unbiasedness and variable-selection consistency. Numerical experiments are conducted to show the sparsity-recovery ability of adversarial training under $\ell_\infty$-perturbation and to compare the empirical performance between classic adversarial training and adaptive adversarial training.



## **45. Better Representations via Adversarial Training in Pre-Training: A Theoretical Perspective**

cs.LG

To appear in AISTATS2024

**SubmitDate**: 2024-01-26    [abs](http://arxiv.org/abs/2401.15248v1) [paper-pdf](http://arxiv.org/pdf/2401.15248v1)

**Authors**: Yue Xing, Xiaofeng Lin, Qifan Song, Yi Xu, Belinda Zeng, Guang Cheng

**Abstract**: Pre-training is known to generate universal representations for downstream tasks in large-scale deep learning such as large language models. Existing literature, e.g., \cite{kim2020adversarial}, empirically observe that the downstream tasks can inherit the adversarial robustness of the pre-trained model. We provide theoretical justifications for this robustness inheritance phenomenon. Our theoretical results reveal that feature purification plays an important role in connecting the adversarial robustness of the pre-trained model and the downstream tasks in two-layer neural networks. Specifically, we show that (i) with adversarial training, each hidden node tends to pick only one (or a few) feature; (ii) without adversarial training, the hidden nodes can be vulnerable to attacks. This observation is valid for both supervised pre-training and contrastive learning. With purified nodes, it turns out that clean training is enough to achieve adversarial robustness in downstream tasks.



## **46. End-To-End Set-Based Training for Neural Network Verification**

cs.LG

**SubmitDate**: 2024-01-26    [abs](http://arxiv.org/abs/2401.14961v1) [paper-pdf](http://arxiv.org/pdf/2401.14961v1)

**Authors**: Lukas Koller, Tobias Ladner, Matthias Althoff

**Abstract**: Neural networks are vulnerable to adversarial attacks, i.e., small input perturbations can result in substantially different outputs of a neural network. Safety-critical environments require neural networks that are robust against input perturbations. However, training and formally verifying robust neural networks is challenging. We address this challenge by employing, for the first time, a end-to-end set-based training procedure that trains robust neural networks for formal verification. Our training procedure drastically simplifies the subsequent formal robustness verification of the trained neural network. While previous research has predominantly focused on augmenting neural network training with adversarial attacks, our approach leverages set-based computing to train neural networks with entire sets of perturbed inputs. Moreover, we demonstrate that our set-based training procedure effectively trains robust neural networks, which are easier to verify. In many cases, set-based trained neural networks outperform neural networks trained with state-of-the-art adversarial attacks.



## **47. Conserve-Update-Revise to Cure Generalization and Robustness Trade-off in Adversarial Training**

cs.LG

Accepted as a conference paper at ICLR 2024

**SubmitDate**: 2024-01-26    [abs](http://arxiv.org/abs/2401.14948v1) [paper-pdf](http://arxiv.org/pdf/2401.14948v1)

**Authors**: Shruthi Gowda, Bahram Zonooz, Elahe Arani

**Abstract**: Adversarial training improves the robustness of neural networks against adversarial attacks, albeit at the expense of the trade-off between standard and robust generalization. To unveil the underlying factors driving this phenomenon, we examine the layer-wise learning capabilities of neural networks during the transition from a standard to an adversarial setting. Our empirical findings demonstrate that selectively updating specific layers while preserving others can substantially enhance the network's learning capacity. We therefore propose CURE, a novel training framework that leverages a gradient prominence criterion to perform selective conservation, updating, and revision of weights. Importantly, CURE is designed to be dataset- and architecture-agnostic, ensuring its applicability across various scenarios. It effectively tackles both memorization and overfitting issues, thus enhancing the trade-off between robustness and generalization and additionally, this training approach also aids in mitigating "robust overfitting". Furthermore, our study provides valuable insights into the mechanisms of selective adversarial training and offers a promising avenue for future research.



## **48. Where and How to Attack? A Causality-Inspired Recipe for Generating Counterfactual Adversarial Examples**

cs.LG

Accepted by AAAI-2024

**SubmitDate**: 2024-01-26    [abs](http://arxiv.org/abs/2312.13628v2) [paper-pdf](http://arxiv.org/pdf/2312.13628v2)

**Authors**: Ruichu Cai, Yuxuan Zhu, Jie Qiao, Zefeng Liang, Furui Liu, Zhifeng Hao

**Abstract**: Deep neural networks (DNNs) have been demonstrated to be vulnerable to well-crafted \emph{adversarial examples}, which are generated through either well-conceived $\mathcal{L}_p$-norm restricted or unrestricted attacks. Nevertheless, the majority of those approaches assume that adversaries can modify any features as they wish, and neglect the causal generating process of the data, which is unreasonable and unpractical. For instance, a modification in income would inevitably impact features like the debt-to-income ratio within a banking system. By considering the underappreciated causal generating process, first, we pinpoint the source of the vulnerability of DNNs via the lens of causality, then give theoretical results to answer \emph{where to attack}. Second, considering the consequences of the attack interventions on the current state of the examples to generate more realistic adversarial examples, we propose CADE, a framework that can generate \textbf{C}ounterfactual \textbf{AD}versarial \textbf{E}xamples to answer \emph{how to attack}. The empirical results demonstrate CADE's effectiveness, as evidenced by its competitive performance across diverse attack scenarios, including white-box, transfer-based, and random intervention attacks.



## **49. A General Framework for Robust G-Invariance in G-Equivariant Networks**

cs.LG

**SubmitDate**: 2024-01-26    [abs](http://arxiv.org/abs/2310.18564v2) [paper-pdf](http://arxiv.org/pdf/2310.18564v2)

**Authors**: Sophia Sanborn, Nina Miolane

**Abstract**: We introduce a general method for achieving robust group-invariance in group-equivariant convolutional neural networks ($G$-CNNs), which we call the $G$-triple-correlation ($G$-TC) layer. The approach leverages the theory of the triple-correlation on groups, which is the unique, lowest-degree polynomial invariant map that is also complete. Many commonly used invariant maps--such as the max--are incomplete: they remove both group and signal structure. A complete invariant, by contrast, removes only the variation due to the actions of the group, while preserving all information about the structure of the signal. The completeness of the triple correlation endows the $G$-TC layer with strong robustness, which can be observed in its resistance to invariance-based adversarial attacks. In addition, we observe that it yields measurable improvements in classification accuracy over standard Max $G$-Pooling in $G$-CNN architectures. We provide a general and efficient implementation of the method for any discretized group, which requires only a table defining the group's product structure. We demonstrate the benefits of this method for $G$-CNNs defined on both commutative and non-commutative groups--$SO(2)$, $O(2)$, $SO(3)$, and $O(3)$ (discretized as the cyclic $C8$, dihedral $D16$, chiral octahedral $O$ and full octahedral $O_h$ groups)--acting on $\mathbb{R}^2$ and $\mathbb{R}^3$ on both $G$-MNIST and $G$-ModelNet10 datasets.



## **50. Physical Trajectory Inference Attack and Defense in Decentralized POI Recommendation**

cs.IR

**SubmitDate**: 2024-01-26    [abs](http://arxiv.org/abs/2401.14583v1) [paper-pdf](http://arxiv.org/pdf/2401.14583v1)

**Authors**: Jing Long, Tong Chen, Guanhua Ye, Kai Zheng, Nguyen Quoc Viet Hung, Hongzhi Yin

**Abstract**: As an indispensable personalized service within Location-Based Social Networks (LBSNs), the Point-of-Interest (POI) recommendation aims to assist individuals in discovering attractive and engaging places. However, the accurate recommendation capability relies on the powerful server collecting a vast amount of users' historical check-in data, posing significant risks of privacy breaches. Although several collaborative learning (CL) frameworks for POI recommendation enhance recommendation resilience and allow users to keep personal data on-device, they still share personal knowledge to improve recommendation performance, thus leaving vulnerabilities for potential attackers. Given this, we design a new Physical Trajectory Inference Attack (PTIA) to expose users' historical trajectories. Specifically, for each user, we identify the set of interacted POIs by analyzing the aggregated information from the target POIs and their correlated POIs. We evaluate the effectiveness of PTIA on two real-world datasets across two types of decentralized CL frameworks for POI recommendation. Empirical results demonstrate that PTIA poses a significant threat to users' historical trajectories. Furthermore, Local Differential Privacy (LDP), the traditional privacy-preserving method for CL frameworks, has also been proven ineffective against PTIA. In light of this, we propose a novel defense mechanism (AGD) against PTIA based on an adversarial game to eliminate sensitive POIs and their information in correlated POIs. After conducting intensive experiments, AGD has been proven precise and practical, with minimal impact on recommendation performance.



