# Latest Adversarial Attack Papers
**update at 2024-05-17 09:53:10**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. Protecting Your LLMs with Information Bottleneck**

cs.CL

23 pages, 7 figures, 8 tables

**SubmitDate**: 2024-05-16    [abs](http://arxiv.org/abs/2404.13968v2) [paper-pdf](http://arxiv.org/pdf/2404.13968v2)

**Authors**: Zichuan Liu, Zefan Wang, Linjie Xu, Jinyu Wang, Lei Song, Tianchun Wang, Chunlin Chen, Wei Cheng, Jiang Bian

**Abstract**: The advent of large language models (LLMs) has revolutionized the field of natural language processing, yet they might be attacked to produce harmful content. Despite efforts to ethically align LLMs, these are often fragile and can be circumvented by jailbreaking attacks through optimized or manual adversarial prompts. To address this, we introduce the Information Bottleneck Protector (IBProtector), a defense mechanism grounded in the information bottleneck principle, and we modify the objective to avoid trivial solutions. The IBProtector selectively compresses and perturbs prompts, facilitated by a lightweight and trainable extractor, preserving only essential information for the target LLMs to respond with the expected answer. Moreover, we further consider a situation where the gradient is not visible to be compatible with any LLM. Our empirical evaluations show that IBProtector outperforms current defense methods in mitigating jailbreak attempts, without overly affecting response quality or inference speed. Its effectiveness and adaptability across various attack methods and target LLMs underscore the potential of IBProtector as a novel, transferable defense that bolsters the security of LLMs without requiring modifications to the underlying models.



## **2. Adversarial Robustness for Visual Grounding of Multimodal Large Language Models**

cs.CV

ICLR 2024 Workshop on Reliable and Responsible Foundation Models

**SubmitDate**: 2024-05-16    [abs](http://arxiv.org/abs/2405.09981v1) [paper-pdf](http://arxiv.org/pdf/2405.09981v1)

**Authors**: Kuofeng Gao, Yang Bai, Jiawang Bai, Yong Yang, Shu-Tao Xia

**Abstract**: Multi-modal Large Language Models (MLLMs) have recently achieved enhanced performance across various vision-language tasks including visual grounding capabilities. However, the adversarial robustness of visual grounding remains unexplored in MLLMs. To fill this gap, we use referring expression comprehension (REC) as an example task in visual grounding and propose three adversarial attack paradigms as follows. Firstly, untargeted adversarial attacks induce MLLMs to generate incorrect bounding boxes for each object. Besides, exclusive targeted adversarial attacks cause all generated outputs to the same target bounding box. In addition, permuted targeted adversarial attacks aim to permute all bounding boxes among different objects within a single image. Extensive experiments demonstrate that the proposed methods can successfully attack visual grounding capabilities of MLLMs. Our methods not only provide a new perspective for designing novel attacks but also serve as a strong baseline for improving the adversarial robustness for visual grounding of MLLMs.



## **3. Deepfake Generation and Detection: A Benchmark and Survey**

cs.CV

We closely follow the latest developments in  https://github.com/flyingby/Awesome-Deepfake-Generation-and-Detection

**SubmitDate**: 2024-05-16    [abs](http://arxiv.org/abs/2403.17881v4) [paper-pdf](http://arxiv.org/pdf/2403.17881v4)

**Authors**: Gan Pei, Jiangning Zhang, Menghan Hu, Zhenyu Zhang, Chengjie Wang, Yunsheng Wu, Guangtao Zhai, Jian Yang, Chunhua Shen, Dacheng Tao

**Abstract**: Deepfake is a technology dedicated to creating highly realistic facial images and videos under specific conditions, which has significant application potential in fields such as entertainment, movie production, digital human creation, to name a few. With the advancements in deep learning, techniques primarily represented by Variational Autoencoders and Generative Adversarial Networks have achieved impressive generation results. More recently, the emergence of diffusion models with powerful generation capabilities has sparked a renewed wave of research. In addition to deepfake generation, corresponding detection technologies continuously evolve to regulate the potential misuse of deepfakes, such as for privacy invasion and phishing attacks. This survey comprehensively reviews the latest developments in deepfake generation and detection, summarizing and analyzing current state-of-the-arts in this rapidly evolving field. We first unify task definitions, comprehensively introduce datasets and metrics, and discuss developing technologies. Then, we discuss the development of several related sub-fields and focus on researching four representative deepfake fields: face swapping, face reenactment, talking face generation, and facial attribute editing, as well as forgery detection. Subsequently, we comprehensively benchmark representative methods on popular datasets for each field, fully evaluating the latest and influential published works. Finally, we analyze challenges and future research directions of the discussed fields.



## **4. Infrared Adversarial Car Stickers**

cs.CV

Accepted by CVPR 2024

**SubmitDate**: 2024-05-16    [abs](http://arxiv.org/abs/2405.09924v1) [paper-pdf](http://arxiv.org/pdf/2405.09924v1)

**Authors**: Xiaopei Zhu, Yuqiu Liu, Zhanhao Hu, Jianmin Li, Xiaolin Hu

**Abstract**: Infrared physical adversarial examples are of great significance for studying the security of infrared AI systems that are widely used in our lives such as autonomous driving. Previous infrared physical attacks mainly focused on 2D infrared pedestrian detection which may not fully manifest its destructiveness to AI systems. In this work, we propose a physical attack method against infrared detectors based on 3D modeling, which is applied to a real car. The goal is to design a set of infrared adversarial stickers to make cars invisible to infrared detectors at various viewing angles, distances, and scenes. We build a 3D infrared car model with real infrared characteristics and propose an infrared adversarial pattern generation method based on 3D mesh shadow. We propose a 3D control points-based mesh smoothing algorithm and use a set of smoothness loss functions to enhance the smoothness of adversarial meshes and facilitate the sticker implementation. Besides, We designed the aluminum stickers and conducted physical experiments on two real Mercedes-Benz A200L cars. Our adversarial stickers hid the cars from Faster RCNN, an object detector, at various viewing angles, distances, and scenes. The attack success rate (ASR) was 91.49% for real cars. In comparison, the ASRs of random stickers and no sticker were only 6.21% and 0.66%, respectively. In addition, the ASRs of the designed stickers against six unseen object detectors such as YOLOv3 and Deformable DETR were between 73.35%-95.80%, showing good transferability of the attack performance across detectors.



## **5. DiffAM: Diffusion-based Adversarial Makeup Transfer for Facial Privacy Protection**

cs.CV

16 pages, 11 figures

**SubmitDate**: 2024-05-16    [abs](http://arxiv.org/abs/2405.09882v1) [paper-pdf](http://arxiv.org/pdf/2405.09882v1)

**Authors**: Yuhao Sun, Lingyun Yu, Hongtao Xie, Jiaming Li, Yongdong Zhang

**Abstract**: With the rapid development of face recognition (FR) systems, the privacy of face images on social media is facing severe challenges due to the abuse of unauthorized FR systems. Some studies utilize adversarial attack techniques to defend against malicious FR systems by generating adversarial examples. However, the generated adversarial examples, i.e., the protected face images, tend to suffer from subpar visual quality and low transferability. In this paper, we propose a novel face protection approach, dubbed DiffAM, which leverages the powerful generative ability of diffusion models to generate high-quality protected face images with adversarial makeup transferred from reference images. To be specific, we first introduce a makeup removal module to generate non-makeup images utilizing a fine-tuned diffusion model with guidance of textual prompts in CLIP space. As the inverse process of makeup transfer, makeup removal can make it easier to establish the deterministic relationship between makeup domain and non-makeup domain regardless of elaborate text prompts. Then, with this relationship, a CLIP-based makeup loss along with an ensemble attack strategy is introduced to jointly guide the direction of adversarial makeup domain, achieving the generation of protected face images with natural-looking makeup and high black-box transferability. Extensive experiments demonstrate that DiffAM achieves higher visual quality and attack success rates with a gain of 12.98% under black-box setting compared with the state of the arts. The code will be available at https://github.com/HansSunY/DiffAM.



## **6. Box-Free Model Watermarks Are Prone to Black-Box Removal Attacks**

cs.CV

**SubmitDate**: 2024-05-16    [abs](http://arxiv.org/abs/2405.09863v1) [paper-pdf](http://arxiv.org/pdf/2405.09863v1)

**Authors**: Haonan An, Guang Hua, Zhiping Lin, Yuguang Fang

**Abstract**: Box-free model watermarking is an emerging technique to safeguard the intellectual property of deep learning models, particularly those for low-level image processing tasks. Existing works have verified and improved its effectiveness in several aspects. However, in this paper, we reveal that box-free model watermarking is prone to removal attacks, even under the real-world threat model such that the protected model and the watermark extractor are in black boxes. Under this setting, we carry out three studies. 1) We develop an extractor-gradient-guided (EGG) remover and show its effectiveness when the extractor uses ReLU activation only. 2) More generally, for an unknown extractor, we leverage adversarial attacks and design the EGG remover based on the estimated gradients. 3) Under the most stringent condition that the extractor is inaccessible, we design a transferable remover based on a set of private proxy models. In all cases, the proposed removers can successfully remove embedded watermarks while preserving the quality of the processed images, and we also demonstrate that the EGG remover can even replace the watermarks. Extensive experimental results verify the effectiveness and generalizability of the proposed attacks, revealing the vulnerabilities of the existing box-free methods and calling for further research.



## **7. Manifold Integrated Gradients: Riemannian Geometry for Feature Attribution**

cs.LG

Accepted at ICML 2024

**SubmitDate**: 2024-05-16    [abs](http://arxiv.org/abs/2405.09800v1) [paper-pdf](http://arxiv.org/pdf/2405.09800v1)

**Authors**: Eslam Zaher, Maciej Trzaskowski, Quan Nguyen, Fred Roosta

**Abstract**: In this paper, we dive into the reliability concerns of Integrated Gradients (IG), a prevalent feature attribution method for black-box deep learning models. We particularly address two predominant challenges associated with IG: the generation of noisy feature visualizations for vision models and the vulnerability to adversarial attributional attacks. Our approach involves an adaptation of path-based feature attribution, aligning the path of attribution more closely to the intrinsic geometry of the data manifold. Our experiments utilise deep generative models applied to several real-world image datasets. They demonstrate that IG along the geodesics conforms to the curved geometry of the Riemannian data manifold, generating more perceptually intuitive explanations and, subsequently, substantially increasing robustness to targeted attributional attacks.



## **8. IBD-PSC: Input-level Backdoor Detection via Parameter-oriented Scaling Consistency**

cs.LG

Accepted to ICML 2024, 29 pages

**SubmitDate**: 2024-05-16    [abs](http://arxiv.org/abs/2405.09786v1) [paper-pdf](http://arxiv.org/pdf/2405.09786v1)

**Authors**: Linshan Hou, Ruili Feng, Zhongyun Hua, Wei Luo, Leo Yu Zhang, Yiming Li

**Abstract**: Deep neural networks (DNNs) are vulnerable to backdoor attacks, where adversaries can maliciously trigger model misclassifications by implanting a hidden backdoor during model training. This paper proposes a simple yet effective input-level backdoor detection (dubbed IBD-PSC) as a 'firewall' to filter out malicious testing images. Our method is motivated by an intriguing phenomenon, i.e., parameter-oriented scaling consistency (PSC), where the prediction confidences of poisoned samples are significantly more consistent than those of benign ones when amplifying model parameters. In particular, we provide theoretical analysis to safeguard the foundations of the PSC phenomenon. We also design an adaptive method to select BN layers to scale up for effective detection. Extensive experiments are conducted on benchmark datasets, verifying the effectiveness and efficiency of our IBD-PSC method and its resistance to adaptive attacks.



## **9. Towards Evaluating the Robustness of Automatic Speech Recognition Systems via Audio Style Transfer**

cs.SD

Accepted to SecTL (AsiaCCS Workshop) 2024

**SubmitDate**: 2024-05-15    [abs](http://arxiv.org/abs/2405.09470v1) [paper-pdf](http://arxiv.org/pdf/2405.09470v1)

**Authors**: Weifei Jin, Yuxin Cao, Junjie Su, Qi Shen, Kai Ye, Derui Wang, Jie Hao, Ziyao Liu

**Abstract**: In light of the widespread application of Automatic Speech Recognition (ASR) systems, their security concerns have received much more attention than ever before, primarily due to the susceptibility of Deep Neural Networks. Previous studies have illustrated that surreptitiously crafting adversarial perturbations enables the manipulation of speech recognition systems, resulting in the production of malicious commands. These attack methods mostly require adding noise perturbations under $\ell_p$ norm constraints, inevitably leaving behind artifacts of manual modifications. Recent research has alleviated this limitation by manipulating style vectors to synthesize adversarial examples based on Text-to-Speech (TTS) synthesis audio. However, style modifications based on optimization objectives significantly reduce the controllability and editability of audio styles. In this paper, we propose an attack on ASR systems based on user-customized style transfer. We first test the effect of Style Transfer Attack (STA) which combines style transfer and adversarial attack in sequential order. And then, as an improvement, we propose an iterative Style Code Attack (SCA) to maintain audio quality. Experimental results show that our method can meet the need for user-customized styles and achieve a success rate of 82% in attacks, while keeping sound naturalness due to our user study.



## **10. Inexact Unlearning Needs More Careful Evaluations to Avoid a False Sense of Privacy**

cs.LG

**SubmitDate**: 2024-05-15    [abs](http://arxiv.org/abs/2403.01218v2) [paper-pdf](http://arxiv.org/pdf/2403.01218v2)

**Authors**: Jamie Hayes, Ilia Shumailov, Eleni Triantafillou, Amr Khalifa, Nicolas Papernot

**Abstract**: The high cost of model training makes it increasingly desirable to develop techniques for unlearning. These techniques seek to remove the influence of a training example without having to retrain the model from scratch. Intuitively, once a model has unlearned, an adversary that interacts with the model should no longer be able to tell whether the unlearned example was included in the model's training set or not. In the privacy literature, this is known as membership inference. In this work, we discuss adaptations of Membership Inference Attacks (MIAs) to the setting of unlearning (leading to their ``U-MIA'' counterparts). We propose a categorization of existing U-MIAs into ``population U-MIAs'', where the same attacker is instantiated for all examples, and ``per-example U-MIAs'', where a dedicated attacker is instantiated for each example. We show that the latter category, wherein the attacker tailors its membership prediction to each example under attack, is significantly stronger. Indeed, our results show that the commonly used U-MIAs in the unlearning literature overestimate the privacy protection afforded by existing unlearning techniques on both vision and language models. Our investigation reveals a large variance in the vulnerability of different examples to per-example U-MIAs. In fact, several unlearning algorithms lead to a reduced vulnerability for some, but not all, examples that we wish to unlearn, at the expense of increasing it for other examples. Notably, we find that the privacy protection for the remaining training examples may worsen as a consequence of unlearning. We also discuss the fundamental difficulty of equally protecting all examples using existing unlearning schemes, due to the different rates at which examples are unlearned. We demonstrate that naive attempts at tailoring unlearning stopping criteria to different examples fail to alleviate these issues.



## **11. Properties that allow or prohibit transferability of adversarial attacks among quantized networks**

cs.LG

**SubmitDate**: 2024-05-15    [abs](http://arxiv.org/abs/2405.09598v1) [paper-pdf](http://arxiv.org/pdf/2405.09598v1)

**Authors**: Abhishek Shrestha, Jürgen Großmann

**Abstract**: Deep Neural Networks (DNNs) are known to be vulnerable to adversarial examples. Further, these adversarial examples are found to be transferable from the source network in which they are crafted to a black-box target network. As the trend of using deep learning on embedded devices grows, it becomes relevant to study the transferability properties of adversarial examples among compressed networks. In this paper, we consider quantization as a network compression technique and evaluate the performance of transfer-based attacks when the source and target networks are quantized at different bitwidths. We explore how algorithm specific properties affect transferability by considering various adversarial example generation algorithms. Furthermore, we examine transferability in a more realistic scenario where the source and target networks may differ in bitwidth and other model-related properties like capacity and architecture. We find that although quantization reduces transferability, certain attack types demonstrate an ability to enhance it. Additionally, the average transferability of adversarial examples among quantized versions of a network can be used to estimate the transferability to quantized target networks with varying capacity and architecture.



## **12. "Do Anything Now": Characterizing and Evaluating In-The-Wild Jailbreak Prompts on Large Language Models**

cs.CR

**SubmitDate**: 2024-05-15    [abs](http://arxiv.org/abs/2308.03825v2) [paper-pdf](http://arxiv.org/pdf/2308.03825v2)

**Authors**: Xinyue Shen, Zeyuan Chen, Michael Backes, Yun Shen, Yang Zhang

**Abstract**: The misuse of large language models (LLMs) has drawn significant attention from the general public and LLM vendors. One particular type of adversarial prompt, known as jailbreak prompt, has emerged as the main attack vector to bypass the safeguards and elicit harmful content from LLMs. In this paper, employing our new framework JailbreakHub, we conduct a comprehensive analysis of 1,405 jailbreak prompts spanning from December 2022 to December 2023. We identify 131 jailbreak communities and discover unique characteristics of jailbreak prompts and their major attack strategies, such as prompt injection and privilege escalation. We also observe that jailbreak prompts increasingly shift from online Web communities to prompt-aggregation websites and 28 user accounts have consistently optimized jailbreak prompts over 100 days. To assess the potential harm caused by jailbreak prompts, we create a question set comprising 107,250 samples across 13 forbidden scenarios. Leveraging this dataset, our experiments on six popular LLMs show that their safeguards cannot adequately defend jailbreak prompts in all scenarios. Particularly, we identify five highly effective jailbreak prompts that achieve 0.95 attack success rates on ChatGPT (GPT-3.5) and GPT-4, and the earliest one has persisted online for over 240 days. We hope that our study can facilitate the research community and LLM vendors in promoting safer and regulated LLMs.



## **13. Cross-Input Certified Training for Universal Perturbations**

cs.LG

21 pages, 5 figures

**SubmitDate**: 2024-05-15    [abs](http://arxiv.org/abs/2405.09176v1) [paper-pdf](http://arxiv.org/pdf/2405.09176v1)

**Authors**: Changming Xu, Gagandeep Singh

**Abstract**: Existing work in trustworthy machine learning primarily focuses on single-input adversarial perturbations. In many real-world attack scenarios, input-agnostic adversarial attacks, e.g. universal adversarial perturbations (UAPs), are much more feasible. Current certified training methods train models robust to single-input perturbations but achieve suboptimal clean and UAP accuracy, thereby limiting their applicability in practical applications. We propose a novel method, CITRUS, for certified training of networks robust against UAP attackers. We show in an extensive evaluation across different datasets, architectures, and perturbation magnitudes that our method outperforms traditional certified training methods on standard accuracy (up to 10.3\%) and achieves SOTA performance on the more practical certified UAP accuracy metric.



## **14. The Economic Limits of Permissionless Consensus**

cs.DC

**SubmitDate**: 2024-05-15    [abs](http://arxiv.org/abs/2405.09173v1) [paper-pdf](http://arxiv.org/pdf/2405.09173v1)

**Authors**: Eric Budish, Andrew Lewis-Pye, Tim Roughgarden

**Abstract**: The purpose of a consensus protocol is to keep a distributed network of nodes "in sync," even in the presence of an unpredictable communication network and adversarial behavior by some of the participating nodes. In the permissionless setting, these nodes may be operated by unknown players, with each player free to use multiple identifiers and to start or stop running the protocol at any time. Establishing that a permissionless consensus protocol is "secure" thus requires both a distributed computing argument (that the protocol guarantees consistency and liveness unless the fraction of adversarial participation is sufficiently large) and an economic argument (that carrying out an attack would be prohibitively expensive for an attacker). There is a mature toolbox for assembling arguments of the former type; the goal of this paper is to lay the foundations for arguments of the latter type.   An ideal permissionless consensus protocol would, in addition to satisfying standard consistency and liveness guarantees, render consistency violations prohibitively expensive for the attacker without collateral damage to honest participants. We make this idea precise with our notion of the EAAC (expensive to attack in the absence of collapse) property, and prove the following results:   1. In the synchronous and dynamically available setting, with an adversary that controls at least one-half of the overall resources, no protocol can be EAAC.   2. In the partially synchronous and quasi-permissionless setting, with an adversary that controls at least one-third of the overall resources, no protocol can be EAAC.   3. In the synchronous and quasi-permissionless setting, there is a proof-of-stake protocol that, provided the adversary controls less than two-thirds of the overall stake, satisfies the EAAC property.   All three results are optimal with respect to the size of the adversary.



## **15. Dynamic Adversarial Attacks on Autonomous Driving Systems**

cs.RO

**SubmitDate**: 2024-05-15    [abs](http://arxiv.org/abs/2312.06701v2) [paper-pdf](http://arxiv.org/pdf/2312.06701v2)

**Authors**: Amirhosein Chahe, Chenan Wang, Abhishek Jeyapratap, Kaidi Xu, Lifeng Zhou

**Abstract**: This paper introduces an attacking mechanism to challenge the resilience of autonomous driving systems. Specifically, we manipulate the decision-making processes of an autonomous vehicle by dynamically displaying adversarial patches on a screen mounted on another moving vehicle. These patches are optimized to deceive the object detection models into misclassifying targeted objects, e.g., traffic signs. Such manipulation has significant implications for critical multi-vehicle interactions such as intersection crossing and lane changing, which are vital for safe and efficient autonomous driving systems. Particularly, we make four major contributions. First, we introduce a novel adversarial attack approach where the patch is not co-located with its target, enabling more versatile and stealthy attacks. Moreover, our method utilizes dynamic patches displayed on a screen, allowing for adaptive changes and movement, enhancing the flexibility and performance of the attack. To do so, we design a Screen Image Transformation Network (SIT-Net), which simulates environmental effects on the displayed images, narrowing the gap between simulated and real-world scenarios. Further, we integrate a positional loss term into the adversarial training process to increase the success rate of the dynamic attack. Finally, we shift the focus from merely attacking perceptual systems to influencing the decision-making algorithms of self-driving systems. Our experiments demonstrate the first successful implementation of such dynamic adversarial attacks in real-world autonomous driving scenarios, paving the way for advancements in the field of robust and secure autonomous driving.



## **16. Optimizing Sensor Network Design for Multiple Coverage**

cs.LG

**SubmitDate**: 2024-05-15    [abs](http://arxiv.org/abs/2405.09096v1) [paper-pdf](http://arxiv.org/pdf/2405.09096v1)

**Authors**: Lukas Taus, Yen-Hsi Richard Tsai

**Abstract**: Sensor placement optimization methods have been studied extensively. They can be applied to a wide range of applications, including surveillance of known environments, optimal locations for 5G towers, and placement of missile defense systems. However, few works explore the robustness and efficiency of the resulting sensor network concerning sensor failure or adversarial attacks. This paper addresses this issue by optimizing for the least number of sensors to achieve multiple coverage of non-simply connected domains by a prescribed number of sensors. We introduce a new objective function for the greedy (next-best-view) algorithm to design efficient and robust sensor networks and derive theoretical bounds on the network's optimality. We further introduce a Deep Learning model to accelerate the algorithm for near real-time computations. The Deep Learning model requires the generation of training examples. Correspondingly, we show that understanding the geometric properties of the training data set provides important insights into the performance and training process of deep learning techniques. Finally, we demonstrate that a simple parallel version of the greedy approach using a simpler objective can be highly competitive.



## **17. The Pitfalls and Promise of Conformal Inference Under Adversarial Attacks**

cs.LG

ICML2024

**SubmitDate**: 2024-05-14    [abs](http://arxiv.org/abs/2405.08886v1) [paper-pdf](http://arxiv.org/pdf/2405.08886v1)

**Authors**: Ziquan Liu, Yufei Cui, Yan Yan, Yi Xu, Xiangyang Ji, Xue Liu, Antoni B. Chan

**Abstract**: In safety-critical applications such as medical imaging and autonomous driving, where decisions have profound implications for patient health and road safety, it is imperative to maintain both high adversarial robustness to protect against potential adversarial attacks and reliable uncertainty quantification in decision-making. With extensive research focused on enhancing adversarial robustness through various forms of adversarial training (AT), a notable knowledge gap remains concerning the uncertainty inherent in adversarially trained models. To address this gap, this study investigates the uncertainty of deep learning models by examining the performance of conformal prediction (CP) in the context of standard adversarial attacks within the adversarial defense community. It is first unveiled that existing CP methods do not produce informative prediction sets under the commonly used $l_{\infty}$-norm bounded attack if the model is not adversarially trained, which underpins the importance of adversarial training for CP. Our paper next demonstrates that the prediction set size (PSS) of CP using adversarially trained models with AT variants is often worse than using standard AT, inspiring us to research into CP-efficient AT for improved PSS. We propose to optimize a Beta-weighting loss with an entropy minimization regularizer during AT to improve CP-efficiency, where the Beta-weighting loss is shown to be an upper bound of PSS at the population level by our theoretical analysis. Moreover, our empirical study on four image classification datasets across three popular AT baselines validates the effectiveness of the proposed Uncertainty-Reducing AT (AT-UR).



## **18. S3C2 Summit 2024-03: Industry Secure Supply Chain Summit**

cs.CR

This is our WIP paper on the Summit. More versions will be released  soon

**SubmitDate**: 2024-05-14    [abs](http://arxiv.org/abs/2405.08762v1) [paper-pdf](http://arxiv.org/pdf/2405.08762v1)

**Authors**: Greg Tystahl, Yasemin Acar, Michel Cukier, William Enck, Christian Kastner, Alexandros Kapravelos, Dominik Wermke, Laurie Williams

**Abstract**: Supply chain security has become a very important vector to consider when defending against adversary attacks. Due to this, more and more developers are keen on improving their supply chains to make them more robust against future threats. On March 7th, 2024 researchers from the Secure Software Supply Chain Center (S3C2) gathered 14 industry leaders, developers and consumers of the open source ecosystem to discuss the state of supply chain security. The goal of the summit is to share insights between companies and developers alike to foster new collaborations and ideas moving forward. Through this meeting, participants were questions on best practices and thoughts how to improve things for the future. In this paper we summarize the responses and discussions of the summit. The panel questions can be found in the appendix.



## **19. Design and Analysis of Resilient Vehicular Platoon Systems over Wireless Networks**

eess.SY

6 pages, 4 figures, in submission of Globecom 2024

**SubmitDate**: 2024-05-14    [abs](http://arxiv.org/abs/2405.08706v1) [paper-pdf](http://arxiv.org/pdf/2405.08706v1)

**Authors**: Tingyu Shui, Walid Saad

**Abstract**: Connected vehicular platoons provide a promising solution to improve traffic efficiency and ensure road safety. Vehicles in a platoon utilize on-board sensors and wireless vehicle-to-vehicle (V2V) links to share traffic information for cooperative adaptive cruise control. To process real-time control and alert information, there is a need to ensure clock synchronization among the platoon's vehicles. However, adversaries can jeopardize the operation of the platoon by attacking the local clocks of vehicles, leading to clock offsets with the platoon's reference clock. In this paper, a novel framework is proposed for analyzing the resilience of vehicular platoons that are connected using V2V links. In particular, a resilient design based on a diffusion protocol is proposed to re-synchronize the attacked vehicle through wireless V2V links thereby mitigating the impact of variance of the transmission delay during recovery. Then, a novel metric named temporal conditional mean exceedance is defined and analyzed in order to characterize the resilience of the platoon. Subsequently, the conditions pertaining to the V2V links and recovery time needed for a resilient design are derived. Numerical results show that the proposed resilient design is feasible in face of a nine-fold increase in the variance of transmission delay compared to a baseline designed for reliability. Moreover, the proposed approach improves the reliability, defined as the probability of meeting a desired clock offset error requirement, by 45% compared to the baseline.



## **20. Quantum Oblivious LWE Sampling and Insecurity of Standard Model Lattice-Based SNARKs**

cs.CR

**SubmitDate**: 2024-05-14    [abs](http://arxiv.org/abs/2401.03807v2) [paper-pdf](http://arxiv.org/pdf/2401.03807v2)

**Authors**: Thomas Debris-Alazard, Pouria Fallahpour, Damien Stehlé

**Abstract**: The Learning With Errors ($\mathsf{LWE}$) problem asks to find $\mathbf{s}$ from an input of the form $(\mathbf{A}, \mathbf{b} = \mathbf{A}\mathbf{s}+\mathbf{e}) \in (\mathbb{Z}/q\mathbb{Z})^{m \times n} \times (\mathbb{Z}/q\mathbb{Z})^{m}$, for a vector $\mathbf{e}$ that has small-magnitude entries. In this work, we do not focus on solving $\mathsf{LWE}$ but on the task of sampling instances. As these are extremely sparse in their range, it may seem plausible that the only way to proceed is to first create $\mathbf{s}$ and $\mathbf{e}$ and then set $\mathbf{b} = \mathbf{A}\mathbf{s}+\mathbf{e}$. In particular, such an instance sampler knows the solution. This raises the question whether it is possible to obliviously sample $(\mathbf{A}, \mathbf{A}\mathbf{s}+\mathbf{e})$, namely, without knowing the underlying $\mathbf{s}$. A variant of the assumption that oblivious $\mathsf{LWE}$ sampling is hard has been used in a series of works to analyze the security of candidate constructions of Succinct Non interactive Arguments of Knowledge (SNARKs). As the assumption is related to $\mathsf{LWE}$, these SNARKs have been conjectured to be secure in the presence of quantum adversaries.   Our main result is a quantum polynomial-time algorithm that samples well-distributed $\mathsf{LWE}$ instances while provably not knowing the solution, under the assumption that $\mathsf{LWE}$ is hard. Moreover, the approach works for a vast range of $\mathsf{LWE}$ parametrizations, including those used in the above-mentioned SNARKs. This invalidates the assumptions used in their security analyses, although it does not yield attacks against the constructions themselves.



## **21. PLeak: Prompt Leaking Attacks against Large Language Model Applications**

cs.CR

To appear in the Proceedings of The ACM Conference on Computer and  Communications Security (CCS), 2024

**SubmitDate**: 2024-05-14    [abs](http://arxiv.org/abs/2405.06823v2) [paper-pdf](http://arxiv.org/pdf/2405.06823v2)

**Authors**: Bo Hui, Haolin Yuan, Neil Gong, Philippe Burlina, Yinzhi Cao

**Abstract**: Large Language Models (LLMs) enable a new ecosystem with many downstream applications, called LLM applications, with different natural language processing tasks. The functionality and performance of an LLM application highly depend on its system prompt, which instructs the backend LLM on what task to perform. Therefore, an LLM application developer often keeps a system prompt confidential to protect its intellectual property. As a result, a natural attack, called prompt leaking, is to steal the system prompt from an LLM application, which compromises the developer's intellectual property. Existing prompt leaking attacks primarily rely on manually crafted queries, and thus achieve limited effectiveness.   In this paper, we design a novel, closed-box prompt leaking attack framework, called PLeak, to optimize an adversarial query such that when the attacker sends it to a target LLM application, its response reveals its own system prompt. We formulate finding such an adversarial query as an optimization problem and solve it with a gradient-based method approximately. Our key idea is to break down the optimization goal by optimizing adversary queries for system prompts incrementally, i.e., starting from the first few tokens of each system prompt step by step until the entire length of the system prompt.   We evaluate PLeak in both offline settings and for real-world LLM applications, e.g., those on Poe, a popular platform hosting such applications. Our results show that PLeak can effectively leak system prompts and significantly outperforms not only baselines that manually curate queries but also baselines with optimized queries that are modified and adapted from existing jailbreaking attacks. We responsibly reported the issues to Poe and are still waiting for their response. Our implementation is available at this repository: https://github.com/BHui97/PLeak.



## **22. Certifying Robustness of Graph Convolutional Networks for Node Perturbation with Polyhedra Abstract Interpretation**

cs.LG

**SubmitDate**: 2024-05-14    [abs](http://arxiv.org/abs/2405.08645v1) [paper-pdf](http://arxiv.org/pdf/2405.08645v1)

**Authors**: Boqi Chen, Kristóf Marussy, Oszkár Semeráth, Gunter Mussbacher, Dániel Varró

**Abstract**: Graph convolutional neural networks (GCNs) are powerful tools for learning graph-based knowledge representations from training data. However, they are vulnerable to small perturbations in the input graph, which makes them susceptible to input faults or adversarial attacks. This poses a significant problem for GCNs intended to be used in critical applications, which need to provide certifiably robust services even in the presence of adversarial perturbations. We propose an improved GCN robustness certification technique for node classification in the presence of node feature perturbations. We introduce a novel polyhedra-based abstract interpretation approach to tackle specific challenges of graph data and provide tight upper and lower bounds for the robustness of the GCN. Experiments show that our approach simultaneously improves the tightness of robustness bounds as well as the runtime performance of certification. Moreover, our method can be used during training to further improve the robustness of GCNs.



## **23. HookChain: A new perspective for Bypassing EDR Solutions**

cs.CR

46 pages, 22 figures, HookChain, Bypass EDR, Evading EDR, IAT Hook,  Halo's Gate

**SubmitDate**: 2024-05-14    [abs](http://arxiv.org/abs/2404.16856v2) [paper-pdf](http://arxiv.org/pdf/2404.16856v2)

**Authors**: Helvio Carvalho Junior

**Abstract**: In the current digital security ecosystem, where threats evolve rapidly and with complexity, companies developing Endpoint Detection and Response (EDR) solutions are in constant search for innovations that not only keep up but also anticipate emerging attack vectors. In this context, this article introduces the HookChain, a look from another perspective at widely known techniques, which when combined, provide an additional layer of sophisticated evasion against traditional EDR systems. Through a precise combination of IAT Hooking techniques, dynamic SSN resolution, and indirect system calls, HookChain redirects the execution flow of Windows subsystems in a way that remains invisible to the vigilant eyes of EDRs that only act on Ntdll.dll, without requiring changes to the source code of the applications and malwares involved. This work not only challenges current conventions in cybersecurity but also sheds light on a promising path for future protection strategies, leveraging the understanding that continuous evolution is key to the effectiveness of digital security. By developing and exploring the HookChain technique, this study significantly contributes to the body of knowledge in endpoint security, stimulating the development of more robust and adaptive solutions that can effectively address the ever-changing dynamics of digital threats. This work aspires to inspire deep reflection and advancement in the research and development of security technologies that are always several steps ahead of adversaries.   UNDER CONSTRUCTION RESEARCH: This paper is not the final version, as it is currently undergoing final tests against several EDRs. We expect to release the final version by August 2024.



## **24. Secure Aggregation Meets Sparsification in Decentralized Learning**

cs.LG

**SubmitDate**: 2024-05-14    [abs](http://arxiv.org/abs/2405.07708v2) [paper-pdf](http://arxiv.org/pdf/2405.07708v2)

**Authors**: Sayan Biswas, Anne-Marie Kermarrec, Rafael Pires, Rishi Sharma, Milos Vujasinovic

**Abstract**: Decentralized learning (DL) faces increased vulnerability to privacy breaches due to sophisticated attacks on machine learning (ML) models. Secure aggregation is a computationally efficient cryptographic technique that enables multiple parties to compute an aggregate of their private data while keeping their individual inputs concealed from each other and from any central aggregator. To enhance communication efficiency in DL, sparsification techniques are used, selectively sharing only the most crucial parameters or gradients in a model, thereby maintaining efficiency without notably compromising accuracy. However, applying secure aggregation to sparsified models in DL is challenging due to the transmission of disjoint parameter sets by distinct nodes, which can prevent masks from canceling out effectively. This paper introduces CESAR, a novel secure aggregation protocol for DL designed to be compatible with existing sparsification mechanisms. CESAR provably defends against honest-but-curious adversaries and can be formally adapted to counteract collusion between them. We provide a foundational understanding of the interaction between the sparsification carried out by the nodes and the proportion of the parameters shared under CESAR in both colluding and non-colluding environments, offering analytical insight into the working and applicability of the protocol. Experiments on a network with 48 nodes in a 3-regular topology show that with random subsampling, CESAR is always within 0.5% accuracy of decentralized parallel stochastic gradient descent (D-PSGD), while adding only 11% of data overhead. Moreover, it surpasses the accuracy on TopK by up to 0.3% on independent and identically distributed (IID) data.



## **25. UnMarker: A Universal Attack on Defensive Watermarking**

cs.CR

**SubmitDate**: 2024-05-14    [abs](http://arxiv.org/abs/2405.08363v1) [paper-pdf](http://arxiv.org/pdf/2405.08363v1)

**Authors**: Andre Kassis, Urs Hengartner

**Abstract**: Reports regarding the misuse of $\textit{Generative AI}$ ($\textit{GenAI}$) to create harmful deepfakes are emerging daily. Recently, defensive watermarking, which enables $\textit{GenAI}$ providers to hide fingerprints in their images to later use for deepfake detection, has been on the rise. Yet, its potential has not been fully explored. We present $\textit{UnMarker}$ -- the first practical $\textit{universal}$ attack on defensive watermarking. Unlike existing attacks, $\textit{UnMarker}$ requires no detector feedback, no unrealistic knowledge of the scheme or similar models, and no advanced denoising pipelines that may not be available. Instead, being the product of an in-depth analysis of the watermarking paradigm revealing that robust schemes must construct their watermarks in the spectral amplitudes, $\textit{UnMarker}$ employs two novel adversarial optimizations to disrupt the spectra of watermarked images, erasing the watermarks. Evaluations against the $\textit{SOTA}$ prove its effectiveness, not only defeating traditional schemes while retaining superior quality compared to existing attacks but also breaking $\textit{semantic}$ watermarks that alter the image's structure, reducing the best detection rate to $43\%$ and rendering them useless. To our knowledge, $\textit{UnMarker}$ is the first practical attack on $\textit{semantic}$ watermarks, which have been deemed the future of robust watermarking. $\textit{UnMarker}$ casts doubts on the very penitential of this countermeasure and exposes its paradoxical nature as designing schemes for robustness inevitably compromises other robustness aspects.



## **26. SpeechGuard: Exploring the Adversarial Robustness of Multimodal Large Language Models**

cs.CL

9+6 pages, Submitted to ACL 2024

**SubmitDate**: 2024-05-14    [abs](http://arxiv.org/abs/2405.08317v1) [paper-pdf](http://arxiv.org/pdf/2405.08317v1)

**Authors**: Raghuveer Peri, Sai Muralidhar Jayanthi, Srikanth Ronanki, Anshu Bhatia, Karel Mundnich, Saket Dingliwal, Nilaksh Das, Zejiang Hou, Goeric Huybrechts, Srikanth Vishnubhotla, Daniel Garcia-Romero, Sundararajan Srinivasan, Kyu J Han, Katrin Kirchhoff

**Abstract**: Integrated Speech and Large Language Models (SLMs) that can follow speech instructions and generate relevant text responses have gained popularity lately. However, the safety and robustness of these models remains largely unclear. In this work, we investigate the potential vulnerabilities of such instruction-following speech-language models to adversarial attacks and jailbreaking. Specifically, we design algorithms that can generate adversarial examples to jailbreak SLMs in both white-box and black-box attack settings without human involvement. Additionally, we propose countermeasures to thwart such jailbreaking attacks. Our models, trained on dialog data with speech instructions, achieve state-of-the-art performance on spoken question-answering task, scoring over 80% on both safety and helpfulness metrics. Despite safety guardrails, experiments on jailbreaking demonstrate the vulnerability of SLMs to adversarial perturbations and transfer attacks, with average attack success rates of 90% and 10% respectively when evaluated on a dataset of carefully designed harmful questions spanning 12 different toxic categories. However, we demonstrate that our proposed countermeasures reduce the attack success significantly.



## **27. Adversarial Machine Learning Threats to Spacecraft**

cs.LG

Preprint

**SubmitDate**: 2024-05-14    [abs](http://arxiv.org/abs/2405.08834v1) [paper-pdf](http://arxiv.org/pdf/2405.08834v1)

**Authors**: Rajiv Thummala, Shristi Sharma, Matteo Calabrese, Gregory Falco

**Abstract**: Spacecraft are among the earliest autonomous systems. Their ability to function without a human in the loop have afforded some of humanity's grandest achievements. As reliance on autonomy grows, space vehicles will become increasingly vulnerable to attacks designed to disrupt autonomous processes-especially probabilistic ones based on machine learning. This paper aims to elucidate and demonstrate the threats that adversarial machine learning (AML) capabilities pose to spacecraft. First, an AML threat taxonomy for spacecraft is introduced. Next, we demonstrate the execution of AML attacks against spacecraft through experimental simulations using NASA's Core Flight System (cFS) and NASA's On-board Artificial Intelligence Research (OnAIR) Platform. Our findings highlight the imperative for incorporating AML-focused security measures in spacecraft that engage autonomy.



## **28. Adversarial Nibbler: An Open Red-Teaming Method for Identifying Diverse Harms in Text-to-Image Generation**

cs.CY

10 pages, 6 figures

**SubmitDate**: 2024-05-14    [abs](http://arxiv.org/abs/2403.12075v3) [paper-pdf](http://arxiv.org/pdf/2403.12075v3)

**Authors**: Jessica Quaye, Alicia Parrish, Oana Inel, Charvi Rastogi, Hannah Rose Kirk, Minsuk Kahng, Erin van Liemt, Max Bartolo, Jess Tsang, Justin White, Nathan Clement, Rafael Mosquera, Juan Ciro, Vijay Janapa Reddi, Lora Aroyo

**Abstract**: With the rise of text-to-image (T2I) generative AI models reaching wide audiences, it is critical to evaluate model robustness against non-obvious attacks to mitigate the generation of offensive images. By focusing on ``implicitly adversarial'' prompts (those that trigger T2I models to generate unsafe images for non-obvious reasons), we isolate a set of difficult safety issues that human creativity is well-suited to uncover. To this end, we built the Adversarial Nibbler Challenge, a red-teaming methodology for crowdsourcing a diverse set of implicitly adversarial prompts. We have assembled a suite of state-of-the-art T2I models, employed a simple user interface to identify and annotate harms, and engaged diverse populations to capture long-tail safety issues that may be overlooked in standard testing. The challenge is run in consecutive rounds to enable a sustained discovery and analysis of safety pitfalls in T2I models.   In this paper, we present an in-depth account of our methodology, a systematic study of novel attack strategies and discussion of safety failures revealed by challenge participants. We also release a companion visualization tool for easy exploration and derivation of insights from the dataset. The first challenge round resulted in over 10k prompt-image pairs with machine annotations for safety. A subset of 1.5k samples contains rich human annotations of harm types and attack styles. We find that 14% of images that humans consider harmful are mislabeled as ``safe'' by machines. We have identified new attack strategies that highlight the complexity of ensuring T2I model robustness. Our findings emphasize the necessity of continual auditing and adaptation as new vulnerabilities emerge. We are confident that this work will enable proactive, iterative safety assessments and promote responsible development of T2I models.



## **29. RAID: A Shared Benchmark for Robust Evaluation of Machine-Generated Text Detectors**

cs.CL

To appear at ACL 2024

**SubmitDate**: 2024-05-13    [abs](http://arxiv.org/abs/2405.07940v1) [paper-pdf](http://arxiv.org/pdf/2405.07940v1)

**Authors**: Liam Dugan, Alyssa Hwang, Filip Trhlik, Josh Magnus Ludan, Andrew Zhu, Hainiu Xu, Daphne Ippolito, Chris Callison-Burch

**Abstract**: Many commercial and open-source models claim to detect machine-generated text with very high accuracy (99\% or higher). However, very few of these detectors are evaluated on shared benchmark datasets and even when they are, the datasets used for evaluation are insufficiently challenging -- lacking variations in sampling strategy, adversarial attacks, and open-source generative models. In this work we present RAID: the largest and most challenging benchmark dataset for machine-generated text detection. RAID includes over 6 million generations spanning 11 models, 8 domains, 11 adversarial attacks and 4 decoding strategies. Using RAID, we evaluate the out-of-domain and adversarial robustness of 8 open- and 4 closed-source detectors and find that current detectors are easily fooled by adversarial attacks, variations in sampling strategies, repetition penalties, and unseen generative models. We release our dataset and tools to encourage further exploration into detector robustness.



## **30. On the Adversarial Robustness of Learning-based Image Compression Against Rate-Distortion Attacks**

eess.IV

**SubmitDate**: 2024-05-13    [abs](http://arxiv.org/abs/2405.07717v1) [paper-pdf](http://arxiv.org/pdf/2405.07717v1)

**Authors**: Chenhao Wu, Qingbo Wu, Haoran Wei, Shuai Chen, Lei Wang, King Ngi Ngan, Fanman Meng, Hongliang Li

**Abstract**: Despite demonstrating superior rate-distortion (RD) performance, learning-based image compression (LIC) algorithms have been found to be vulnerable to malicious perturbations in recent studies. Adversarial samples in these studies are designed to attack only one dimension of either bitrate or distortion, targeting a submodel with a specific compression ratio. However, adversaries in real-world scenarios are neither confined to singular dimensional attacks nor always have control over compression ratios. This variability highlights the inadequacy of existing research in comprehensively assessing the adversarial robustness of LIC algorithms in practical applications. To tackle this issue, this paper presents two joint rate-distortion attack paradigms at both submodel and algorithm levels, i.e., Specific-ratio Rate-Distortion Attack (SRDA) and Agnostic-ratio Rate-Distortion Attack (ARDA). Additionally, a suite of multi-granularity assessment tools is introduced to evaluate the attack results from various perspectives. On this basis, extensive experiments on eight prominent LIC algorithms are conducted to offer a thorough analysis of their inherent vulnerabilities. Furthermore, we explore the efficacy of two defense techniques in improving the performance under joint rate-distortion attacks. The findings from these experiments can provide a valuable reference for the development of compression algorithms with enhanced adversarial robustness.



## **31. DP-DCAN: Differentially Private Deep Contrastive Autoencoder Network for Single-cell Clustering**

cs.LG

**SubmitDate**: 2024-05-13    [abs](http://arxiv.org/abs/2311.03410v2) [paper-pdf](http://arxiv.org/pdf/2311.03410v2)

**Authors**: Huifa Li, Jie Fu, Zhili Chen, Xiaomin Yang, Haitao Liu, Xinpeng Ling

**Abstract**: Single-cell RNA sequencing (scRNA-seq) is important to transcriptomic analysis of gene expression. Recently, deep learning has facilitated the analysis of high-dimensional single-cell data. Unfortunately, deep learning models may leak sensitive information about users. As a result, Differential Privacy (DP) is increasingly used to protect privacy. However, existing DP methods usually perturb whole neural networks to achieve differential privacy, and hence result in great performance overheads. To address this challenge, in this paper, we take advantage of the uniqueness of the autoencoder that it outputs only the dimension-reduced vector in the middle of the network, and design a Differentially Private Deep Contrastive Autoencoder Network (DP-DCAN) by partial network perturbation for single-cell clustering. Since only partial network is added with noise, the performance improvement is obvious and twofold: one part of network is trained with less noise due to a bigger privacy budget, and the other part is trained without any noise. Experimental results of six datasets have verified that DP-DCAN is superior to the traditional DP scheme with whole network perturbation. Moreover, DP-DCAN demonstrates strong robustness to adversarial attacks.



## **32. CrossCert: A Cross-Checking Detection Approach to Patch Robustness Certification for Deep Learning Models**

cs.SE

23 pages, 2 figures, accepted by FSE 2024 (The ACM International  Conference on the Foundations of Software Engineering)

**SubmitDate**: 2024-05-13    [abs](http://arxiv.org/abs/2405.07668v1) [paper-pdf](http://arxiv.org/pdf/2405.07668v1)

**Authors**: Qilin Zhou, Zhengyuan Wei, Haipeng Wang, Bo Jiang, W. K. Chan

**Abstract**: Patch robustness certification is an emerging kind of defense technique against adversarial patch attacks with provable guarantees. There are two research lines: certified recovery and certified detection. They aim to label malicious samples with provable guarantees correctly and issue warnings for malicious samples predicted to non-benign labels with provable guarantees, respectively. However, existing certified detection defenders suffer from protecting labels subject to manipulation, and existing certified recovery defenders cannot systematically warn samples about their labels. A certified defense that simultaneously offers robust labels and systematic warning protection against patch attacks is desirable. This paper proposes a novel certified defense technique called CrossCert. CrossCert formulates a novel approach by cross-checking two certified recovery defenders to provide unwavering certification and detection certification. Unwavering certification ensures that a certified sample, when subjected to a patched perturbation, will always be returned with a benign label without triggering any warnings with a provable guarantee. To our knowledge, CrossCert is the first certified detection technique to offer this guarantee. Our experiments show that, with a slightly lower performance than ViP and comparable performance with PatchCensor in terms of detection certification, CrossCert certifies a significant proportion of samples with the guarantee of unwavering certification.



## **33. Backdoor Removal for Generative Large Language Models**

cs.CR

**SubmitDate**: 2024-05-13    [abs](http://arxiv.org/abs/2405.07667v1) [paper-pdf](http://arxiv.org/pdf/2405.07667v1)

**Authors**: Haoran Li, Yulin Chen, Zihao Zheng, Qi Hu, Chunkit Chan, Heshan Liu, Yangqiu Song

**Abstract**: With rapid advances, generative large language models (LLMs) dominate various Natural Language Processing (NLP) tasks from understanding to reasoning. Yet, language models' inherent vulnerabilities may be exacerbated due to increased accessibility and unrestricted model training on massive textual data from the Internet. A malicious adversary may publish poisoned data online and conduct backdoor attacks on the victim LLMs pre-trained on the poisoned data. Backdoored LLMs behave innocuously for normal queries and generate harmful responses when the backdoor trigger is activated. Despite significant efforts paid to LLMs' safety issues, LLMs are still struggling against backdoor attacks. As Anthropic recently revealed, existing safety training strategies, including supervised fine-tuning (SFT) and Reinforcement Learning from Human Feedback (RLHF), fail to revoke the backdoors once the LLM is backdoored during the pre-training stage. In this paper, we present Simulate and Eliminate (SANDE) to erase the undesired backdoored mappings for generative LLMs. We initially propose Overwrite Supervised Fine-tuning (OSFT) for effective backdoor removal when the trigger is known. Then, to handle the scenarios where the trigger patterns are unknown, we integrate OSFT into our two-stage framework, SANDE. Unlike previous works that center on the identification of backdoors, our safety-enhanced LLMs are able to behave normally even when the exact triggers are activated. We conduct comprehensive experiments to show that our proposed SANDE is effective against backdoor attacks while bringing minimal harm to LLMs' powerful capability without any additional access to unbackdoored clean models. We will release the reproducible code.



## **34. Environmental Matching Attack Against Unmanned Aerial Vehicles Object Detection**

cs.CV

**SubmitDate**: 2024-05-13    [abs](http://arxiv.org/abs/2405.07595v1) [paper-pdf](http://arxiv.org/pdf/2405.07595v1)

**Authors**: Dehong Kong, Siyuan Liang, Wenqi Ren

**Abstract**: Object detection techniques for Unmanned Aerial Vehicles (UAVs) rely on Deep Neural Networks (DNNs), which are vulnerable to adversarial attacks. Nonetheless, adversarial patches generated by existing algorithms in the UAV domain pay very little attention to the naturalness of adversarial patches. Moreover, imposing constraints directly on adversarial patches makes it difficult to generate patches that appear natural to the human eye while ensuring a high attack success rate. We notice that patches are natural looking when their overall color is consistent with the environment. Therefore, we propose a new method named Environmental Matching Attack(EMA) to address the issue of optimizing the adversarial patch under the constraints of color. To the best of our knowledge, this paper is the first to consider natural patches in the domain of UAVs. The EMA method exploits strong prior knowledge of a pretrained stable diffusion to guide the optimization direction of the adversarial patch, where the text guidance can restrict the color of the patch. To better match the environment, the contrast and brightness of the patch are appropriately adjusted. Instead of optimizing the adversarial patch itself, we optimize an adversarial perturbation patch which initializes to zero so that the model can better trade off attacking performance and naturalness. Experiments conducted on the DroneVehicle and Carpk datasets have shown that our work can reach nearly the same attack performance in the digital attack(no greater than 2 in mAP$\%$), surpass the baseline method in the physical specific scenarios, and exhibit a significant advantage in terms of naturalness in visualization and color difference with the environment.



## **35. Towards Rational Consensus in Honest Majority**

cs.GT

**SubmitDate**: 2024-05-13    [abs](http://arxiv.org/abs/2405.07557v1) [paper-pdf](http://arxiv.org/pdf/2405.07557v1)

**Authors**: Varul Srivastava, Sujit Gujar

**Abstract**: Distributed consensus protocols reach agreement among $n$ players in the presence of $f$ adversaries; different protocols support different values of $f$. Existing works study this problem for different adversary types (captured by threat models). There are three primary threat models: (i) Crash fault tolerance (CFT), (ii) Byzantine fault tolerance (BFT), and (iii) Rational fault tolerance (RFT), each more general than the previous. Agreement in repeated rounds on both (1) the proposed value in each round and (2) the ordering among agreed-upon values across multiple rounds is called Atomic BroadCast (ABC). ABC is more generalized than consensus and is employed in blockchains.   This work studies ABC under the RFT threat model. We consider $t$ byzantine and $k$ rational adversaries among $n$ players. We also study different types of rational players based on their utility towards (1) liveness attack, (2) censorship or (3) disagreement (forking attack). We study the problem of ABC under this general threat model in partially-synchronous networks. We show (1) ABC is impossible for $n/3< (t+k) <n/2$ if rational players prefer liveness or censorship attacks and (2) the consensus protocol proposed by Ranchal-Pedrosa and Gramoli cannot be generalized to solve ABC due to insecure Nash equilibrium (resulting in disagreement). For ABC in partially synchronous network settings, we propose a novel protocol \textsf{pRFT}(practical Rational Fault Tolerance). We show \textsf{pRFT} achieves ABC if (a) rational players prefer only disagreement attacks and (b) $t < \frac{n}{4}$ and $(t + k) < \frac{n}{2}$. In \textsf{pRFT}, we incorporate accountability (capturing deviating players) within the protocol by leveraging honest players. We also show that the message complexity of \textsf{pRFT} is at par with the best consensus protocols that guarantee accountability.



## **36. On Securing Analog Lagrange Coded Computing from Colluding Adversaries**

cs.IT

To appear in the proceedings of IEEE ISIT 2024

**SubmitDate**: 2024-05-13    [abs](http://arxiv.org/abs/2405.07454v1) [paper-pdf](http://arxiv.org/pdf/2405.07454v1)

**Authors**: Rimpi Borah, J. Harshan

**Abstract**: Analog Lagrange Coded Computing (ALCC) is a recently proposed coded computing paradigm wherein certain computations over analog datasets can be efficiently performed using distributed worker nodes through floating point implementation. While ALCC is known to preserve privacy of data from the workers, it is not resilient to adversarial workers that return erroneous computation results. Pointing at this security vulnerability, we focus on securing ALCC from a wide range of non-colluding and colluding adversarial workers. As a foundational step, we make use of error-correction algorithms for Discrete Fourier Transform (DFT) codes to build novel algorithms to nullify the erroneous computations returned from the adversaries. Furthermore, when such a robust ALCC is implemented in practical settings, we show that the presence of precision errors in the system can be exploited by the adversaries to propose novel colluding attacks to degrade the computation accuracy. As the main takeaway, we prove a counter-intuitive result that not all the adversaries should inject noise in their computations in order to optimally degrade the accuracy of the ALCC framework. This is the first work of its kind to address the vulnerability of ALCC against colluding adversaries.



## **37. Universal Coding for Shannon Ciphers under Side-Channel Attacks**

cs.IT

6 pages, 3 figures. arXiv admin note: substantial text overlap with  arXiv:1801.02563, arXiv:2201.11670, arXiv:1901.05940

**SubmitDate**: 2024-05-13    [abs](http://arxiv.org/abs/2302.01314v3) [paper-pdf](http://arxiv.org/pdf/2302.01314v3)

**Authors**: Yasutada Oohama, Bagus Santoso

**Abstract**: We study the universal coding under side-channel attacks posed and investigated by Oohama and Santoso (2022). They proposed a theoretical security model for Shannon cipher system under side-channel attacks, where the adversary is not only allowed to collect ciphertexts by eavesdropping the public communication channel, but is also allowed to collect the physical information leaked by the devices where the cipher system is implemented on such as running time, power consumption, electromagnetic radiation, etc. For any distributions of the plain text, any noisy channels through which the adversary observe the corrupted version of the key, and any measurement device used for collecting the physical information, we can derive an achievable rate region for reliability and security such that if we compress the ciphertext with rate within the achievable rate region, then: (1) anyone with secret key will be able to decrypt and decode the ciphertext correctly, but (2) any adversary who obtains the ciphertext and also the side physical information will not be able to obtain any information about the hidden source as long as the leaked physical information is encoded with a rate within the rate region.



## **38. The Janus Interface: How Fine-Tuning in Large Language Models Amplifies the Privacy Risks**

cs.CR

**SubmitDate**: 2024-05-12    [abs](http://arxiv.org/abs/2310.15469v2) [paper-pdf](http://arxiv.org/pdf/2310.15469v2)

**Authors**: Xiaoyi Chen, Siyuan Tang, Rui Zhu, Shijun Yan, Lei Jin, Zihao Wang, Liya Su, Zhikun Zhang, XiaoFeng Wang, Haixu Tang

**Abstract**: The rapid advancements of large language models (LLMs) have raised public concerns about the privacy leakage of personally identifiable information (PII) within their extensive training datasets. Recent studies have demonstrated that an adversary could extract highly sensitive privacy data from the training data of LLMs with carefully designed prompts. However, these attacks suffer from the model's tendency to hallucinate and catastrophic forgetting (CF) in the pre-training stage, rendering the veracity of divulged PIIs negligible. In our research, we propose a novel attack, Janus, which exploits the fine-tuning interface to recover forgotten PIIs from the pre-training data in LLMs. We formalize the privacy leakage problem in LLMs and explain why forgotten PIIs can be recovered through empirical analysis on open-source language models. Based upon these insights, we evaluate the performance of Janus on both open-source language models and two latest LLMs, i.e., GPT-3.5-Turbo and LLaMA-2-7b. Our experiment results show that Janus amplifies the privacy risks by over 10 times in comparison with the baseline and significantly outperforms the state-of-the-art privacy extraction attacks including prefix attacks and in-context learning (ICL). Furthermore, our analysis validates that existing fine-tuning APIs provided by OpenAI and Azure AI Studio are susceptible to our Janus attack, allowing an adversary to conduct such an attack at a low cost.



## **39. Synthesizing Iris Images using Generative Adversarial Networks: Survey and Comparative Analysis**

cs.CV

**SubmitDate**: 2024-05-11    [abs](http://arxiv.org/abs/2404.17105v2) [paper-pdf](http://arxiv.org/pdf/2404.17105v2)

**Authors**: Shivangi Yadav, Arun Ross

**Abstract**: Biometric systems based on iris recognition are currently being used in border control applications and mobile devices. However, research in iris recognition is stymied by various factors such as limited datasets of bonafide irides and presentation attack instruments; restricted intra-class variations; and privacy concerns. Some of these issues can be mitigated by the use of synthetic iris data. In this paper, we present a comprehensive review of state-of-the-art GAN-based synthetic iris image generation techniques, evaluating their strengths and limitations in producing realistic and useful iris images that can be used for both training and testing iris recognition systems and presentation attack detectors. In this regard, we first survey the various methods that have been used for synthetic iris generation and specifically consider generators based on StyleGAN, RaSGAN, CIT-GAN, iWarpGAN, StarGAN, etc. We then analyze the images generated by these models for realism, uniqueness, and biometric utility. This comprehensive analysis highlights the pros and cons of various GANs in the context of developing robust iris matchers and presentation attack detectors.



## **40. Tree Proof-of-Position Algorithms**

cs.DS

**SubmitDate**: 2024-05-10    [abs](http://arxiv.org/abs/2405.06761v1) [paper-pdf](http://arxiv.org/pdf/2405.06761v1)

**Authors**: Aida Manzano Kharman, Pietro Ferraro, Homayoun Hamedmoghadam, Robert Shorten

**Abstract**: We present a novel class of proof-of-position algorithms: Tree-Proof-of-Position (T-PoP). This algorithm is decentralised, collaborative and can be computed in a privacy preserving manner, such that agents do not need to reveal their position publicly. We make no assumptions of honest behaviour in the system, and consider varying ways in which agents may misbehave. Our algorithm is therefore resilient to highly adversarial scenarios. This makes it suitable for a wide class of applications, namely those in which trust in a centralised infrastructure may not be assumed, or high security risk scenarios. Our algorithm has a worst case quadratic runtime, making it suitable for hardware constrained IoT applications. We also provide a mathematical model that summarises T-PoP's performance for varying operating conditions. We then simulate T-PoP's behaviour with a large number of agent-based simulations, which are in complete agreement with our mathematical model, thus demonstrating its validity. T-PoP can achieve high levels of reliability and security by tuning its operating conditions, both in high and low density environments. Finally, we also present a mathematical model to probabilistically detect platooning attacks.



## **41. Certified $\ell_2$ Attribution Robustness via Uniformly Smoothed Attributions**

cs.LG

**SubmitDate**: 2024-05-10    [abs](http://arxiv.org/abs/2405.06361v1) [paper-pdf](http://arxiv.org/pdf/2405.06361v1)

**Authors**: Fan Wang, Adams Wai-Kin Kong

**Abstract**: Model attribution is a popular tool to explain the rationales behind model predictions. However, recent work suggests that the attributions are vulnerable to minute perturbations, which can be added to input samples to fool the attributions while maintaining the prediction outputs. Although empirical studies have shown positive performance via adversarial training, an effective certified defense method is eminently needed to understand the robustness of attributions. In this work, we propose to use uniform smoothing technique that augments the vanilla attributions by noises uniformly sampled from a certain space. It is proved that, for all perturbations within the attack region, the cosine similarity between uniformly smoothed attribution of perturbed sample and the unperturbed sample is guaranteed to be lower bounded. We also derive alternative formulations of the certification that is equivalent to the original one and provides the maximum size of perturbation or the minimum smoothing radius such that the attribution can not be perturbed. We evaluate the proposed method on three datasets and show that the proposed method can effectively protect the attributions from attacks, regardless of the architecture of networks, training schemes and the size of the datasets.



## **42. Evaluating Adversarial Robustness in the Spatial Frequency Domain**

cs.CV

14 pages

**SubmitDate**: 2024-05-10    [abs](http://arxiv.org/abs/2405.06345v1) [paper-pdf](http://arxiv.org/pdf/2405.06345v1)

**Authors**: Keng-Hsin Liao, Chin-Yuan Yeh, Hsi-Wen Chen, Ming-Syan Chen

**Abstract**: Convolutional Neural Networks (CNNs) have dominated the majority of computer vision tasks. However, CNNs' vulnerability to adversarial attacks has raised concerns about deploying these models to safety-critical applications. In contrast, the Human Visual System (HVS), which utilizes spatial frequency channels to process visual signals, is immune to adversarial attacks. As such, this paper presents an empirical study exploring the vulnerability of CNN models in the frequency domain. Specifically, we utilize the discrete cosine transform (DCT) to construct the Spatial-Frequency (SF) layer to produce a block-wise frequency spectrum of an input image and formulate Spatial Frequency CNNs (SF-CNNs) by replacing the initial feature extraction layers of widely-used CNN backbones with the SF layer. Through extensive experiments, we observe that SF-CNN models are more robust than their CNN counterparts under both white-box and black-box attacks. To further explain the robustness of SF-CNNs, we compare the SF layer with a trainable convolutional layer with identical kernel sizes using two mixing strategies to show that the lower frequency components contribute the most to the adversarial robustness of SF-CNNs. We believe our observations can guide the future design of robust CNN models.



## **43. Improving Transferable Targeted Adversarial Attack via Normalized Logit Calibration and Truncated Feature Mixing**

cs.CV

**SubmitDate**: 2024-05-10    [abs](http://arxiv.org/abs/2405.06340v1) [paper-pdf](http://arxiv.org/pdf/2405.06340v1)

**Authors**: Juanjuan Weng, Zhiming Luo, Shaozi Li

**Abstract**: This paper aims to enhance the transferability of adversarial samples in targeted attacks, where attack success rates remain comparatively low. To achieve this objective, we propose two distinct techniques for improving the targeted transferability from the loss and feature aspects. First, in previous approaches, logit calibrations used in targeted attacks primarily focus on the logit margin between the targeted class and the untargeted classes among samples, neglecting the standard deviation of the logit. In contrast, we introduce a new normalized logit calibration method that jointly considers the logit margin and the standard deviation of logits. This approach effectively calibrates the logits, enhancing the targeted transferability. Second, previous studies have demonstrated that mixing the features of clean samples during optimization can significantly increase transferability. Building upon this, we further investigate a truncated feature mixing method to reduce the impact of the source training model, resulting in additional improvements. The truncated feature is determined by removing the Rank-1 feature associated with the largest singular value decomposed from the high-level convolutional layers of the clean sample. Extensive experiments conducted on the ImageNet-Compatible and CIFAR-10 datasets demonstrate the individual and mutual benefits of our proposed two components, which outperform the state-of-the-art methods by a large margin in black-box targeted attacks.



## **44. PUMA: margin-based data pruning**

cs.LG

**SubmitDate**: 2024-05-10    [abs](http://arxiv.org/abs/2405.06298v1) [paper-pdf](http://arxiv.org/pdf/2405.06298v1)

**Authors**: Javier Maroto, Pascal Frossard

**Abstract**: Deep learning has been able to outperform humans in terms of classification accuracy in many tasks. However, to achieve robustness to adversarial perturbations, the best methodologies require to perform adversarial training on a much larger training set that has been typically augmented using generative models (e.g., diffusion models). Our main objective in this work, is to reduce these data requirements while achieving the same or better accuracy-robustness trade-offs. We focus on data pruning, where some training samples are removed based on the distance to the model classification boundary (i.e., margin). We find that the existing approaches that prune samples with low margin fails to increase robustness when we add a lot of synthetic data, and explain this situation with a perceptron learning task. Moreover, we find that pruning high margin samples for better accuracy increases the harmful impact of mislabeled perturbed data in adversarial training, hurting both robustness and accuracy. We thus propose PUMA, a new data pruning strategy that computes the margin using DeepFool, and prunes the training samples of highest margin without hurting performance by jointly adjusting the training attack norm on the samples of lowest margin. We show that PUMA can be used on top of the current state-of-the-art methodology in robustness, and it is able to significantly improve the model performance unlike the existing data pruning strategies. Not only PUMA achieves similar robustness with less data, but it also significantly increases the model accuracy, improving the performance trade-off.



## **45. Exploring the Interplay of Interpretability and Robustness in Deep Neural Networks: A Saliency-guided Approach**

cs.CV

**SubmitDate**: 2024-05-10    [abs](http://arxiv.org/abs/2405.06278v1) [paper-pdf](http://arxiv.org/pdf/2405.06278v1)

**Authors**: Amira Guesmi, Nishant Suresh Aswani, Muhammad Shafique

**Abstract**: Adversarial attacks pose a significant challenge to deploying deep learning models in safety-critical applications. Maintaining model robustness while ensuring interpretability is vital for fostering trust and comprehension in these models. This study investigates the impact of Saliency-guided Training (SGT) on model robustness, a technique aimed at improving the clarity of saliency maps to deepen understanding of the model's decision-making process. Experiments were conducted on standard benchmark datasets using various deep learning architectures trained with and without SGT. Findings demonstrate that SGT enhances both model robustness and interpretability. Additionally, we propose a novel approach combining SGT with standard adversarial training to achieve even greater robustness while preserving saliency map quality. Our strategy is grounded in the assumption that preserving salient features crucial for correctly classifying adversarial examples enhances model robustness, while masking non-relevant features improves interpretability. Our technique yields significant gains, achieving a 35\% and 20\% improvement in robustness against PGD attack with noise magnitudes of $0.2$ and $0.02$ for the MNIST and CIFAR-10 datasets, respectively, while producing high-quality saliency maps.



## **46. Disttack: Graph Adversarial Attacks Toward Distributed GNN Training**

cs.LG

Accepted by 30th International European Conference on Parallel and  Distributed Computing(Euro-Par 2024)

**SubmitDate**: 2024-05-10    [abs](http://arxiv.org/abs/2405.06247v1) [paper-pdf](http://arxiv.org/pdf/2405.06247v1)

**Authors**: Yuxiang Zhang, Xin Liu, Meng Wu, Wei Yan, Mingyu Yan, Xiaochun Ye, Dongrui Fan

**Abstract**: Graph Neural Networks (GNNs) have emerged as potent models for graph learning. Distributing the training process across multiple computing nodes is the most promising solution to address the challenges of ever-growing real-world graphs. However, current adversarial attack methods on GNNs neglect the characteristics and applications of the distributed scenario, leading to suboptimal performance and inefficiency in attacking distributed GNN training.   In this study, we introduce Disttack, the first framework of adversarial attacks for distributed GNN training that leverages the characteristics of frequent gradient updates in a distributed system. Specifically, Disttack corrupts distributed GNN training by injecting adversarial attacks into one single computing node. The attacked subgraphs are precisely perturbed to induce an abnormal gradient ascent in backpropagation, disrupting gradient synchronization between computing nodes and thus leading to a significant performance decline of the trained GNN. We evaluate Disttack on four large real-world graphs by attacking five widely adopted GNNs. Compared with the state-of-the-art attack method, experimental results demonstrate that Disttack amplifies the model accuracy degradation by 2.75$\times$ and achieves speedup by 17.33$\times$ on average while maintaining unnoticeability.



## **47. Concealing Backdoor Model Updates in Federated Learning by Trigger-Optimized Data Poisoning**

cs.CR

**SubmitDate**: 2024-05-10    [abs](http://arxiv.org/abs/2405.06206v1) [paper-pdf](http://arxiv.org/pdf/2405.06206v1)

**Authors**: Yujie Zhang, Neil Gong, Michael K. Reiter

**Abstract**: Federated Learning (FL) is a decentralized machine learning method that enables participants to collaboratively train a model without sharing their private data. Despite its privacy and scalability benefits, FL is susceptible to backdoor attacks, where adversaries poison the local training data of a subset of clients using a backdoor trigger, aiming to make the aggregated model produce malicious results when the same backdoor condition is met by an inference-time input. Existing backdoor attacks in FL suffer from common deficiencies: fixed trigger patterns and reliance on the assistance of model poisoning. State-of-the-art defenses based on Byzantine-robust aggregation exhibit a good defense performance on these attacks because of the significant divergence between malicious and benign model updates. To effectively conceal malicious model updates among benign ones, we propose DPOT, a backdoor attack strategy in FL that dynamically constructs backdoor objectives by optimizing a backdoor trigger, making backdoor data have minimal effect on model updates. We provide theoretical justifications for DPOT's attacking principle and display experimental results showing that DPOT, via only a data-poisoning attack, effectively undermines state-of-the-art defenses and outperforms existing backdoor attack techniques on various datasets.



## **48. Muting Whisper: A Universal Acoustic Adversarial Attack on Speech Foundation Models**

cs.CL

**SubmitDate**: 2024-05-09    [abs](http://arxiv.org/abs/2405.06134v1) [paper-pdf](http://arxiv.org/pdf/2405.06134v1)

**Authors**: Vyas Raina, Rao Ma, Charles McGhee, Kate Knill, Mark Gales

**Abstract**: Recent developments in large speech foundation models like Whisper have led to their widespread use in many automatic speech recognition (ASR) applications. These systems incorporate `special tokens' in their vocabulary, such as $\texttt{<endoftext>}$, to guide their language generation process. However, we demonstrate that these tokens can be exploited by adversarial attacks to manipulate the model's behavior. We propose a simple yet effective method to learn a universal acoustic realization of Whisper's $\texttt{<endoftext>}$ token, which, when prepended to any speech signal, encourages the model to ignore the speech and only transcribe the special token, effectively `muting' the model. Our experiments demonstrate that the same, universal 0.64-second adversarial audio segment can successfully mute a target Whisper ASR model for over 97\% of speech samples. Moreover, we find that this universal adversarial audio segment often transfers to new datasets and tasks. Overall this work demonstrates the vulnerability of Whisper models to `muting' adversarial attacks, where such attacks can pose both risks and potential benefits in real-world settings: for example the attack can be used to bypass speech moderation systems, or conversely the attack can also be used to protect private speech data.



## **49. Hard Work Does Not Always Pay Off: Poisoning Attacks on Neural Architecture Search**

cs.LG

**SubmitDate**: 2024-05-09    [abs](http://arxiv.org/abs/2405.06073v1) [paper-pdf](http://arxiv.org/pdf/2405.06073v1)

**Authors**: Zachary Coalson, Huazheng Wang, Qingyun Wu, Sanghyun Hong

**Abstract**: In this paper, we study the robustness of "data-centric" approaches to finding neural network architectures (known as neural architecture search) to data distribution shifts. To audit this robustness, we present a data poisoning attack, when injected to the training data used for architecture search that can prevent the victim algorithm from finding an architecture with optimal accuracy. We first define the attack objective for crafting poisoning samples that can induce the victim to generate sub-optimal architectures. To this end, we weaponize existing search algorithms to generate adversarial architectures that serve as our objectives. We also present techniques that the attacker can use to significantly reduce the computational costs of crafting poisoning samples. In an extensive evaluation of our poisoning attack on a representative architecture search algorithm, we show its surprising robustness. Because our attack employs clean-label poisoning, we also evaluate its robustness against label noise. We find that random label-flipping is more effective in generating sub-optimal architectures than our clean-label attack. Our results suggests that care must be taken for the data this emerging approach uses, and future work is needed to develop robust algorithms.



## **50. BB-Patch: BlackBox Adversarial Patch-Attack using Zeroth-Order Optimization**

cs.CV

**SubmitDate**: 2024-05-09    [abs](http://arxiv.org/abs/2405.06049v1) [paper-pdf](http://arxiv.org/pdf/2405.06049v1)

**Authors**: Satyadwyoom Kumar, Saurabh Gupta, Arun Balaji Buduru

**Abstract**: Deep Learning has become popular due to its vast applications in almost all domains. However, models trained using deep learning are prone to failure for adversarial samples and carry a considerable risk in sensitive applications. Most of these adversarial attack strategies assume that the adversary has access to the training data, the model parameters, and the input during deployment, hence, focus on perturbing the pixel level information present in the input image.   Adversarial Patches were introduced to the community which helped in bringing out the vulnerability of deep learning models in a much more pragmatic manner but here the attacker has a white-box access to the model parameters. Recently, there has been an attempt to develop these adversarial attacks using black-box techniques. However, certain assumptions such as availability large training data is not valid for a real-life scenarios. In a real-life scenario, the attacker can only assume the type of model architecture used from a select list of state-of-the-art architectures while having access to only a subset of input dataset. Hence, we propose an black-box adversarial attack strategy that produces adversarial patches which can be applied anywhere in the input image to perform an adversarial attack.



