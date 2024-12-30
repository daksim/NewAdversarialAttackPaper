# Latest Adversarial Attack Papers
**update at 2024-12-30 10:07:18**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

[Attacks and Defenses in Large language Models](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM.md)

## **1. An Engorgio Prompt Makes Large Language Model Babble on**

cs.CR

**SubmitDate**: 2024-12-27    [abs](http://arxiv.org/abs/2412.19394v1) [paper-pdf](http://arxiv.org/pdf/2412.19394v1)

**Authors**: Jianshuo Dong, Ziyuan Zhang, Qingjie Zhang, Han Qiu, Tianwei Zhang, Hao Wang, Hewu Li, Qi Li, Chao Zhang, Ke Xu

**Abstract**: Auto-regressive large language models (LLMs) have yielded impressive performance in many real-world tasks. However, the new paradigm of these LLMs also exposes novel threats. In this paper, we explore their vulnerability to inference cost attacks, where a malicious user crafts Engorgio prompts to intentionally increase the computation cost and latency of the inference process. We design Engorgio, a novel methodology, to efficiently generate adversarial Engorgio prompts to affect the target LLM's service availability. Engorgio has the following two technical contributions. (1) We employ a parameterized distribution to track LLMs' prediction trajectory. (2) Targeting the auto-regressive nature of LLMs' inference process, we propose novel loss functions to stably suppress the appearance of the <EOS> token, whose occurrence will interrupt the LLM's generation process. We conduct extensive experiments on 13 open-sourced LLMs with parameters ranging from 125M to 30B. The results show that Engorgio prompts can successfully induce LLMs to generate abnormally long outputs (i.e., roughly 2-13$\times$ longer to reach 90%+ of the output length limit) in a white-box scenario and our real-world experiment demonstrates Engergio's threat to LLM service with limited computing resources. The code is accessible at https://github.com/jianshuod/Engorgio-prompt.



## **2. Quantum-Inspired Weight-Constrained Neural Network: Reducing Variable Numbers by 100x Compared to Standard Neural Networks**

quant-ph

13 pages, 5 figures. Comments are welcome

**SubmitDate**: 2024-12-26    [abs](http://arxiv.org/abs/2412.19355v1) [paper-pdf](http://arxiv.org/pdf/2412.19355v1)

**Authors**: Shaozhi Li, M Sabbir Salek, Binayyak Roy, Yao Wang, Mashrur Chowdhury

**Abstract**: Although quantum machine learning has shown great promise, the practical application of quantum computers remains constrained in the noisy intermediate-scale quantum era. To take advantage of quantum machine learning, we investigate the underlying mathematical principles of these quantum models and adapt them to classical machine learning frameworks. Specifically, we develop a classical weight-constrained neural network that generates weights based on quantum-inspired insights. We find that this approach can reduce the number of variables in a classical neural network by a factor of 135 while preserving its learnability. In addition, we develop a dropout method to enhance the robustness of quantum machine learning models, which are highly susceptible to adversarial attacks. This technique can also be applied to improve the adversarial resilience of the classical weight-constrained neural network, which is essential for industry applications, such as self-driving vehicles. Our work offers a novel approach to reduce the complexity of large classical neural networks, addressing a critical challenge in machine learning.



## **3. Federated Hybrid Training and Self-Adversarial Distillation: Towards Robust Edge Networks**

cs.CV

**SubmitDate**: 2024-12-26    [abs](http://arxiv.org/abs/2412.19354v1) [paper-pdf](http://arxiv.org/pdf/2412.19354v1)

**Authors**: Yu Qiao, Apurba Adhikary, Kitae Kim, Eui-Nam Huh, Zhu Han, Choong Seon Hong

**Abstract**: Federated learning (FL) is a distributed training technology that enhances data privacy in mobile edge networks by allowing data owners to collaborate without transmitting raw data to the edge server. However, data heterogeneity and adversarial attacks pose challenges to develop an unbiased and robust global model for edge deployment. To address this, we propose Federated hyBrid Adversarial training and self-adversarial disTillation (FedBAT), a new framework designed to improve both robustness and generalization of the global model. FedBAT seamlessly integrates hybrid adversarial training and self-adversarial distillation into the conventional FL framework from data augmentation and feature distillation perspectives. From a data augmentation perspective, we propose hybrid adversarial training to defend against adversarial attacks by balancing accuracy and robustness through a weighted combination of standard and adversarial training. From a feature distillation perspective, we introduce a novel augmentation-invariant adversarial distillation method that aligns local adversarial features of augmented images with their corresponding unbiased global clean features. This alignment can effectively mitigate bias from data heterogeneity while enhancing both the robustness and generalization of the global model. Extensive experimental results across multiple datasets demonstrate that FedBAT yields comparable or superior performance gains in improving robustness while maintaining accuracy compared to several baselines.



## **4. xSRL: Safety-Aware Explainable Reinforcement Learning -- Safety as a Product of Explainability**

cs.AI

Accepted to 24th International Conference on Autonomous Agents and  Multiagent Systems (AAMAS 2025)

**SubmitDate**: 2024-12-26    [abs](http://arxiv.org/abs/2412.19311v1) [paper-pdf](http://arxiv.org/pdf/2412.19311v1)

**Authors**: Risal Shahriar Shefin, Md Asifur Rahman, Thai Le, Sarra Alqahtani

**Abstract**: Reinforcement learning (RL) has shown great promise in simulated environments, such as games, where failures have minimal consequences. However, the deployment of RL agents in real-world systems such as autonomous vehicles, robotics, UAVs, and medical devices demands a higher level of safety and transparency, particularly when facing adversarial threats. Safe RL algorithms have been developed to address these concerns by optimizing both task performance and safety constraints. However, errors are inevitable, and when they occur, it is essential that the RL agents can also explain their actions to human operators. This makes trust in the safety mechanisms of RL systems crucial for effective deployment. Explainability plays a key role in building this trust by providing clear, actionable insights into the agent's decision-making process, ensuring that safety-critical decisions are well understood. While machine learning (ML) has seen significant advances in interpretability and visualization, explainability methods for RL remain limited. Current tools fail to address the dynamic, sequential nature of RL and its needs to balance task performance with safety constraints over time. The re-purposing of traditional ML methods, such as saliency maps, is inadequate for safety-critical RL applications where mistakes can result in severe consequences. To bridge this gap, we propose xSRL, a framework that integrates both local and global explanations to provide a comprehensive understanding of RL agents' behavior. xSRL also enables developers to identify policy vulnerabilities through adversarial attacks, offering tools to debug and patch agents without retraining. Our experiments and user studies demonstrate xSRL's effectiveness in increasing safety in RL systems, making them more reliable and trustworthy for real-world deployment. Code is available at https://github.com/risal-shefin/xSRL.



## **5. Game-Theoretically Secure Distributed Protocols for Fair Allocation in Coalitional Games**

cs.GT

**SubmitDate**: 2024-12-26    [abs](http://arxiv.org/abs/2412.19192v1) [paper-pdf](http://arxiv.org/pdf/2412.19192v1)

**Authors**: T-H. Hubert Chan, Qipeng Kuang, Quan Xue

**Abstract**: We consider game-theoretically secure distributed protocols for coalition games that approximate the Shapley value with small multiplicative error. Since all known existing approximation algorithms for the Shapley value are randomized, it is a challenge to design efficient distributed protocols among mutually distrusted players when there is no central authority to generate unbiased randomness. The game-theoretic notion of maximin security has been proposed to offer guarantees to an honest player's reward even if all other players are susceptible to an adversary.   Permutation sampling is often used in approximation algorithms for the Shapley value. A previous work in 1994 by Zlotkin et al. proposed a simple constant-round distributed permutation generation protocol based on commitment scheme, but it is vulnerable to rushing attacks. The protocol, however, can detect such attacks.   In this work, we model the limited resources of an adversary by a violation budget that determines how many times it can perform such detectable attacks. Therefore, by repeating the number of permutation samples, an honest player's reward can be guaranteed to be close to its Shapley value. We explore both high probability and expected maximin security. We obtain an upper bound on the number of permutation samples for high probability maximin security, even with an unknown violation budget. Furthermore, we establish a matching lower bound for the weaker notion of expected maximin security in specific permutation generation protocols. We have also performed experiments on both synthetic and real data to empirically verify our results.



## **6. TSCheater: Generating High-Quality Tibetan Adversarial Texts via Visual Similarity**

cs.CL

Camera-Ready Version; Accepted at ICASSP 2025

**SubmitDate**: 2024-12-26    [abs](http://arxiv.org/abs/2412.02371v3) [paper-pdf](http://arxiv.org/pdf/2412.02371v3)

**Authors**: Xi Cao, Quzong Gesang, Yuan Sun, Nuo Qun, Tashi Nyima

**Abstract**: Language models based on deep neural networks are vulnerable to textual adversarial attacks. While rich-resource languages like English are receiving focused attention, Tibetan, a cross-border language, is gradually being studied due to its abundant ancient literature and critical language strategy. Currently, there are several Tibetan adversarial text generation methods, but they do not fully consider the textual features of Tibetan script and overestimate the quality of generated adversarial texts. To address this issue, we propose a novel Tibetan adversarial text generation method called TSCheater, which considers the characteristic of Tibetan encoding and the feature that visually similar syllables have similar semantics. This method can also be transferred to other abugidas, such as Devanagari script. We utilize a self-constructed Tibetan syllable visual similarity database called TSVSDB to generate substitution candidates and adopt a greedy algorithm-based scoring mechanism to determine substitution order. After that, we conduct the method on eight victim language models. Experimentally, TSCheater outperforms existing methods in attack effectiveness, perturbation magnitude, semantic similarity, visual similarity, and human acceptance. Finally, we construct the first Tibetan adversarial robustness evaluation benchmark called AdvTS, which is generated by existing methods and proofread by humans.



## **7. DiffPatch: Generating Customizable Adversarial Patches using Diffusion Model**

cs.CV

**SubmitDate**: 2024-12-26    [abs](http://arxiv.org/abs/2412.01440v2) [paper-pdf](http://arxiv.org/pdf/2412.01440v2)

**Authors**: Zhixiang Wang, Guangnan Ye, Xiaosen Wang, Siheng Chen, Zhibo Wang, Xingjun Ma, Yu-Gang Jiang

**Abstract**: Physical adversarial patches printed on clothing can easily allow individuals to evade person detectors. However, most existing adversarial patch generation methods prioritize attack effectiveness over stealthiness, resulting in patches that are aesthetically unpleasing. Although existing methods using generative adversarial networks or diffusion models can produce more natural-looking patches, they often struggle to balance stealthiness with attack effectiveness and lack flexibility for user customization. To address these challenges, we propose a novel diffusion-based customizable patch generation framework termed DiffPatch, specifically tailored for creating naturalistic and customizable adversarial patches. Our approach enables users to utilize a reference image as the source, rather than starting from random noise, and incorporates masks to craft naturalistic patches of various shapes, not limited to squares. To prevent the original semantics from being lost during the diffusion process, we employ Null-text inversion to map random noise samples to a single input image and generate patches through Incomplete Diffusion Optimization (IDO). Notably, while maintaining a natural appearance, our method achieves a comparable attack performance to state-of-the-art non-naturalistic patches when using similarly sized attacks. Using DiffPatch, we have created a physical adversarial T-shirt dataset, AdvPatch-1K, specifically targeting YOLOv5s. This dataset includes over a thousand images across diverse scenarios, validating the effectiveness of our attack in real-world environments. Moreover, it provides a valuable resource for future research.



## **8. Provable Robust Saliency-based Explanations**

cs.LG

Accepted to NeurIPS 2024

**SubmitDate**: 2024-12-26    [abs](http://arxiv.org/abs/2212.14106v4) [paper-pdf](http://arxiv.org/pdf/2212.14106v4)

**Authors**: Chao Chen, Chenghua Guo, Rufeng Chen, Guixiang Ma, Ming Zeng, Xiangwen Liao, Xi Zhang, Sihong Xie

**Abstract**: To foster trust in machine learning models, explanations must be faithful and stable for consistent insights. Existing relevant works rely on the $\ell_p$ distance for stability assessment, which diverges from human perception. Besides, existing adversarial training (AT) associated with intensive computations may lead to an arms race. To address these challenges, we introduce a novel metric to assess the stability of top-$k$ salient features. We introduce R2ET which trains for stable explanation by efficient and effective regularizer, and analyze R2ET by multi-objective optimization to prove numerical and statistical stability of explanations. Moreover, theoretical connections between R2ET and certified robustness justify R2ET's stability in all attacks. Extensive experiments across various data modalities and model architectures show that R2ET achieves superior stability against stealthy attacks, and generalizes effectively across different explanation methods.



## **9. Imperceptible Adversarial Attacks on Point Clouds Guided by Point-to-Surface Field**

cs.CV

Accepted by ICASSP 2025

**SubmitDate**: 2024-12-26    [abs](http://arxiv.org/abs/2412.19015v1) [paper-pdf](http://arxiv.org/pdf/2412.19015v1)

**Authors**: Keke Tang, Weiyao Ke, Weilong Peng, Xiaofei Wang, Ziyong Du, Zhize Wu, Peican Zhu, Zhihong Tian

**Abstract**: Adversarial attacks on point clouds are crucial for assessing and improving the adversarial robustness of 3D deep learning models. Traditional solutions strictly limit point displacement during attacks, making it challenging to balance imperceptibility with adversarial effectiveness. In this paper, we attribute the inadequate imperceptibility of adversarial attacks on point clouds to deviations from the underlying surface. To address this, we introduce a novel point-to-surface (P2S) field that adjusts adversarial perturbation directions by dragging points back to their original underlying surface. Specifically, we use a denoising network to learn the gradient field of the logarithmic density function encoding the shape's surface, and apply a distance-aware adjustment to perturbation directions during attacks, thereby enhancing imperceptibility. Extensive experiments show that adversarial attacks guided by our P2S field are more imperceptible, outperforming state-of-the-art methods.



## **10. Bridging Interpretability and Robustness Using LIME-Guided Model Refinement**

cs.LG

10 pages, 15 figures

**SubmitDate**: 2024-12-25    [abs](http://arxiv.org/abs/2412.18952v1) [paper-pdf](http://arxiv.org/pdf/2412.18952v1)

**Authors**: Navid Nayyem, Abdullah Rakin, Longwei Wang

**Abstract**: This paper explores the intricate relationship between interpretability and robustness in deep learning models. Despite their remarkable performance across various tasks, deep learning models often exhibit critical vulnerabilities, including susceptibility to adversarial attacks, over-reliance on spurious correlations, and a lack of transparency in their decision-making processes. To address these limitations, we propose a novel framework that leverages Local Interpretable Model-Agnostic Explanations (LIME) to systematically enhance model robustness. By identifying and mitigating the influence of irrelevant or misleading features, our approach iteratively refines the model, penalizing reliance on these features during training. Empirical evaluations on multiple benchmark datasets demonstrate that LIME-guided refinement not only improves interpretability but also significantly enhances resistance to adversarial perturbations and generalization to out-of-distribution data.



## **11. Improving Integrated Gradient-based Transferable Adversarial Examples by Refining the Integration Path**

cs.CR

Accepted by AAAI 2025

**SubmitDate**: 2024-12-25    [abs](http://arxiv.org/abs/2412.18844v1) [paper-pdf](http://arxiv.org/pdf/2412.18844v1)

**Authors**: Yuchen Ren, Zhengyu Zhao, Chenhao Lin, Bo Yang, Lu Zhou, Zhe Liu, Chao Shen

**Abstract**: Transferable adversarial examples are known to cause threats in practical, black-box attack scenarios. A notable approach to improving transferability is using integrated gradients (IG), originally developed for model interpretability. In this paper, we find that existing IG-based attacks have limited transferability due to their naive adoption of IG in model interpretability. To address this limitation, we focus on the IG integration path and refine it in three aspects: multiplicity, monotonicity, and diversity, supported by theoretical analyses. We propose the Multiple Monotonic Diversified Integrated Gradients (MuMoDIG) attack, which can generate highly transferable adversarial examples on different CNN and ViT models and defenses. Experiments validate that MuMoDIG outperforms the latest IG-based attack by up to 37.3\% and other state-of-the-art attacks by 8.4\%. In general, our study reveals that migrating established techniques to improve transferability may require non-trivial efforts. Code is available at \url{https://github.com/RYC-98/MuMoDIG}.



## **12. Distortion-Aware Adversarial Attacks on Bounding Boxes of Object Detectors**

cs.CV

**SubmitDate**: 2024-12-25    [abs](http://arxiv.org/abs/2412.18815v1) [paper-pdf](http://arxiv.org/pdf/2412.18815v1)

**Authors**: Pham Phuc, Son Vuong, Khang Nguyen, Tuan Dang

**Abstract**: Deep learning-based object detection has become ubiquitous in the last decade due to its high accuracy in many real-world applications. With this growing trend, these models are interested in being attacked by adversaries, with most of the results being on classifiers, which do not match the context of practical object detection. In this work, we propose a novel method to fool object detectors, expose the vulnerability of state-of-the-art detectors, and promote later works to build more robust detectors to adversarial examples. Our method aims to generate adversarial images by perturbing object confidence scores during training, which is crucial in predicting confidence for each class in the testing phase. Herein, we provide a more intuitive technique to embed additive noises based on detected objects' masks and the training loss with distortion control over the original image by leveraging the gradient of iterative images. To verify the proposed method, we perform adversarial attacks against different object detectors, including the most recent state-of-the-art models like YOLOv8, Faster R-CNN, RetinaNet, and Swin Transformer. We also evaluate our technique on MS COCO 2017 and PASCAL VOC 2012 datasets and analyze the trade-off between success attack rate and image distortion. Our experiments show that the achievable success attack rate is up to $100$\% and up to $98$\% when performing white-box and black-box attacks, respectively. The source code and relevant documentation for this work are available at the following link: https://github.com/anonymous20210106/attack_detector



## **13. Protective Perturbations against Unauthorized Data Usage in Diffusion-based Image Generation**

cs.CV

**SubmitDate**: 2024-12-25    [abs](http://arxiv.org/abs/2412.18791v1) [paper-pdf](http://arxiv.org/pdf/2412.18791v1)

**Authors**: Sen Peng, Jijia Yang, Mingyue Wang, Jianfei He, Xiaohua Jia

**Abstract**: Diffusion-based text-to-image models have shown immense potential for various image-related tasks. However, despite their prominence and popularity, customizing these models using unauthorized data also brings serious privacy and intellectual property issues. Existing methods introduce protective perturbations based on adversarial attacks, which are applied to the customization samples. In this systematization of knowledge, we present a comprehensive survey of protective perturbation methods designed to prevent unauthorized data usage in diffusion-based image generation. We establish the threat model and categorize the downstream tasks relevant to these methods, providing a detailed analysis of their designs. We also propose a completed evaluation framework for these perturbation techniques, aiming to advance research in this field.



## **14. Attack-in-the-Chain: Bootstrapping Large Language Models for Attacks Against Black-box Neural Ranking Models**

cs.IR

Accepted by AAAI25

**SubmitDate**: 2024-12-25    [abs](http://arxiv.org/abs/2412.18770v1) [paper-pdf](http://arxiv.org/pdf/2412.18770v1)

**Authors**: Yu-An Liu, Ruqing Zhang, Jiafeng Guo, Maarten de Rijke, Yixing Fan, Xueqi Cheng

**Abstract**: Neural ranking models (NRMs) have been shown to be highly effective in terms of retrieval performance. Unfortunately, they have also displayed a higher degree of sensitivity to attacks than previous generation models. To help expose and address this lack of robustness, we introduce a novel ranking attack framework named Attack-in-the-Chain, which tracks interactions between large language models (LLMs) and NRMs based on chain-of-thought (CoT) prompting to generate adversarial examples under black-box settings. Our approach starts by identifying anchor documents with higher ranking positions than the target document as nodes in the reasoning chain. We then dynamically assign the number of perturbation words to each node and prompt LLMs to execute attacks. Finally, we verify the attack performance of all nodes at each reasoning step and proceed to generate the next reasoning step. Empirical results on two web search benchmarks show the effectiveness of our method.



## **15. Token Highlighter: Inspecting and Mitigating Jailbreak Prompts for Large Language Models**

cs.CR

Accepted by AAAI 2025. Project page:  https://huggingface.co/spaces/TrustSafeAI/Token-Highlighter

**SubmitDate**: 2024-12-25    [abs](http://arxiv.org/abs/2412.18171v2) [paper-pdf](http://arxiv.org/pdf/2412.18171v2)

**Authors**: Xiaomeng Hu, Pin-Yu Chen, Tsung-Yi Ho

**Abstract**: Large Language Models (LLMs) are increasingly being integrated into services such as ChatGPT to provide responses to user queries. To mitigate potential harm and prevent misuse, there have been concerted efforts to align the LLMs with human values and legal compliance by incorporating various techniques, such as Reinforcement Learning from Human Feedback (RLHF), into the training of the LLMs. However, recent research has exposed that even aligned LLMs are susceptible to adversarial manipulations known as Jailbreak Attacks. To address this challenge, this paper proposes a method called Token Highlighter to inspect and mitigate the potential jailbreak threats in the user query. Token Highlighter introduced a concept called Affirmation Loss to measure the LLM's willingness to answer the user query. It then uses the gradient of Affirmation Loss for each token in the user query to locate the jailbreak-critical tokens. Further, Token Highlighter exploits our proposed Soft Removal technique to mitigate the jailbreak effects of critical tokens via shrinking their token embeddings. Experimental results on two aligned LLMs (LLaMA-2 and Vicuna-V1.5) demonstrate that the proposed method can effectively defend against a variety of Jailbreak Attacks while maintaining competent performance on benign questions of the AlpacaEval benchmark. In addition, Token Highlighter is a cost-effective and interpretable defense because it only needs to query the protected LLM once to compute the Affirmation Loss and can highlight the critical tokens upon refusal.



## **16. Evaluating the Adversarial Robustness of Detection Transformers**

cs.CV

**SubmitDate**: 2024-12-25    [abs](http://arxiv.org/abs/2412.18718v1) [paper-pdf](http://arxiv.org/pdf/2412.18718v1)

**Authors**: Amirhossein Nazeri, Chunheng Zhao, Pierluigi Pisu

**Abstract**: Robust object detection is critical for autonomous driving and mobile robotics, where accurate detection of vehicles, pedestrians, and obstacles is essential for ensuring safety. Despite the advancements in object detection transformers (DETRs), their robustness against adversarial attacks remains underexplored. This paper presents a comprehensive evaluation of DETR model and its variants under both white-box and black-box adversarial attacks, using the MS-COCO and KITTI datasets to cover general and autonomous driving scenarios. We extend prominent white-box attack methods (FGSM, PGD, and CW) to assess DETR vulnerability, demonstrating that DETR models are significantly susceptible to adversarial attacks, similar to traditional CNN-based detectors. Our extensive transferability analysis reveals high intra-network transferability among DETR variants, but limited cross-network transferability to CNN-based models. Additionally, we propose a novel untargeted attack designed specifically for DETR, exploiting its intermediate loss functions to induce misclassification with minimal perturbations. Visualizations of self-attention feature maps provide insights into how adversarial attacks affect the internal representations of DETR models. These findings reveal critical vulnerabilities in detection transformers under standard adversarial attacks, emphasizing the need for future research to enhance the robustness of transformer-based object detectors in safety-critical applications.



## **17. SurvAttack: Black-Box Attack On Survival Models through Ontology-Informed EHR Perturbation**

cs.LG

**SubmitDate**: 2024-12-24    [abs](http://arxiv.org/abs/2412.18706v1) [paper-pdf](http://arxiv.org/pdf/2412.18706v1)

**Authors**: Mohsen Nayebi Kerdabadi, Arya Hadizadeh Moghaddam, Bin Liu, Mei Liu, Zijun Yao

**Abstract**: Survival analysis (SA) models have been widely studied in mining electronic health records (EHRs), particularly in forecasting the risk of critical conditions for prioritizing high-risk patients. However, their vulnerability to adversarial attacks is much less explored in the literature. Developing black-box perturbation algorithms and evaluating their impact on state-of-the-art survival models brings two benefits to medical applications. First, it can effectively evaluate the robustness of models in pre-deployment testing. Also, exploring how subtle perturbations would result in significantly different outcomes can provide counterfactual insights into the clinical interpretation of model prediction. In this work, we introduce SurvAttack, a novel black-box adversarial attack framework leveraging subtle clinically compatible, and semantically consistent perturbations on longitudinal EHRs to degrade survival models' predictive performance. We specifically develop a greedy algorithm to manipulate medical codes with various adversarial actions throughout a patient's medical history. Then, these adversarial actions are prioritized using a composite scoring strategy based on multi-aspect perturbation quality, including saliency, perturbation stealthiness, and clinical meaningfulness. The proposed adversarial EHR perturbation algorithm is then used in an efficient SA-specific strategy to attack a survival model when estimating the temporal ranking of survival urgency for patients. To demonstrate the significance of our work, we conduct extensive experiments, including baseline comparisons, explainability analysis, and case studies. The experimental results affirm our research's effectiveness in illustrating the vulnerabilities of patient survival models, model interpretation, and ultimately contributing to healthcare quality.



## **18. Adversarial Attack Against Images Classification based on Generative Adversarial Networks**

cs.CV

7 pages, 6 figures

**SubmitDate**: 2024-12-24    [abs](http://arxiv.org/abs/2412.16662v2) [paper-pdf](http://arxiv.org/pdf/2412.16662v2)

**Authors**: Yahe Yang

**Abstract**: Adversarial attacks on image classification systems have always been an important problem in the field of machine learning, and generative adversarial networks (GANs), as popular models in the field of image generation, have been widely used in various novel scenarios due to their powerful generative capabilities. However, with the popularity of generative adversarial networks, the misuse of fake image technology has raised a series of security problems, such as malicious tampering with other people's photos and videos, and invasion of personal privacy. Inspired by the generative adversarial networks, this work proposes a novel adversarial attack method, aiming to gain insight into the weaknesses of the image classification system and improve its anti-attack ability. Specifically, the generative adversarial networks are used to generate adversarial samples with small perturbations but enough to affect the decision-making of the classifier, and the adversarial samples are generated through the adversarial learning of the training generator and the classifier. From extensive experiment analysis, we evaluate the effectiveness of the method on a classical image classification dataset, and the results show that our model successfully deceives a variety of advanced classifiers while maintaining the naturalness of adversarial samples.



## **19. An Empirical Analysis of Federated Learning Models Subject to Label-Flipping Adversarial Attack**

cs.LG

**SubmitDate**: 2024-12-24    [abs](http://arxiv.org/abs/2412.18507v1) [paper-pdf](http://arxiv.org/pdf/2412.18507v1)

**Authors**: Kunal Bhatnagar, Sagana Chattanathan, Angela Dang, Bhargav Eranki, Ronnit Rana, Charan Sridhar, Siddharth Vedam, Angie Yao, Mark Stamp

**Abstract**: In this paper, we empirically analyze adversarial attacks on selected federated learning models. The specific learning models considered are Multinominal Logistic Regression (MLR), Support Vector Classifier (SVC), Multilayer Perceptron (MLP), Convolution Neural Network (CNN), %Recurrent Neural Network (RNN), Random Forest, XGBoost, and Long Short-Term Memory (LSTM). For each model, we simulate label-flipping attacks, experimenting extensively with 10 federated clients and 100 federated clients. We vary the percentage of adversarial clients from 10% to 100% and, simultaneously, the percentage of labels flipped by each adversarial client is also varied from 10% to 100%. Among other results, we find that models differ in their inherent robustness to the two vectors in our label-flipping attack, i.e., the percentage of adversarial clients, and the percentage of labels flipped by each adversarial client. We discuss the potential practical implications of our results.



## **20. Prompted Contextual Vectors for Spear-Phishing Detection**

cs.LG

**SubmitDate**: 2024-12-24    [abs](http://arxiv.org/abs/2402.08309v3) [paper-pdf](http://arxiv.org/pdf/2402.08309v3)

**Authors**: Daniel Nahmias, Gal Engelberg, Dan Klein, Asaf Shabtai

**Abstract**: Spear-phishing attacks present a significant security challenge, with large language models (LLMs) escalating the threat by generating convincing emails and facilitating target reconnaissance. To address this, we propose a detection approach based on a novel document vectorization method that utilizes an ensemble of LLMs to create representation vectors. By prompting LLMs to reason and respond to human-crafted questions, we quantify the presence of common persuasion principles in the email's content, producing prompted contextual document vectors for a downstream supervised machine learning model. We evaluate our method using a unique dataset generated by a proprietary system that automates target reconnaissance and spear-phishing email creation. Our method achieves a 91\% F1 score in identifying LLM-generated spear-phishing emails, with the training set comprising only traditional phishing and benign emails. Key contributions include a novel document vectorization method utilizing LLM reasoning, a publicly available dataset of high-quality spear-phishing emails, and the demonstrated effectiveness of our method in detecting such emails. This methodology can be utilized for various document classification tasks, particularly in adversarial problem domains.



## **21. Unveiling the Threat of Fraud Gangs to Graph Neural Networks: Multi-Target Graph Injection Attacks against GNN-Based Fraud Detectors**

cs.LG

19 pages, 5 figures, 12 tables, The 39th AAAI Conference on  Artificial Intelligence (AAAI 2025)

**SubmitDate**: 2024-12-24    [abs](http://arxiv.org/abs/2412.18370v1) [paper-pdf](http://arxiv.org/pdf/2412.18370v1)

**Authors**: Jinhyeok Choi, Heehyeon Kim, Joyce Jiyoung Whang

**Abstract**: Graph neural networks (GNNs) have emerged as an effective tool for fraud detection, identifying fraudulent users, and uncovering malicious behaviors. However, attacks against GNN-based fraud detectors and their risks have rarely been studied, thereby leaving potential threats unaddressed. Recent findings suggest that frauds are increasingly organized as gangs or groups. In this work, we design attack scenarios where fraud gangs aim to make their fraud nodes misclassified as benign by camouflaging their illicit activities in collusion. Based on these scenarios, we study adversarial attacks against GNN-based fraud detectors by simulating attacks of fraud gangs in three real-world fraud cases: spam reviews, fake news, and medical insurance frauds. We define these attacks as multi-target graph injection attacks and propose MonTi, a transformer-based Multi-target one-Time graph injection attack model. MonTi simultaneously generates attributes and edges of all attack nodes with a transformer encoder, capturing interdependencies between attributes and edges more effectively than most existing graph injection attack methods that generate these elements sequentially. Additionally, MonTi adaptively allocates the degree budget for each attack node to explore diverse injection structures involving target, candidate, and attack nodes, unlike existing methods that fix the degree budget across all attack nodes. Experiments show that MonTi outperforms the state-of-the-art graph injection attack methods on five real-world graphs.



## **22. Hypergraph Attacks via Injecting Homogeneous Nodes into Elite Hyperedges**

cs.LG

9 pages, The 39th Annual AAAI Conference on Artificial  Intelligence(2025)

**SubmitDate**: 2024-12-24    [abs](http://arxiv.org/abs/2412.18365v1) [paper-pdf](http://arxiv.org/pdf/2412.18365v1)

**Authors**: Meixia He, Peican Zhu, Keke Tang, Yangming Guo

**Abstract**: Recent studies have shown that Hypergraph Neural Networks (HGNNs) are vulnerable to adversarial attacks. Existing approaches focus on hypergraph modification attacks guided by gradients, overlooking node spanning in the hypergraph and the group identity of hyperedges, thereby resulting in limited attack performance and detectable attacks. In this manuscript, we present a novel framework, i.e., Hypergraph Attacks via Injecting Homogeneous Nodes into Elite Hyperedges (IE-Attack), to tackle these challenges. Initially, utilizing the node spanning in the hypergraph, we propose the elite hyperedges sampler to identify hyperedges to be injected. Subsequently, a node generator utilizing Kernel Density Estimation (KDE) is proposed to generate the homogeneous node with the group identity of hyperedges. Finally, by injecting the homogeneous node into elite hyperedges, IE-Attack improves the attack performance and enhances the imperceptibility of attacks. Extensive experiments are conducted on five authentic datasets to validate the effectiveness of IE-Attack and the corresponding superiority to state-of-the-art methods.



## **23. Level Up with ML Vulnerability Identification: Leveraging Domain Constraints in Feature Space for Robust Android Malware Detection**

cs.LG

The paper was accepted by ACM Transactions on Privacy and Security on  2 December 2024

**SubmitDate**: 2024-12-24    [abs](http://arxiv.org/abs/2205.15128v4) [paper-pdf](http://arxiv.org/pdf/2205.15128v4)

**Authors**: Hamid Bostani, Zhengyu Zhao, Zhuoran Liu, Veelasha Moonsamy

**Abstract**: Machine Learning (ML) promises to enhance the efficacy of Android Malware Detection (AMD); however, ML models are vulnerable to realistic evasion attacks--crafting realizable Adversarial Examples (AEs) that satisfy Android malware domain constraints. To eliminate ML vulnerabilities, defenders aim to identify susceptible regions in the feature space where ML models are prone to deception. The primary approach to identifying vulnerable regions involves investigating realizable AEs, but generating these feasible apps poses a challenge. For instance, previous work has relied on generating either feature-space norm-bounded AEs or problem-space realizable AEs in adversarial hardening. The former is efficient but lacks full coverage of vulnerable regions while the latter can uncover these regions by satisfying domain constraints but is known to be time-consuming. To address these limitations, we propose an approach to facilitate the identification of vulnerable regions. Specifically, we introduce a new interpretation of Android domain constraints in the feature space, followed by a novel technique that learns them. Our empirical evaluations across various evasion attacks indicate effective detection of AEs using learned domain constraints, with an average of 89.6%. Furthermore, extensive experiments on different Android malware detectors demonstrate that utilizing our learned domain constraints in Adversarial Training (AT) outperforms other AT-based defenses that rely on norm-bounded AEs or state-of-the-art non-uniform perturbations. Finally, we show that retraining a malware detector with a wide variety of feature-space realizable AEs results in a 77.9% robustness improvement against realizable AEs generated by unknown problem-space transformations, with up to 70x faster training than using problem-space realizable AEs.



## **24. ErasableMask: A Robust and Erasable Privacy Protection Scheme against Black-box Face Recognition Models**

cs.CV

**SubmitDate**: 2024-12-24    [abs](http://arxiv.org/abs/2412.17038v2) [paper-pdf](http://arxiv.org/pdf/2412.17038v2)

**Authors**: Sipeng Shen, Yunming Zhang, Dengpan Ye, Xiuwen Shi, Long Tang, Haoran Duan, Jiacheng Deng, Ziyi Liu

**Abstract**: While face recognition (FR) models have brought remarkable convenience in face verification and identification, they also pose substantial privacy risks to the public. Existing facial privacy protection schemes usually adopt adversarial examples to disrupt face verification of FR models. However, these schemes often suffer from weak transferability against black-box FR models and permanently damage the identifiable information that cannot fulfill the requirements of authorized operations such as forensics and authentication. To address these limitations, we propose ErasableMask, a robust and erasable privacy protection scheme against black-box FR models. Specifically, via rethinking the inherent relationship between surrogate FR models, ErasableMask introduces a novel meta-auxiliary attack, which boosts black-box transferability by learning more general features in a stable and balancing optimization strategy. It also offers a perturbation erasion mechanism that supports the erasion of semantic perturbations in protected face without degrading image quality. To further improve performance, ErasableMask employs a curriculum learning strategy to mitigate optimization conflicts between adversarial attack and perturbation erasion. Extensive experiments on the CelebA-HQ and FFHQ datasets demonstrate that ErasableMask achieves the state-of-the-art performance in transferability, achieving over 72% confidence on average in commercial FR systems. Moreover, ErasableMask also exhibits outstanding perturbation erasion performance, achieving over 90% erasion success rate.



## **25. Can Large Language Models Improve the Adversarial Robustness of Graph Neural Networks?**

cs.LG

accepted by KDD 2025

**SubmitDate**: 2024-12-24    [abs](http://arxiv.org/abs/2408.08685v3) [paper-pdf](http://arxiv.org/pdf/2408.08685v3)

**Authors**: Zhongjian Zhang, Xiao Wang, Huichi Zhou, Yue Yu, Mengmei Zhang, Cheng Yang, Chuan Shi

**Abstract**: Graph neural networks (GNNs) are vulnerable to adversarial attacks, especially for topology perturbations, and many methods that improve the robustness of GNNs have received considerable attention. Recently, we have witnessed the significant success of large language models (LLMs), leading many to explore the great potential of LLMs on GNNs. However, they mainly focus on improving the performance of GNNs by utilizing LLMs to enhance the node features. Therefore, we ask: Will the robustness of GNNs also be enhanced with the powerful understanding and inference capabilities of LLMs? By presenting the empirical results, we find that despite that LLMs can improve the robustness of GNNs, there is still an average decrease of 23.1% in accuracy, implying that the GNNs remain extremely vulnerable against topology attacks. Therefore, another question is how to extend the capabilities of LLMs on graph adversarial robustness. In this paper, we propose an LLM-based robust graph structure inference framework, LLM4RGNN, which distills the inference capabilities of GPT-4 into a local LLM for identifying malicious edges and an LM-based edge predictor for finding missing important edges, so as to recover a robust graph structure. Extensive experiments demonstrate that LLM4RGNN consistently improves the robustness across various GNNs. Even in some cases where the perturbation ratio increases to 40%, the accuracy of GNNs is still better than that on the clean graph. The source code can be found in https://github.com/zhongjian-zhang/LLM4RGNN.



## **26. On the Effectiveness of Adversarial Training on Malware Classifiers**

cs.LG

**SubmitDate**: 2024-12-24    [abs](http://arxiv.org/abs/2412.18218v1) [paper-pdf](http://arxiv.org/pdf/2412.18218v1)

**Authors**: Hamid Bostani, Jacopo Cortellazzi, Daniel Arp, Fabio Pierazzi, Veelasha Moonsamy, Lorenzo Cavallaro

**Abstract**: Adversarial Training (AT) has been widely applied to harden learning-based classifiers against adversarial evasive attacks. However, its effectiveness in identifying and strengthening vulnerable areas of the model's decision space while maintaining high performance on clean data of malware classifiers remains an under-explored area. In this context, the robustness that AT achieves has often been assessed against unrealistic or weak adversarial attacks, which negatively affect performance on clean data and are arguably no longer threats. Previous work seems to suggest robustness is a task-dependent property of AT. We instead argue it is a more complex problem that requires exploring AT and the intertwined roles played by certain factors within data, feature representations, classifiers, and robust optimization settings, as well as proper evaluation factors, such as the realism of evasion attacks, to gain a true sense of AT's effectiveness. In our paper, we address this gap by systematically exploring the role such factors have in hardening malware classifiers through AT. Contrary to recent prior work, a key observation of our research and extensive experiments confirm the hypotheses that all such factors influence the actual effectiveness of AT, as demonstrated by the varying degrees of success from our empirical analysis. We identify five evaluation pitfalls that affect state-of-the-art studies and summarize our insights in ten takeaways to draw promising research directions toward better understanding the factors' settings under which adversarial training works at best.



## **27. Robustness-aware Automatic Prompt Optimization**

cs.CL

**SubmitDate**: 2024-12-24    [abs](http://arxiv.org/abs/2412.18196v1) [paper-pdf](http://arxiv.org/pdf/2412.18196v1)

**Authors**: Zeru Shi, Zhenting Wang, Yongye Su, Weidi Luo, Fan Yang, Yongfeng Zhang

**Abstract**: The performance of Large Language Models (LLMs) is based on the quality of the prompts and the semantic and structural integrity information of the input data. However, current prompt generation methods primarily focus on generating prompts for clean input data, often overlooking the impact of perturbed inputs on prompt performance. To address this limitation, we propose BATprompt (By Adversarial Training prompt), a novel method for prompt generation designed to withstand input perturbations (such as typos in the input). Inspired by adversarial training techniques, BATprompt demonstrates strong performance on a variety of perturbed tasks through a two-step process: adversarial perturbation and iterative optimization on unperturbed input via LLM. Unlike conventional adversarial attack methods, BATprompt avoids reliance on real gradients or model parameters. Instead, it leverages the advanced reasoning, language understanding and self reflection capabilities of LLMs to simulate gradients, guiding the generation of adversarial perturbations and optimizing prompt performance. In our experiments, we evaluate BATprompt on multiple datasets across both language understanding and generation tasks. The results indicate that BATprompt outperforms existing prompt generation methods, delivering superior robustness and performance under diverse perturbation scenarios.



## **28. Sparse-PGD: A Unified Framework for Sparse Adversarial Perturbations Generation**

cs.LG

Extended version. Codes are available at  https://github.com/CityU-MLO/sPGD

**SubmitDate**: 2024-12-24    [abs](http://arxiv.org/abs/2405.05075v3) [paper-pdf](http://arxiv.org/pdf/2405.05075v3)

**Authors**: Xuyang Zhong, Chen Liu

**Abstract**: This work studies sparse adversarial perturbations, including both unstructured and structured ones. We propose a framework based on a white-box PGD-like attack method named Sparse-PGD to effectively and efficiently generate such perturbations. Furthermore, we combine Sparse-PGD with a black-box attack to comprehensively and more reliably evaluate the models' robustness against unstructured and structured sparse adversarial perturbations. Moreover, the efficiency of Sparse-PGD enables us to conduct adversarial training to build robust models against various sparse perturbations. Extensive experiments demonstrate that our proposed attack algorithm exhibits strong performance in different scenarios. More importantly, compared with other robust models, our adversarially trained model demonstrates state-of-the-art robustness against various sparse attacks.



## **29. AEIOU: A Unified Defense Framework against NSFW Prompts in Text-to-Image Models**

cs.CR

**SubmitDate**: 2024-12-24    [abs](http://arxiv.org/abs/2412.18123v1) [paper-pdf](http://arxiv.org/pdf/2412.18123v1)

**Authors**: Yiming Wang, Jiahao Chen, Qingming Li, Xing Yang, Shouling Ji

**Abstract**: As text-to-image (T2I) models continue to advance and gain widespread adoption, their associated safety issues are becoming increasingly prominent. Malicious users often exploit these models to generate Not-Safe-for-Work (NSFW) images using harmful or adversarial prompts, highlighting the critical need for robust safeguards to ensure the integrity and compliance of model outputs. Current internal safeguards frequently degrade image quality, while external detection methods often suffer from low accuracy and inefficiency.   In this paper, we introduce AEIOU, a defense framework that is Adaptable, Efficient, Interpretable, Optimizable, and Unified against NSFW prompts in T2I models. AEIOU extracts NSFW features from the hidden states of the model's text encoder, utilizing the separable nature of these features to detect NSFW prompts. The detection process is efficient, requiring minimal inference time. AEIOU also offers real-time interpretation of results and supports optimization through data augmentation techniques. The framework is versatile, accommodating various T2I architectures. Our extensive experiments show that AEIOU significantly outperforms both commercial and open-source moderation tools, achieving over 95% accuracy across all datasets and improving efficiency by at least tenfold. It effectively counters adaptive attacks and excels in few-shot and multi-label scenarios.



## **30. A Tunable Despeckling Neural Network Stabilized via Diffusion Equation**

cs.CV

**SubmitDate**: 2024-12-23    [abs](http://arxiv.org/abs/2411.15921v2) [paper-pdf](http://arxiv.org/pdf/2411.15921v2)

**Authors**: Yi Ran, Zhichang Guo, Jia Li, Yao Li, Martin Burger, Boying Wu

**Abstract**: The removal of multiplicative Gamma noise is a critical research area in the application of synthetic aperture radar (SAR) imaging, where neural networks serve as a potent tool. However, real-world data often diverges from theoretical models, exhibiting various disturbances, which makes the neural network less effective. Adversarial attacks can be used as a criterion for judging the adaptability of neural networks to real data, since adversarial attacks can find the most extreme perturbations that make neural networks ineffective. In this work, the diffusion equation is designed as a regularization block to provide sufficient regularity to the whole neural network, due to its spontaneous dissipative nature. We propose a tunable, regularized neural network framework that unrolls a shallow denoising neural network block and a diffusion regularity block into a single network for end-to-end training. The linear heat equation, known for its inherent smoothness and low-pass filtering properties, is adopted as the diffusion regularization block. In our model, a single time step hyperparameter governs the smoothness of the outputs and can be adjusted dynamically, significantly enhancing flexibility. The stability and convergence of our model are theoretically proven. Experimental results demonstrate that the proposed model effectively eliminates high-frequency oscillations induced by adversarial attacks. Finally, the proposed model is benchmarked against several state-of-the-art denoising methods on simulated images, adversarial samples, and real SAR images, achieving superior performance in both quantitative and visual evaluations.



## **31. Large Language Model Safety: A Holistic Survey**

cs.AI

158 pages, 18 figures

**SubmitDate**: 2024-12-23    [abs](http://arxiv.org/abs/2412.17686v1) [paper-pdf](http://arxiv.org/pdf/2412.17686v1)

**Authors**: Dan Shi, Tianhao Shen, Yufei Huang, Zhigen Li, Yongqi Leng, Renren Jin, Chuang Liu, Xinwei Wu, Zishan Guo, Linhao Yu, Ling Shi, Bojian Jiang, Deyi Xiong

**Abstract**: The rapid development and deployment of large language models (LLMs) have introduced a new frontier in artificial intelligence, marked by unprecedented capabilities in natural language understanding and generation. However, the increasing integration of these models into critical applications raises substantial safety concerns, necessitating a thorough examination of their potential risks and associated mitigation strategies.   This survey provides a comprehensive overview of the current landscape of LLM safety, covering four major categories: value misalignment, robustness to adversarial attacks, misuse, and autonomous AI risks. In addition to the comprehensive review of the mitigation methodologies and evaluation resources on these four aspects, we further explore four topics related to LLM safety: the safety implications of LLM agents, the role of interpretability in enhancing LLM safety, the technology roadmaps proposed and abided by a list of AI companies and institutes for LLM safety, and AI governance aimed at LLM safety with discussions on international cooperation, policy proposals, and prospective regulatory directions.   Our findings underscore the necessity for a proactive, multifaceted approach to LLM safety, emphasizing the integration of technical solutions, ethical considerations, and robust governance frameworks. This survey is intended to serve as a foundational resource for academy researchers, industry practitioners, and policymakers, offering insights into the challenges and opportunities associated with the safe integration of LLMs into society. Ultimately, it seeks to contribute to the safe and beneficial development of LLMs, aligning with the overarching goal of harnessing AI for societal advancement and well-being. A curated list of related papers has been publicly available at https://github.com/tjunlp-lab/Awesome-LLM-Safety-Papers.



## **32. Emerging Security Challenges of Large Language Models**

cs.CR

A version of this appeared in the larger Dagstuhl seminar 23431  report (https://doi.org/10.4230/DagRep.13.10.90)

**SubmitDate**: 2024-12-23    [abs](http://arxiv.org/abs/2412.17614v1) [paper-pdf](http://arxiv.org/pdf/2412.17614v1)

**Authors**: Herve Debar, Sven Dietrich, Pavel Laskov, Emil C. Lupu, Eirini Ntoutsi

**Abstract**: Large language models (LLMs) have achieved record adoption in a short period of time across many different sectors including high importance areas such as education [4] and healthcare [23]. LLMs are open-ended models trained on diverse data without being tailored for specific downstream tasks, enabling broad applicability across various domains. They are commonly used for text generation, but also widely used to assist with code generation [3], and even analysis of security information, as Microsoft Security Copilot demonstrates [18]. Traditional Machine Learning (ML) models are vulnerable to adversarial attacks [9]. So the concerns on the potential security implications of such wide scale adoption of LLMs have led to the creation of this working group on the security of LLMs. During the Dagstuhl seminar on "Network Attack Detection and Defense - AI-Powered Threats and Responses", the working group discussions focused on the vulnerability of LLMs to adversarial attacks, rather than their potential use in generating malware or enabling cyberattacks. Although we note the potential threat represented by the latter, the role of the LLMs in such uses is mostly as an accelerator for development, similar to what it is in benign use. To make the analysis more specific, the working group employed ChatGPT as a concrete example of an LLM and addressed the following points, which also form the structure of this report: 1. How do LLMs differ in vulnerabilities from traditional ML models? 2. What are the attack objectives in LLMs? 3. How complex it is to assess the risks posed by the vulnerabilities of LLMs? 4. What is the supply chain in LLMs, how data flow in and out of systems and what are the security implications? We conclude with an overview of open challenges and outlook.



## **33. Retention Score: Quantifying Jailbreak Risks for Vision Language Models**

cs.AI

14 pages, 8 figures, AAAI 2025

**SubmitDate**: 2024-12-23    [abs](http://arxiv.org/abs/2412.17544v1) [paper-pdf](http://arxiv.org/pdf/2412.17544v1)

**Authors**: Zaitang Li, Pin-Yu Chen, Tsung-Yi Ho

**Abstract**: The emergence of Vision-Language Models (VLMs) is a significant advancement in integrating computer vision with Large Language Models (LLMs) to enhance multi-modal machine learning capabilities. However, this progress has also made VLMs vulnerable to sophisticated adversarial attacks, raising concerns about their reliability. The objective of this paper is to assess the resilience of VLMs against jailbreak attacks that can compromise model safety compliance and result in harmful outputs. To evaluate a VLM's ability to maintain its robustness against adversarial input perturbations, we propose a novel metric called the \textbf{Retention Score}. Retention Score is a multi-modal evaluation metric that includes Retention-I and Retention-T scores for quantifying jailbreak risks in visual and textual components of VLMs. Our process involves generating synthetic image-text pairs using a conditional diffusion model. These pairs are then predicted for toxicity score by a VLM alongside a toxicity judgment classifier. By calculating the margin in toxicity scores, we can quantify the robustness of the VLM in an attack-agnostic manner. Our work has four main contributions. First, we prove that Retention Score can serve as a certified robustness metric. Second, we demonstrate that most VLMs with visual components are less robust against jailbreak attacks than the corresponding plain VLMs. Additionally, we evaluate black-box VLM APIs and find that the security settings in Google Gemini significantly affect the score and robustness. Moreover, the robustness of GPT4V is similar to the medium settings of Gemini. Finally, our approach offers a time-efficient alternative to existing adversarial attack methods and provides consistent model robustness rankings when evaluated on VLMs including MiniGPT-4, InstructBLIP, and LLaVA.



## **34. Gröbner Basis Cryptanalysis of Ciminion and Hydra**

cs.CR

**SubmitDate**: 2024-12-23    [abs](http://arxiv.org/abs/2405.05040v2) [paper-pdf](http://arxiv.org/pdf/2405.05040v2)

**Authors**: Matthias Johann Steiner

**Abstract**: Ciminion and Hydra are two recently introduced symmetric key Pseudo-Random Functions for Multi-Party Computation applications. For efficiency both primitives utilize quadratic permutations at round level. Therefore, polynomial system solving-based attacks pose a serious threat to these primitives. For Ciminion, we construct a quadratic degree reverse lexicographic (DRL) Gr\"obner basis for the iterated polynomial model via linear transformations. With the Gr\"obner basis we can simplify cryptanalysis since we do not need to impose genericity assumptions anymore to derive complexity estimations. For Hydra, with the help of a computer algebra program like SageMath we construct a DRL Gr\"obner basis for the iterated model via linear transformations and a linear change of coordinates. In the Hydra proposal it was claimed that $r_\mathcal{H} = 31$ rounds are sufficient to provide $128$ bits of security against Gr\"obner basis attacks for an ideal adversary with $\omega = 2$. However, via our Hydra Gr\"obner basis standard term order conversion to a lexicographic (LEX) Gr\"obner basis requires just $126$ bits with $\omega = 2$. Moreover, via a dedicated polynomial system solving technique up to $r_\mathcal{H} = 33$ rounds can be attacked below $128$ bits for an ideal adversary.



## **35. Ensembler: Protect Collaborative Inference Privacy from Model Inversion Attack via Selective Ensemble**

cs.CR

in submission

**SubmitDate**: 2024-12-23    [abs](http://arxiv.org/abs/2401.10859v2) [paper-pdf](http://arxiv.org/pdf/2401.10859v2)

**Authors**: Dancheng Liu, Chenhui Xu, Jiajie Li, Amir Nassereldine, Jinjun Xiong

**Abstract**: For collaborative inference through a cloud computing platform, it is sometimes essential for the client to shield its sensitive information from the cloud provider. In this paper, we introduce Ensembler, an extensible framework designed to substantially increase the difficulty of conducting model inversion attacks by adversarial parties. Ensembler leverages selective model ensemble on the adversarial server to obfuscate the reconstruction of the client's private information. Our experiments demonstrate that Ensembler can effectively shield input images from reconstruction attacks, even when the client only retains one layer of the network locally. Ensembler significantly outperforms baseline methods by up to 43.5% in structural similarity while only incurring 4.8% time overhead during inference.



## **36. SEAS: Self-Evolving Adversarial Safety Optimization for Large Language Models**

cs.CL

**SubmitDate**: 2024-12-23    [abs](http://arxiv.org/abs/2408.02632v2) [paper-pdf](http://arxiv.org/pdf/2408.02632v2)

**Authors**: Muxi Diao, Rumei Li, Shiyang Liu, Guogang Liao, Jingang Wang, Xunliang Cai, Weiran Xu

**Abstract**: As large language models (LLMs) continue to advance in capability and influence, ensuring their security and preventing harmful outputs has become crucial. A promising approach to address these concerns involves training models to automatically generate adversarial prompts for red teaming. However, the evolving subtlety of vulnerabilities in LLMs challenges the effectiveness of current adversarial methods, which struggle to specifically target and explore the weaknesses of these models. To tackle these challenges, we introduce the $\mathbf{S}\text{elf-}\mathbf{E}\text{volving }\mathbf{A}\text{dversarial }\mathbf{S}\text{afety }\mathbf{(SEAS)}$ optimization framework, which enhances security by leveraging data generated by the model itself. SEAS operates through three iterative stages: Initialization, Attack, and Adversarial Optimization, refining both the Red Team and Target models to improve robustness and safety. This framework reduces reliance on manual testing and significantly enhances the security capabilities of LLMs. Our contributions include a novel adversarial framework, a comprehensive safety dataset, and after three iterations, the Target model achieves a security level comparable to GPT-4, while the Red Team model shows a marked increase in attack success rate (ASR) against advanced models. Our code and datasets are released at https://SEAS-LLM.github.io/.



## **37. The Superalignment of Superhuman Intelligence with Large Language Models**

cs.CL

Under review of Science China

**SubmitDate**: 2024-12-23    [abs](http://arxiv.org/abs/2412.11145v2) [paper-pdf](http://arxiv.org/pdf/2412.11145v2)

**Authors**: Minlie Huang, Yingkang Wang, Shiyao Cui, Pei Ke, Jie Tang

**Abstract**: We have witnessed superhuman intelligence thanks to the fast development of large language models and multimodal language models. As the application of such superhuman models becomes more and more popular, a critical question arises here: how can we ensure superhuman models are still safe, reliable and aligned well to human values? In this position paper, we discuss the concept of superalignment from the learning perspective to answer this question by outlining the learning paradigm shift from large-scale pretraining, supervised fine-tuning, to alignment training. We define superalignment as designing effective and efficient alignment algorithms to learn from noisy-labeled data (point-wise samples or pair-wise preference data) in a scalable way when the task becomes very complex for human experts to annotate and the model is stronger than human experts. We highlight some key research problems in superalignment, namely, weak-to-strong generalization, scalable oversight, and evaluation. We then present a conceptual framework for superalignment, which consists of three modules: an attacker which generates adversary queries trying to expose the weaknesses of a learner model; a learner which will refine itself by learning from scalable feedbacks generated by a critic model along with minimal human experts; and a critic which generates critics or explanations for a given query-response pair, with a target of improving the learner by criticizing. We discuss some important research problems in each component of this framework and highlight some interesting research ideas that are closely related to our proposed framework, for instance, self-alignment, self-play, self-refinement, and more. Last, we highlight some future research directions for superalignment, including identification of new emergent risks and multi-dimensional alignment.



## **38. DynamicPAE: Generating Scene-Aware Physical Adversarial Examples in Real-Time**

cs.CV

This work has been submitted to the IEEE for possible publication

**SubmitDate**: 2024-12-23    [abs](http://arxiv.org/abs/2412.08053v2) [paper-pdf](http://arxiv.org/pdf/2412.08053v2)

**Authors**: Jin Hu, Xianglong Liu, Jiakai Wang, Junkai Zhang, Xianqi Yang, Haotong Qin, Yuqing Ma, Ke Xu

**Abstract**: Physical adversarial examples (PAEs) are regarded as "whistle-blowers" of real-world risks in deep-learning applications. However, current PAE generation studies show limited adaptive attacking ability to diverse and varying scenes. The key challenges in generating dynamic PAEs are exploring their patterns under noisy gradient feedback and adapting the attack to agnostic scenario natures. To address the problems, we present DynamicPAE, the first generative framework that enables scene-aware real-time physical attacks beyond static attacks. Specifically, to train the dynamic PAE generator under noisy gradient feedback, we introduce the residual-driven sample trajectory guidance technique, which redefines the training task to break the limited feedback information restriction that leads to the degeneracy problem. Intuitively, it allows the gradient feedback to be passed to the generator through a low-noise auxiliary task, thereby guiding the optimization away from degenerate solutions and facilitating a more comprehensive and stable exploration of feasible PAEs. To adapt the generator to agnostic scenario natures, we introduce the context-aligned scene expectation simulation process, consisting of the conditional-uncertainty-aligned data module and the skewness-aligned objective re-weighting module. The former enhances robustness in the context of incomplete observation by employing a conditional probabilistic model for domain randomization, while the latter facilitates consistent stealth control across different attack targets by automatically reweighting losses based on the skewness indicator. Extensive digital and physical evaluations demonstrate the superior attack performance of DynamicPAE, attaining a 1.95 $\times$ boost (65.55% average AP drop under attack) on representative object detectors (e.g., Yolo-v8) over state-of-the-art static PAE generating methods.



## **39. Robustness of Large Language Models Against Adversarial Attacks**

cs.CL

**SubmitDate**: 2024-12-22    [abs](http://arxiv.org/abs/2412.17011v1) [paper-pdf](http://arxiv.org/pdf/2412.17011v1)

**Authors**: Yiyi Tao, Yixian Shen, Hang Zhang, Yanxin Shen, Lun Wang, Chuanqi Shi, Shaoshuai Du

**Abstract**: The increasing deployment of Large Language Models (LLMs) in various applications necessitates a rigorous evaluation of their robustness against adversarial attacks. In this paper, we present a comprehensive study on the robustness of GPT LLM family. We employ two distinct evaluation methods to assess their resilience. The first method introduce character-level text attack in input prompts, testing the models on three sentiment classification datasets: StanfordNLP/IMDB, Yelp Reviews, and SST-2. The second method involves using jailbreak prompts to challenge the safety mechanisms of the LLMs. Our experiments reveal significant variations in the robustness of these models, demonstrating their varying degrees of vulnerability to both character-level and semantic-level adversarial attacks. These findings underscore the necessity for improved adversarial training and enhanced safety mechanisms to bolster the robustness of LLMs.



## **40. Breaking Barriers in Physical-World Adversarial Examples: Improving Robustness and Transferability via Robust Feature**

cs.CV

Accepted by AAAI2025

**SubmitDate**: 2024-12-22    [abs](http://arxiv.org/abs/2412.16958v1) [paper-pdf](http://arxiv.org/pdf/2412.16958v1)

**Authors**: Yichen Wang, Yuxuan Chou, Ziqi Zhou, Hangtao Zhang, Wei Wan, Shengshan Hu, Minghui Li

**Abstract**: As deep neural networks (DNNs) are widely applied in the physical world, many researches are focusing on physical-world adversarial examples (PAEs), which introduce perturbations to inputs and cause the model's incorrect outputs. However, existing PAEs face two challenges: unsatisfactory attack performance (i.e., poor transferability and insufficient robustness to environment conditions), and difficulty in balancing attack effectiveness with stealthiness, where better attack effectiveness often makes PAEs more perceptible.   In this paper, we explore a novel perturbation-based method to overcome the challenges. For the first challenge, we introduce a strategy Deceptive RF injection based on robust features (RFs) that are predictive, robust to perturbations, and consistent across different models. Specifically, it improves the transferability and robustness of PAEs by covering RFs of other classes onto the predictive features in clean images. For the second challenge, we introduce another strategy Adversarial Semantic Pattern Minimization, which removes most perturbations and retains only essential adversarial patterns in AEsBased on the two strategies, we design our method Robust Feature Coverage Attack (RFCoA), comprising Robust Feature Disentanglement and Adversarial Feature Fusion. In the first stage, we extract target class RFs in feature space. In the second stage, we use attention-based feature fusion to overlay these RFs onto predictive features of clean images and remove unnecessary perturbations. Experiments show our method's superior transferability, robustness, and stealthiness compared to existing state-of-the-art methods. Additionally, our method's effectiveness can extend to Large Vision-Language Models (LVLMs), indicating its potential applicability to more complex tasks.



## **41. NumbOD: A Spatial-Frequency Fusion Attack Against Object Detectors**

cs.CV

Accepted by AAAI 2025

**SubmitDate**: 2024-12-22    [abs](http://arxiv.org/abs/2412.16955v1) [paper-pdf](http://arxiv.org/pdf/2412.16955v1)

**Authors**: Ziqi Zhou, Bowen Li, Yufei Song, Zhifei Yu, Shengshan Hu, Wei Wan, Leo Yu Zhang, Dezhong Yao, Hai Jin

**Abstract**: With the advancement of deep learning, object detectors (ODs) with various architectures have achieved significant success in complex scenarios like autonomous driving. Previous adversarial attacks against ODs have been focused on designing customized attacks targeting their specific structures (e.g., NMS and RPN), yielding some results but simultaneously constraining their scalability. Moreover, most efforts against ODs stem from image-level attacks originally designed for classification tasks, resulting in redundant computations and disturbances in object-irrelevant areas (e.g., background). Consequently, how to design a model-agnostic efficient attack to comprehensively evaluate the vulnerabilities of ODs remains challenging and unresolved. In this paper, we propose NumbOD, a brand-new spatial-frequency fusion attack against various ODs, aimed at disrupting object detection within images. We directly leverage the features output by the OD without relying on its internal structures to craft adversarial examples. Specifically, we first design a dual-track attack target selection strategy to select high-quality bounding boxes from OD outputs for targeting. Subsequently, we employ directional perturbations to shift and compress predicted boxes and change classification results to deceive ODs. Additionally, we focus on manipulating the high-frequency components of images to confuse ODs' attention on critical objects, thereby enhancing the attack efficiency. Our extensive experiments on nine ODs and two datasets show that NumbOD achieves powerful attack performance and high stealthiness.



## **42. Preventing Non-intrusive Load Monitoring Privacy Invasion: A Precise Adversarial Attack Scheme for Networked Smart Meters**

cs.CR

**SubmitDate**: 2024-12-22    [abs](http://arxiv.org/abs/2412.16893v1) [paper-pdf](http://arxiv.org/pdf/2412.16893v1)

**Authors**: Jialing He, Jiacheng Wang, Ning Wang, Shangwei Guo, Liehuang Zhu, Dusit Niyato, Tao Xiang

**Abstract**: Smart grid, through networked smart meters employing the non-intrusive load monitoring (NILM) technique, can considerably discern the usage patterns of residential appliances. However, this technique also incurs privacy leakage. To address this issue, we propose an innovative scheme based on adversarial attack in this paper. The scheme effectively prevents NILM models from violating appliance-level privacy, while also ensuring accurate billing calculation for users. To achieve this objective, we overcome two primary challenges. First, as NILM models fall under the category of time-series regression models, direct application of traditional adversarial attacks designed for classification tasks is not feasible. To tackle this issue, we formulate a novel adversarial attack problem tailored specifically for NILM and providing a theoretical foundation for utilizing the Jacobian of the NILM model to generate imperceptible perturbations. Leveraging the Jacobian, our scheme can produce perturbations, which effectively misleads the signal prediction of NILM models to safeguard users' appliance-level privacy. The second challenge pertains to fundamental utility requirements, where existing adversarial attack schemes struggle to achieve accurate billing calculation for users. To handle this problem, we introduce an additional constraint, mandating that the sum of added perturbations within a billing period must be precisely zero. Experimental validation on real-world power datasets REDD and UK-DALE demonstrates the efficacy of our proposed solutions, which can significantly amplify the discrepancy between the output of the targeted NILM model and the actual power signal of appliances, and enable accurate billing at the same time. Additionally, our solutions exhibit transferability, making the generated perturbation signal from one target model applicable to other diverse NILM models.



## **43. Towards More Robust Retrieval-Augmented Generation: Evaluating RAG Under Adversarial Poisoning Attacks**

cs.IR

**SubmitDate**: 2024-12-21    [abs](http://arxiv.org/abs/2412.16708v1) [paper-pdf](http://arxiv.org/pdf/2412.16708v1)

**Authors**: Jinyan Su, Jin Peng Zhou, Zhengxin Zhang, Preslav Nakov, Claire Cardie

**Abstract**: Retrieval-Augmented Generation (RAG) systems have emerged as a promising solution to mitigate LLM hallucinations and enhance their performance in knowledge-intensive domains. However, these systems are vulnerable to adversarial poisoning attacks, where malicious passages injected into retrieval databases can mislead the model into generating factually incorrect outputs. In this paper, we investigate both the retrieval and the generation components of RAG systems to understand how to enhance their robustness against such attacks. From the retrieval perspective, we analyze why and how the adversarial contexts are retrieved and assess how the quality of the retrieved passages impacts downstream generation. From a generation perspective, we evaluate whether LLMs' advanced critical thinking and internal knowledge capabilities can be leveraged to mitigate the impact of adversarial contexts, i.e., using skeptical prompting as a self-defense mechanism. Our experiments and findings provide actionable insights into designing safer and more resilient retrieval-augmented frameworks, paving the way for their reliable deployment in real-world applications.



## **44. PB-UAP: Hybrid Universal Adversarial Attack For Image Segmentation**

cs.CV

Accepted by ICASSP 2025

**SubmitDate**: 2024-12-21    [abs](http://arxiv.org/abs/2412.16651v1) [paper-pdf](http://arxiv.org/pdf/2412.16651v1)

**Authors**: Yufei Song, Ziqi Zhou, Minghui Li, Xianlong Wang, Menghao Deng, Wei Wan, Shengshan Hu, Leo Yu Zhang

**Abstract**: With the rapid advancement of deep learning, the model robustness has become a significant research hotspot, \ie, adversarial attacks on deep neural networks. Existing works primarily focus on image classification tasks, aiming to alter the model's predicted labels. Due to the output complexity and deeper network architectures, research on adversarial examples for segmentation models is still limited, particularly for universal adversarial perturbations. In this paper, we propose a novel universal adversarial attack method designed for segmentation models, which includes dual feature separation and low-frequency scattering modules. The two modules guide the training of adversarial examples in the pixel and frequency space, respectively. Experiments demonstrate that our method achieves high attack success rates surpassing the state-of-the-art methods, and exhibits strong transferability across different models.



## **45. POEX: Policy Executable Embodied AI Jailbreak Attacks**

cs.RO

Homepage: https://poex-eai-jailbreak.github.io/

**SubmitDate**: 2024-12-21    [abs](http://arxiv.org/abs/2412.16633v1) [paper-pdf](http://arxiv.org/pdf/2412.16633v1)

**Authors**: Xuancun Lu, Zhengxian Huang, Xinfeng Li, Xiaoyu ji, Wenyuan Xu

**Abstract**: The integration of large language models (LLMs) into the planning module of Embodied Artificial Intelligence (Embodied AI) systems has greatly enhanced their ability to translate complex user instructions into executable policies. In this paper, we demystified how traditional LLM jailbreak attacks behave in the Embodied AI context. We conducted a comprehensive safety analysis of the LLM-based planning module of embodied AI systems against jailbreak attacks. Using the carefully crafted Harmful-RLbench, we accessed 20 open-source and proprietary LLMs under traditional jailbreak attacks, and highlighted two key challenges when adopting the prior jailbreak techniques to embodied AI contexts: (1) The harmful text output by LLMs does not necessarily induce harmful policies in Embodied AI context, and (2) even we can generate harmful policies, we have to guarantee they are executable in practice. To overcome those challenges, we propose Policy Executable (POEX) jailbreak attacks, where harmful instructions and optimized suffixes are injected into LLM-based planning modules, leading embodied AI to perform harmful actions in both simulated and physical environments. Our approach involves constraining adversarial suffixes to evade detection and fine-tuning a policy evaluater to improve the executability of harmful policies. We conducted extensive experiments on both a robotic arm embodied AI platform and simulators, to validate the attack and policy success rates on 136 harmful instructions from Harmful-RLbench. Our findings expose serious safety vulnerabilities in LLM-based planning modules, including the ability of POEX to be transferred across models. Finally, we propose mitigation strategies, such as safety-constrained prompts, pre- and post-planning checks, to address these vulnerabilities and ensure the safe deployment of embodied AI in real-world settings.



## **46. Automated Progressive Red Teaming**

cs.CR

Accepted by COLING 2025

**SubmitDate**: 2024-12-21    [abs](http://arxiv.org/abs/2407.03876v3) [paper-pdf](http://arxiv.org/pdf/2407.03876v3)

**Authors**: Bojian Jiang, Yi Jing, Tianhao Shen, Tong Wu, Qing Yang, Deyi Xiong

**Abstract**: Ensuring the safety of large language models (LLMs) is paramount, yet identifying potential vulnerabilities is challenging. While manual red teaming is effective, it is time-consuming, costly and lacks scalability. Automated red teaming (ART) offers a more cost-effective alternative, automatically generating adversarial prompts to expose LLM vulnerabilities. However, in current ART efforts, a robust framework is absent, which explicitly frames red teaming as an effectively learnable task. To address this gap, we propose Automated Progressive Red Teaming (APRT) as an effectively learnable framework. APRT leverages three core modules: an Intention Expanding LLM that generates diverse initial attack samples, an Intention Hiding LLM that crafts deceptive prompts, and an Evil Maker to manage prompt diversity and filter ineffective samples. The three modules collectively and progressively explore and exploit LLM vulnerabilities through multi-round interactions. In addition to the framework, we further propose a novel indicator, Attack Effectiveness Rate (AER) to mitigate the limitations of existing evaluation metrics. By measuring the likelihood of eliciting unsafe but seemingly helpful responses, AER aligns closely with human evaluations. Extensive experiments with both automatic and human evaluations, demonstrate the effectiveness of ARPT across both open- and closed-source LLMs. Specifically, APRT effectively elicits 54% unsafe yet useful responses from Meta's Llama-3-8B-Instruct, 50% from GPT-4o (API access), and 39% from Claude-3.5 (API access), showcasing its robust attack capability and transferability across LLMs (especially from open-source LLMs to closed-source LLMs).



## **47. PGD-Imp: Rethinking and Unleashing Potential of Classic PGD with Dual Strategies for Imperceptible Adversarial Attacks**

cs.LG

accepted by ICASSP 2025

**SubmitDate**: 2024-12-21    [abs](http://arxiv.org/abs/2412.11168v2) [paper-pdf](http://arxiv.org/pdf/2412.11168v2)

**Authors**: Jin Li, Zitong Yu, Ziqiang He, Z. Jane Wang, Xiangui Kang

**Abstract**: Imperceptible adversarial attacks have recently attracted increasing research interests. Existing methods typically incorporate external modules or loss terms other than a simple $l_p$-norm into the attack process to achieve imperceptibility, while we argue that such additional designs may not be necessary. In this paper, we rethink the essence of imperceptible attacks and propose two simple yet effective strategies to unleash the potential of PGD, the common and classical attack, for imperceptibility from an optimization perspective. Specifically, the Dynamic Step Size is introduced to find the optimal solution with minimal attack cost towards the decision boundary of the attacked model, and the Adaptive Early Stop strategy is adopted to reduce the redundant strength of adversarial perturbations to the minimum level. The proposed PGD-Imperceptible (PGD-Imp) attack achieves state-of-the-art results in imperceptible adversarial attacks for both untargeted and targeted scenarios. When performing untargeted attacks against ResNet-50, PGD-Imp attains 100$\%$ (+0.3$\%$) ASR, 0.89 (-1.76) $l_2$ distance, and 52.93 (+9.2) PSNR with 57s (-371s) running time, significantly outperforming existing methods.



## **48. WiP: Deception-in-Depth Using Multiple Layers of Deception**

cs.CR

Presented at HoTSoS 2024

**SubmitDate**: 2024-12-21    [abs](http://arxiv.org/abs/2412.16430v1) [paper-pdf](http://arxiv.org/pdf/2412.16430v1)

**Authors**: Jason Landsborough, Neil C. Rowe, Thuy D. Nguyen, Sunny Fugate

**Abstract**: Deception is being increasingly explored as a cyberdefense strategy to protect operational systems. We are studying implementation of deception-in-depth strategies with initially three logical layers: network, host, and data. We draw ideas from military deception, network orchestration, software deception, file deception, fake honeypots, and moving-target defenses. We are building a prototype representing our ideas and will be testing it in several adversarial environments. We hope to show that deploying a broad range of deception techniques can be more effective in protecting systems than deploying single techniques. Unlike traditional deception methods that try to encourage active engagement from attackers to collect intelligence, we focus on deceptions that can be used on real machines to discourage attacks.



## **49. Chain-of-Scrutiny: Detecting Backdoor Attacks for Large Language Models**

cs.CR

**SubmitDate**: 2024-12-21    [abs](http://arxiv.org/abs/2406.05948v2) [paper-pdf](http://arxiv.org/pdf/2406.05948v2)

**Authors**: Xi Li, Yusen Zhang, Renze Lou, Chen Wu, Jiaqi Wang

**Abstract**: Large Language Models (LLMs), especially those accessed via APIs, have demonstrated impressive capabilities across various domains. However, users without technical expertise often turn to (untrustworthy) third-party services, such as prompt engineering, to enhance their LLM experience, creating vulnerabilities to adversarial threats like backdoor attacks. Backdoor-compromised LLMs generate malicious outputs to users when inputs contain specific "triggers" set by attackers. Traditional defense strategies, originally designed for small-scale models, are impractical for API-accessible LLMs due to limited model access, high computational costs, and data requirements. To address these limitations, we propose Chain-of-Scrutiny (CoS) which leverages LLMs' unique reasoning abilities to mitigate backdoor attacks. It guides the LLM to generate reasoning steps for a given input and scrutinizes for consistency with the final output -- any inconsistencies indicating a potential attack. It is well-suited for the popular API-only LLM deployments, enabling detection at minimal cost and with little data. User-friendly and driven by natural language, it allows non-experts to perform the defense independently while maintaining transparency. We validate the effectiveness of CoS through extensive experiments on various tasks and LLMs, with results showing greater benefits for more powerful LLMs.



## **50. EMPRA: Embedding Perturbation Rank Attack against Neural Ranking Models**

cs.IR

**SubmitDate**: 2024-12-20    [abs](http://arxiv.org/abs/2412.16382v1) [paper-pdf](http://arxiv.org/pdf/2412.16382v1)

**Authors**: Amin Bigdeli, Negar Arabzadeh, Ebrahim Bagheri, Charles L. A. Clarke

**Abstract**: Recent research has shown that neural information retrieval techniques may be susceptible to adversarial attacks. Adversarial attacks seek to manipulate the ranking of documents, with the intention of exposing users to targeted content. In this paper, we introduce the Embedding Perturbation Rank Attack (EMPRA) method, a novel approach designed to perform adversarial attacks on black-box Neural Ranking Models (NRMs). EMPRA manipulates sentence-level embeddings, guiding them towards pertinent context related to the query while preserving semantic integrity. This process generates adversarial texts that seamlessly integrate with the original content and remain imperceptible to humans. Our extensive evaluation conducted on the widely-used MS MARCO V1 passage collection demonstrate the effectiveness of EMPRA against a wide range of state-of-the-art baselines in promoting a specific set of target documents within a given ranked results. Specifically, EMPRA successfully achieves a re-ranking of almost 96% of target documents originally ranked between 51-100 to rank within the top 10. Furthermore, EMPRA does not depend on surrogate models for adversarial text generation, enhancing its robustness against different NRMs in realistic settings.



