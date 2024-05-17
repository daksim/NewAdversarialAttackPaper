# Latest Adversarial Attack Papers
**update at 2024-05-17 09:53:10**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Protecting Your LLMs with Information Bottleneck**

通过信息瓶颈保护您的LLC cs.CL

23 pages, 7 figures, 8 tables

**SubmitDate**: 2024-05-16    [abs](http://arxiv.org/abs/2404.13968v2) [paper-pdf](http://arxiv.org/pdf/2404.13968v2)

**Authors**: Zichuan Liu, Zefan Wang, Linjie Xu, Jinyu Wang, Lei Song, Tianchun Wang, Chunlin Chen, Wei Cheng, Jiang Bian

**Abstract**: The advent of large language models (LLMs) has revolutionized the field of natural language processing, yet they might be attacked to produce harmful content. Despite efforts to ethically align LLMs, these are often fragile and can be circumvented by jailbreaking attacks through optimized or manual adversarial prompts. To address this, we introduce the Information Bottleneck Protector (IBProtector), a defense mechanism grounded in the information bottleneck principle, and we modify the objective to avoid trivial solutions. The IBProtector selectively compresses and perturbs prompts, facilitated by a lightweight and trainable extractor, preserving only essential information for the target LLMs to respond with the expected answer. Moreover, we further consider a situation where the gradient is not visible to be compatible with any LLM. Our empirical evaluations show that IBProtector outperforms current defense methods in mitigating jailbreak attempts, without overly affecting response quality or inference speed. Its effectiveness and adaptability across various attack methods and target LLMs underscore the potential of IBProtector as a novel, transferable defense that bolsters the security of LLMs without requiring modifications to the underlying models.

摘要: 大型语言模型的出现给自然语言处理领域带来了革命性的变化，但它们可能会受到攻击，产生有害的内容。尽管努力在道德上调整LLM，但这些往往是脆弱的，可以通过优化或手动对抗性提示通过越狱攻击来绕过。为了解决这个问题，我们引入了信息瓶颈保护器(IBProtector)，这是一种基于信息瓶颈原理的防御机制，我们修改了目标以避免琐碎的解决方案。IBProtector有选择地压缩和干扰提示，由一个轻量级和可训练的提取程序促进，只保留目标LLMS的基本信息，以响应预期的答案。此外，我们还进一步考虑了梯度不可见的情况，以与任何LLM相容。我们的经验评估表明，在不过度影响响应质量或推理速度的情况下，IBProtector在缓解越狱企图方面优于现有的防御方法。它对各种攻击方法和目标LLM的有效性和适应性突显了IBProtector作为一种新型、可转移的防御系统的潜力，无需修改底层模型即可增强LLM的安全性。



## **2. Adversarial Robustness for Visual Grounding of Multimodal Large Language Models**

多模式大型语言模型视觉基础的对抗鲁棒性 cs.CV

ICLR 2024 Workshop on Reliable and Responsible Foundation Models

**SubmitDate**: 2024-05-16    [abs](http://arxiv.org/abs/2405.09981v1) [paper-pdf](http://arxiv.org/pdf/2405.09981v1)

**Authors**: Kuofeng Gao, Yang Bai, Jiawang Bai, Yong Yang, Shu-Tao Xia

**Abstract**: Multi-modal Large Language Models (MLLMs) have recently achieved enhanced performance across various vision-language tasks including visual grounding capabilities. However, the adversarial robustness of visual grounding remains unexplored in MLLMs. To fill this gap, we use referring expression comprehension (REC) as an example task in visual grounding and propose three adversarial attack paradigms as follows. Firstly, untargeted adversarial attacks induce MLLMs to generate incorrect bounding boxes for each object. Besides, exclusive targeted adversarial attacks cause all generated outputs to the same target bounding box. In addition, permuted targeted adversarial attacks aim to permute all bounding boxes among different objects within a single image. Extensive experiments demonstrate that the proposed methods can successfully attack visual grounding capabilities of MLLMs. Our methods not only provide a new perspective for designing novel attacks but also serve as a strong baseline for improving the adversarial robustness for visual grounding of MLLMs.

摘要: 多模式大型语言模型(MLLM)最近在包括视觉基础能力在内的各种视觉语言任务中获得了增强的性能。然而，在最大似然最小二乘法中，视觉接地的对抗稳健性仍未被探索。为了填补这一空白，我们使用指称表达理解(REC)作为视觉基础的示例任务，并提出了以下三种对抗性攻击范式。首先，无针对性的对抗性攻击会导致MLLMS为每个对象生成错误的包围盒。此外，排他性定向对抗性攻击会导致所有生成的输出都指向相同的目标边界框。此外，置换定向对抗性攻击旨在置换单个图像中不同对象之间的所有包围盒。大量实验表明，所提出的方法能够成功地攻击MLLMS的视觉接地能力。我们的方法不仅为设计新的攻击提供了新的视角，而且为提高MLLMS视觉接地的对抗性稳健性提供了强有力的基线。



## **3. Deepfake Generation and Detection: A Benchmark and Survey**

Deepfake生成和检测：基准和调查 cs.CV

We closely follow the latest developments in  https://github.com/flyingby/Awesome-Deepfake-Generation-and-Detection

**SubmitDate**: 2024-05-16    [abs](http://arxiv.org/abs/2403.17881v4) [paper-pdf](http://arxiv.org/pdf/2403.17881v4)

**Authors**: Gan Pei, Jiangning Zhang, Menghan Hu, Zhenyu Zhang, Chengjie Wang, Yunsheng Wu, Guangtao Zhai, Jian Yang, Chunhua Shen, Dacheng Tao

**Abstract**: Deepfake is a technology dedicated to creating highly realistic facial images and videos under specific conditions, which has significant application potential in fields such as entertainment, movie production, digital human creation, to name a few. With the advancements in deep learning, techniques primarily represented by Variational Autoencoders and Generative Adversarial Networks have achieved impressive generation results. More recently, the emergence of diffusion models with powerful generation capabilities has sparked a renewed wave of research. In addition to deepfake generation, corresponding detection technologies continuously evolve to regulate the potential misuse of deepfakes, such as for privacy invasion and phishing attacks. This survey comprehensively reviews the latest developments in deepfake generation and detection, summarizing and analyzing current state-of-the-arts in this rapidly evolving field. We first unify task definitions, comprehensively introduce datasets and metrics, and discuss developing technologies. Then, we discuss the development of several related sub-fields and focus on researching four representative deepfake fields: face swapping, face reenactment, talking face generation, and facial attribute editing, as well as forgery detection. Subsequently, we comprehensively benchmark representative methods on popular datasets for each field, fully evaluating the latest and influential published works. Finally, we analyze challenges and future research directions of the discussed fields.

摘要: 深伪是一项致力于在特定条件下创建高真实感面部图像和视频的技术，在娱乐、电影制作、数字人类创作等领域具有巨大的应用潜力。随着深度学习的进步，以变式自动编码器和生成式对抗性网络为主要代表的技术已经取得了令人印象深刻的生成结果。最近，具有强大发电能力的扩散模型的出现引发了新一轮的研究浪潮。除了深度假冒的生成，相应的检测技术也在不断发展，以规范深度假冒的潜在滥用，例如用于侵犯隐私和网络钓鱼攻击。这项调查全面回顾了深度伪码生成和检测的最新进展，总结和分析了这一快速发展领域的最新技术。我们首先统一任务定义，全面介绍数据集和指标，并讨论开发技术。然后，讨论了几个相关的子领域的发展，重点研究了四个有代表性的深度伪领域：人脸交换、人脸重演、说话人脸生成、人脸属性编辑以及伪造检测。随后，我们在每个领域的热门数据集上综合基准有代表性的方法，充分评价最新和有影响力的已发表作品。最后，分析了所讨论领域面临的挑战和未来的研究方向。



## **4. Infrared Adversarial Car Stickers**

红外对抗汽车贴纸 cs.CV

Accepted by CVPR 2024

**SubmitDate**: 2024-05-16    [abs](http://arxiv.org/abs/2405.09924v1) [paper-pdf](http://arxiv.org/pdf/2405.09924v1)

**Authors**: Xiaopei Zhu, Yuqiu Liu, Zhanhao Hu, Jianmin Li, Xiaolin Hu

**Abstract**: Infrared physical adversarial examples are of great significance for studying the security of infrared AI systems that are widely used in our lives such as autonomous driving. Previous infrared physical attacks mainly focused on 2D infrared pedestrian detection which may not fully manifest its destructiveness to AI systems. In this work, we propose a physical attack method against infrared detectors based on 3D modeling, which is applied to a real car. The goal is to design a set of infrared adversarial stickers to make cars invisible to infrared detectors at various viewing angles, distances, and scenes. We build a 3D infrared car model with real infrared characteristics and propose an infrared adversarial pattern generation method based on 3D mesh shadow. We propose a 3D control points-based mesh smoothing algorithm and use a set of smoothness loss functions to enhance the smoothness of adversarial meshes and facilitate the sticker implementation. Besides, We designed the aluminum stickers and conducted physical experiments on two real Mercedes-Benz A200L cars. Our adversarial stickers hid the cars from Faster RCNN, an object detector, at various viewing angles, distances, and scenes. The attack success rate (ASR) was 91.49% for real cars. In comparison, the ASRs of random stickers and no sticker were only 6.21% and 0.66%, respectively. In addition, the ASRs of the designed stickers against six unseen object detectors such as YOLOv3 and Deformable DETR were between 73.35%-95.80%, showing good transferability of the attack performance across detectors.

摘要: 红外物理对抗例子对于研究自动驾驶等广泛应用于我们生活中的红外AI系统的安全性具有重要意义。以前的红外物理攻击主要集中在2D红外行人检测上，这可能不能充分体现其对人工智能系统的破坏性。在这项工作中，我们提出了一种基于3D建模的对红外探测器的物理攻击方法，并将其应用于真实汽车。目标是设计一套红外对抗性贴纸，使汽车在不同的视角、距离和场景下都能被红外探测器看不见。建立了具有真实红外特征的三维红外汽车模型，提出了一种基于三维网格阴影的红外对抗模式生成方法。提出了一种基于三维控制点的网格光顺算法，并使用一组光滑度损失函数来增强对抗性网格的光滑度，便于粘贴的实现。此外，我们设计了铝制贴纸，并在两辆真实的梅赛德斯-奔驰A200L轿车上进行了物理实验。我们的对抗性贴纸在不同的视角、距离和场景下将汽车隐藏起来，以躲避速度更快的RCNN，一个物体探测器。实车攻击成功率(ASR)为91.49%。相比之下，随机贴纸和不贴纸的ASR分别只有6.21%和0.66%。此外，所设计的标签对YOLOv3、可变形DETR等6种隐形目标探测器的ASR在73.35%~95.80%之间，表现出良好的跨探测器攻击性能的可转移性。



## **5. DiffAM: Diffusion-based Adversarial Makeup Transfer for Facial Privacy Protection**

迪夫AM：基于扩散的对抗性化妆转移，用于面部隐私保护 cs.CV

16 pages, 11 figures

**SubmitDate**: 2024-05-16    [abs](http://arxiv.org/abs/2405.09882v1) [paper-pdf](http://arxiv.org/pdf/2405.09882v1)

**Authors**: Yuhao Sun, Lingyun Yu, Hongtao Xie, Jiaming Li, Yongdong Zhang

**Abstract**: With the rapid development of face recognition (FR) systems, the privacy of face images on social media is facing severe challenges due to the abuse of unauthorized FR systems. Some studies utilize adversarial attack techniques to defend against malicious FR systems by generating adversarial examples. However, the generated adversarial examples, i.e., the protected face images, tend to suffer from subpar visual quality and low transferability. In this paper, we propose a novel face protection approach, dubbed DiffAM, which leverages the powerful generative ability of diffusion models to generate high-quality protected face images with adversarial makeup transferred from reference images. To be specific, we first introduce a makeup removal module to generate non-makeup images utilizing a fine-tuned diffusion model with guidance of textual prompts in CLIP space. As the inverse process of makeup transfer, makeup removal can make it easier to establish the deterministic relationship between makeup domain and non-makeup domain regardless of elaborate text prompts. Then, with this relationship, a CLIP-based makeup loss along with an ensemble attack strategy is introduced to jointly guide the direction of adversarial makeup domain, achieving the generation of protected face images with natural-looking makeup and high black-box transferability. Extensive experiments demonstrate that DiffAM achieves higher visual quality and attack success rates with a gain of 12.98% under black-box setting compared with the state of the arts. The code will be available at https://github.com/HansSunY/DiffAM.

摘要: 随着人脸识别系统的快速发展，由于未经授权的人脸识别系统的滥用，社交媒体上人脸图像的隐私面临着严峻的挑战。一些研究利用对抗性攻击技术通过生成对抗性实例来防御恶意FR系统。然而，生成的敌意例子，即受保护的人脸图像，往往存在视觉质量不佳和可转移性低的问题。在本文中，我们提出了一种新的人脸保护方法，称为DIFAM，它利用扩散模型的强大生成能力来生成高质量的受保护的人脸图像，其中包含从参考图像转换来的对抗性化妆。具体地说，我们首先介绍了一个卸妆模块，该模块利用一个微调的扩散模型，在剪辑空间中以文本提示为指导来生成非化妆图像。作为化妆转移的逆过程，卸妆可以更容易地建立化妆域和非化妆域之间的确定性关系，而不需要考虑精心设计的文字提示。然后，在这种关系下，引入了基于剪辑的化妆损失和系综攻击策略，共同指导对抗性化妆领域的方向，实现了化妆自然、黑盒可转移性高的受保护人脸图像的生成。大量实验表明，与现有技术相比，DIFAM算法在黑盒环境下获得了更高的视觉质量和攻击成功率，提高了12.98%。代码将在https://github.com/HansSunY/DiffAM.上提供



## **6. Box-Free Model Watermarks Are Prone to Black-Box Removal Attacks**

无框模型水印容易受到黑匣子删除攻击 cs.CV

**SubmitDate**: 2024-05-16    [abs](http://arxiv.org/abs/2405.09863v1) [paper-pdf](http://arxiv.org/pdf/2405.09863v1)

**Authors**: Haonan An, Guang Hua, Zhiping Lin, Yuguang Fang

**Abstract**: Box-free model watermarking is an emerging technique to safeguard the intellectual property of deep learning models, particularly those for low-level image processing tasks. Existing works have verified and improved its effectiveness in several aspects. However, in this paper, we reveal that box-free model watermarking is prone to removal attacks, even under the real-world threat model such that the protected model and the watermark extractor are in black boxes. Under this setting, we carry out three studies. 1) We develop an extractor-gradient-guided (EGG) remover and show its effectiveness when the extractor uses ReLU activation only. 2) More generally, for an unknown extractor, we leverage adversarial attacks and design the EGG remover based on the estimated gradients. 3) Under the most stringent condition that the extractor is inaccessible, we design a transferable remover based on a set of private proxy models. In all cases, the proposed removers can successfully remove embedded watermarks while preserving the quality of the processed images, and we also demonstrate that the EGG remover can even replace the watermarks. Extensive experimental results verify the effectiveness and generalizability of the proposed attacks, revealing the vulnerabilities of the existing box-free methods and calling for further research.

摘要: 无盒模型水印是一种新兴的保护深度学习模型知识产权的技术，尤其是用于低层图像处理任务的模型。已有的工作在几个方面验证和改进了它的有效性。然而，在本文中，我们揭示了无盒模型水印容易受到移除攻击，即使在真实世界的威胁模型下，受保护的模型和水印抽取器都在黑盒中。在此背景下，我们开展了三个方面的研究。1)我们开发了一种萃取器-梯度引导(EGG)去除器，并在仅使用RELU激活的情况下展示了其有效性。2)更一般地，对于未知的提取者，我们利用对抗性攻击，并基于估计的梯度来设计鸡蛋去除器。3)在抽取器不可访问的最严格条件下，基于一组私有代理模型设计了一个可转移的抽取器。在所有情况下，所提出的去除器都可以在保持处理图像质量的情况下成功地去除嵌入的水印，并且我们还证明了鸡蛋去除器甚至可以替换水印。大量的实验结果验证了所提出的攻击方法的有效性和泛化能力，揭示了现有去盒方法的弱点，需要进一步研究。



## **7. Manifold Integrated Gradients: Riemannian Geometry for Feature Attribution**

多元积分：特征属性的Riemann几何 cs.LG

Accepted at ICML 2024

**SubmitDate**: 2024-05-16    [abs](http://arxiv.org/abs/2405.09800v1) [paper-pdf](http://arxiv.org/pdf/2405.09800v1)

**Authors**: Eslam Zaher, Maciej Trzaskowski, Quan Nguyen, Fred Roosta

**Abstract**: In this paper, we dive into the reliability concerns of Integrated Gradients (IG), a prevalent feature attribution method for black-box deep learning models. We particularly address two predominant challenges associated with IG: the generation of noisy feature visualizations for vision models and the vulnerability to adversarial attributional attacks. Our approach involves an adaptation of path-based feature attribution, aligning the path of attribution more closely to the intrinsic geometry of the data manifold. Our experiments utilise deep generative models applied to several real-world image datasets. They demonstrate that IG along the geodesics conforms to the curved geometry of the Riemannian data manifold, generating more perceptually intuitive explanations and, subsequently, substantially increasing robustness to targeted attributional attacks.

摘要: 在本文中，我们深入探讨了集成属性（IG）的可靠性问题，这是一种用于黑匣子深度学习模型的流行特征归因方法。我们特别解决了与IG相关的两个主要挑战：视觉模型的有噪特征可视化的生成以及对抗性归因攻击的脆弱性。我们的方法涉及对基于路径的特征属性的调整，将属性路径更紧密地与数据集合的内在几何图形对齐。我们的实验利用应用于多个现实世界图像数据集的深度生成模型。他们证明，沿着测地线的IG符合Riemann数据多管齐下的弯曲几何，从而生成更直观的解释，并随后大幅提高了对有针对性的归因攻击的鲁棒性。



## **8. IBD-PSC: Input-level Backdoor Detection via Parameter-oriented Scaling Consistency**

IBD-OSC：通过面向参数的缩放一致性进行输入级后门检测 cs.LG

Accepted to ICML 2024, 29 pages

**SubmitDate**: 2024-05-16    [abs](http://arxiv.org/abs/2405.09786v1) [paper-pdf](http://arxiv.org/pdf/2405.09786v1)

**Authors**: Linshan Hou, Ruili Feng, Zhongyun Hua, Wei Luo, Leo Yu Zhang, Yiming Li

**Abstract**: Deep neural networks (DNNs) are vulnerable to backdoor attacks, where adversaries can maliciously trigger model misclassifications by implanting a hidden backdoor during model training. This paper proposes a simple yet effective input-level backdoor detection (dubbed IBD-PSC) as a 'firewall' to filter out malicious testing images. Our method is motivated by an intriguing phenomenon, i.e., parameter-oriented scaling consistency (PSC), where the prediction confidences of poisoned samples are significantly more consistent than those of benign ones when amplifying model parameters. In particular, we provide theoretical analysis to safeguard the foundations of the PSC phenomenon. We also design an adaptive method to select BN layers to scale up for effective detection. Extensive experiments are conducted on benchmark datasets, verifying the effectiveness and efficiency of our IBD-PSC method and its resistance to adaptive attacks.

摘要: 深度神经网络（DNN）很容易受到后门攻击，对手可以通过在模型训练期间植入隐藏后门来恶意触发模型错误分类。本文提出了一种简单而有效的输入级后门检测（称为IBD-OSC）作为“防火墙”来过滤恶意测试图像。我们的方法的动机是一个有趣的现象，即，面向参数的缩放一致性（OSC），其中在放大模型参数时，中毒样本的预测置信度明显比良性样本的预测置信度更一致。特别是，我们提供理论分析来捍卫CSC现象的基础。我们还设计了一种自适应方法来选择BN层以扩大规模以进行有效检测。在基准数据集上进行了大量实验，验证了我们的IBD-OSC方法的有效性和效率及其对自适应攻击的抵抗力。



## **9. Towards Evaluating the Robustness of Automatic Speech Recognition Systems via Audio Style Transfer**

通过音频风格转移评估自动语音识别系统的稳健性 cs.SD

Accepted to SecTL (AsiaCCS Workshop) 2024

**SubmitDate**: 2024-05-15    [abs](http://arxiv.org/abs/2405.09470v1) [paper-pdf](http://arxiv.org/pdf/2405.09470v1)

**Authors**: Weifei Jin, Yuxin Cao, Junjie Su, Qi Shen, Kai Ye, Derui Wang, Jie Hao, Ziyao Liu

**Abstract**: In light of the widespread application of Automatic Speech Recognition (ASR) systems, their security concerns have received much more attention than ever before, primarily due to the susceptibility of Deep Neural Networks. Previous studies have illustrated that surreptitiously crafting adversarial perturbations enables the manipulation of speech recognition systems, resulting in the production of malicious commands. These attack methods mostly require adding noise perturbations under $\ell_p$ norm constraints, inevitably leaving behind artifacts of manual modifications. Recent research has alleviated this limitation by manipulating style vectors to synthesize adversarial examples based on Text-to-Speech (TTS) synthesis audio. However, style modifications based on optimization objectives significantly reduce the controllability and editability of audio styles. In this paper, we propose an attack on ASR systems based on user-customized style transfer. We first test the effect of Style Transfer Attack (STA) which combines style transfer and adversarial attack in sequential order. And then, as an improvement, we propose an iterative Style Code Attack (SCA) to maintain audio quality. Experimental results show that our method can meet the need for user-customized styles and achieve a success rate of 82% in attacks, while keeping sound naturalness due to our user study.

摘要: 随着自动语音识别(ASR)系统的广泛应用，其安全问题受到了前所未有的关注，这主要是由于深度神经网络的敏感性。以前的研究表明，秘密地制作敌意扰动能够操纵语音识别系统，导致产生恶意命令。这些攻击方法大多需要在$\ell_p$范数约束下添加噪声扰动，不可避免地会留下人工修改的伪影。最近的研究通过操纵风格向量来合成基于文本到语音(TTS)合成音频的对抗性示例，从而缓解了这一限制。然而，基于优化目标的风格修改显著降低了音频风格的可控性和可编辑性。本文提出了一种基于用户自定义风格转移的ASR系统攻击方法。我们首先测试了风格转移攻击(STA)的效果，该攻击按顺序将风格转移和对抗性攻击结合在一起。然后，作为改进，我们提出了一种迭代样式码攻击(SCA)来保持音频质量。实验结果表明，该方法能够满足用户对个性化风格的需求，攻击成功率达到82%，同时由于我们的用户学习，保持了声音的自然性。



## **10. Inexact Unlearning Needs More Careful Evaluations to Avoid a False Sense of Privacy**

不精确的遗忘需要更仔细的评估，以避免错误的隐私感 cs.LG

**SubmitDate**: 2024-05-15    [abs](http://arxiv.org/abs/2403.01218v2) [paper-pdf](http://arxiv.org/pdf/2403.01218v2)

**Authors**: Jamie Hayes, Ilia Shumailov, Eleni Triantafillou, Amr Khalifa, Nicolas Papernot

**Abstract**: The high cost of model training makes it increasingly desirable to develop techniques for unlearning. These techniques seek to remove the influence of a training example without having to retrain the model from scratch. Intuitively, once a model has unlearned, an adversary that interacts with the model should no longer be able to tell whether the unlearned example was included in the model's training set or not. In the privacy literature, this is known as membership inference. In this work, we discuss adaptations of Membership Inference Attacks (MIAs) to the setting of unlearning (leading to their ``U-MIA'' counterparts). We propose a categorization of existing U-MIAs into ``population U-MIAs'', where the same attacker is instantiated for all examples, and ``per-example U-MIAs'', where a dedicated attacker is instantiated for each example. We show that the latter category, wherein the attacker tailors its membership prediction to each example under attack, is significantly stronger. Indeed, our results show that the commonly used U-MIAs in the unlearning literature overestimate the privacy protection afforded by existing unlearning techniques on both vision and language models. Our investigation reveals a large variance in the vulnerability of different examples to per-example U-MIAs. In fact, several unlearning algorithms lead to a reduced vulnerability for some, but not all, examples that we wish to unlearn, at the expense of increasing it for other examples. Notably, we find that the privacy protection for the remaining training examples may worsen as a consequence of unlearning. We also discuss the fundamental difficulty of equally protecting all examples using existing unlearning schemes, due to the different rates at which examples are unlearned. We demonstrate that naive attempts at tailoring unlearning stopping criteria to different examples fail to alleviate these issues.

摘要: 模型训练的高昂成本使得开发忘却学习的技术变得越来越受欢迎。这些技术寻求消除训练示例的影响，而不必从头开始重新训练模型。直观地说，一旦模型取消学习，与该模型交互的对手应该不再能够判断未学习的示例是否包括在该模型的训练集中。在隐私文献中，这被称为成员关系推断。在这项工作中，我们讨论了成员关系推理攻击(MIA)对遗忘环境的适应(导致它们的‘U-MIA’对应)。我们提出了一种现有U-MIA的分类，其中针对所有示例实例化相同的攻击者，其中针对每个示例实例化一个专用攻击者。我们表明，后一类，其中攻击者根据每个被攻击的例子定制其成员预测，明显更强。事实上，我们的结果表明，遗忘文献中常用的U-MIA高估了现有遗忘技术在视觉和语言模型上提供的隐私保护。我们的调查显示，不同示例对每个示例的U-MIA的脆弱性存在很大差异。事实上，几种忘记算法降低了我们希望忘记的一些(但不是所有)示例的脆弱性，但代价是增加了其他示例的脆弱性。值得注意的是，我们发现，由于遗忘，其余训练样本的隐私保护可能会恶化。我们还讨论了使用现有的遗忘方案平等地保护所有例子的基本困难，因为例子被遗忘的比率不同。我们证明，根据不同的例子调整遗忘停止标准的天真尝试无法缓解这些问题。



## **11. Properties that allow or prohibit transferability of adversarial attacks among quantized networks**

允许或禁止对抗攻击在量化网络之间转移的属性 cs.LG

**SubmitDate**: 2024-05-15    [abs](http://arxiv.org/abs/2405.09598v1) [paper-pdf](http://arxiv.org/pdf/2405.09598v1)

**Authors**: Abhishek Shrestha, Jürgen Großmann

**Abstract**: Deep Neural Networks (DNNs) are known to be vulnerable to adversarial examples. Further, these adversarial examples are found to be transferable from the source network in which they are crafted to a black-box target network. As the trend of using deep learning on embedded devices grows, it becomes relevant to study the transferability properties of adversarial examples among compressed networks. In this paper, we consider quantization as a network compression technique and evaluate the performance of transfer-based attacks when the source and target networks are quantized at different bitwidths. We explore how algorithm specific properties affect transferability by considering various adversarial example generation algorithms. Furthermore, we examine transferability in a more realistic scenario where the source and target networks may differ in bitwidth and other model-related properties like capacity and architecture. We find that although quantization reduces transferability, certain attack types demonstrate an ability to enhance it. Additionally, the average transferability of adversarial examples among quantized versions of a network can be used to estimate the transferability to quantized target networks with varying capacity and architecture.

摘要: 深度神经网络(DNN)很容易受到敌意例子的影响。此外，发现这些敌意的例子可以从它们被制作的源网络转移到黑盒目标网络。随着在嵌入式设备上使用深度学习的趋势的发展，研究对抗性例子在压缩网络中的可转移性变得非常重要。在本文中，我们将量化作为一种网络压缩技术，并对源网络和目标网络在不同比特宽度上进行量化时基于传输的攻击的性能进行评估。通过考虑不同的对抗性实例生成算法，我们探讨了算法的特定属性如何影响可转移性。此外，我们在更现实的场景中检查了可转移性，其中源网络和目标网络的位宽以及其他与模型相关的属性(如容量和体系结构)可能不同。我们发现，虽然量化降低了可转移性，但某些攻击类型表现出了增强可转移性的能力。此外，网络的量化版本之间的对抗性例子的平均可转移性可以用来估计到具有不同容量和体系结构的量化目标网络的可转移性。



## **12. "Do Anything Now": Characterizing and Evaluating In-The-Wild Jailbreak Prompts on Large Language Models**

“立即做任何事情”：描述和评估大型语言模型上的In-The-Wild越狱预言 cs.CR

**SubmitDate**: 2024-05-15    [abs](http://arxiv.org/abs/2308.03825v2) [paper-pdf](http://arxiv.org/pdf/2308.03825v2)

**Authors**: Xinyue Shen, Zeyuan Chen, Michael Backes, Yun Shen, Yang Zhang

**Abstract**: The misuse of large language models (LLMs) has drawn significant attention from the general public and LLM vendors. One particular type of adversarial prompt, known as jailbreak prompt, has emerged as the main attack vector to bypass the safeguards and elicit harmful content from LLMs. In this paper, employing our new framework JailbreakHub, we conduct a comprehensive analysis of 1,405 jailbreak prompts spanning from December 2022 to December 2023. We identify 131 jailbreak communities and discover unique characteristics of jailbreak prompts and their major attack strategies, such as prompt injection and privilege escalation. We also observe that jailbreak prompts increasingly shift from online Web communities to prompt-aggregation websites and 28 user accounts have consistently optimized jailbreak prompts over 100 days. To assess the potential harm caused by jailbreak prompts, we create a question set comprising 107,250 samples across 13 forbidden scenarios. Leveraging this dataset, our experiments on six popular LLMs show that their safeguards cannot adequately defend jailbreak prompts in all scenarios. Particularly, we identify five highly effective jailbreak prompts that achieve 0.95 attack success rates on ChatGPT (GPT-3.5) and GPT-4, and the earliest one has persisted online for over 240 days. We hope that our study can facilitate the research community and LLM vendors in promoting safer and regulated LLMs.

摘要: 大型语言模型(LLM)的滥用引起了公众和LLM供应商的极大关注。一种特殊类型的对抗性提示，即越狱提示，已经成为绕过安全措施并从LLMS引出有害内容的主要攻击媒介。在本文中，我们使用我们的新框架JailBreak Hub，对从2022年12月到2023年12月的1405个越狱提示进行了全面的分析。我们识别了131个越狱社区，发现了越狱提示的独特特征及其主要攻击策略，如提示注入和特权提升。我们还观察到越狱提示越来越多地从在线网络社区转移到提示聚合网站，28个用户账户在100天内持续优化越狱提示。为了评估越狱提示造成的潜在危害，我们创建了一个包含13个禁止场景的107,250个样本的问题集。利用这个数据集，我们在六个流行的LLM上的实验表明，它们的保护措施不足以在所有场景中防御越狱提示。特别是，我们确定了五个高效的越狱提示，它们在ChatGPT(GPT-3.5)和GPT-4上的攻击成功率达到了0.95%，其中最早的一个在线时间超过了240天。我们希望我们的研究能够促进研究界和LLM供应商推广更安全和规范的LLM。



## **13. Cross-Input Certified Training for Universal Perturbations**

针对普遍扰动的交叉输入认证培训 cs.LG

21 pages, 5 figures

**SubmitDate**: 2024-05-15    [abs](http://arxiv.org/abs/2405.09176v1) [paper-pdf](http://arxiv.org/pdf/2405.09176v1)

**Authors**: Changming Xu, Gagandeep Singh

**Abstract**: Existing work in trustworthy machine learning primarily focuses on single-input adversarial perturbations. In many real-world attack scenarios, input-agnostic adversarial attacks, e.g. universal adversarial perturbations (UAPs), are much more feasible. Current certified training methods train models robust to single-input perturbations but achieve suboptimal clean and UAP accuracy, thereby limiting their applicability in practical applications. We propose a novel method, CITRUS, for certified training of networks robust against UAP attackers. We show in an extensive evaluation across different datasets, architectures, and perturbation magnitudes that our method outperforms traditional certified training methods on standard accuracy (up to 10.3\%) and achieves SOTA performance on the more practical certified UAP accuracy metric.

摘要: 可信机器学习的现有工作主要集中在单输入对抗性扰动上。在许多现实世界的攻击场景中，输入不可知的对抗性攻击，例如通用对抗性扰动（UPC），更为可行。当前经过认证的训练方法训练模型对单输入扰动具有鲁棒性，但实现了次优的干净和UAP准确性，从而限制了其在实际应用中的适用性。我们提出了一种新颖的方法CITRUS，用于对抵御UAP攻击者的强大网络进行认证训练。我们在对不同数据集、架构和扰动幅度的广泛评估中表明，我们的方法在标准准确性方面优于传统认证训练方法（高达10.3%），并在更实用的认证UAP准确性指标方面实现了SOTA性能。



## **14. The Economic Limits of Permissionless Consensus**

无许可共识的经济局限 cs.DC

**SubmitDate**: 2024-05-15    [abs](http://arxiv.org/abs/2405.09173v1) [paper-pdf](http://arxiv.org/pdf/2405.09173v1)

**Authors**: Eric Budish, Andrew Lewis-Pye, Tim Roughgarden

**Abstract**: The purpose of a consensus protocol is to keep a distributed network of nodes "in sync," even in the presence of an unpredictable communication network and adversarial behavior by some of the participating nodes. In the permissionless setting, these nodes may be operated by unknown players, with each player free to use multiple identifiers and to start or stop running the protocol at any time. Establishing that a permissionless consensus protocol is "secure" thus requires both a distributed computing argument (that the protocol guarantees consistency and liveness unless the fraction of adversarial participation is sufficiently large) and an economic argument (that carrying out an attack would be prohibitively expensive for an attacker). There is a mature toolbox for assembling arguments of the former type; the goal of this paper is to lay the foundations for arguments of the latter type.   An ideal permissionless consensus protocol would, in addition to satisfying standard consistency and liveness guarantees, render consistency violations prohibitively expensive for the attacker without collateral damage to honest participants. We make this idea precise with our notion of the EAAC (expensive to attack in the absence of collapse) property, and prove the following results:   1. In the synchronous and dynamically available setting, with an adversary that controls at least one-half of the overall resources, no protocol can be EAAC.   2. In the partially synchronous and quasi-permissionless setting, with an adversary that controls at least one-third of the overall resources, no protocol can be EAAC.   3. In the synchronous and quasi-permissionless setting, there is a proof-of-stake protocol that, provided the adversary controls less than two-thirds of the overall stake, satisfies the EAAC property.   All three results are optimal with respect to the size of the adversary.

摘要: 共识协议的目的是保持分布式节点网络的同步，即使在一些参与节点存在不可预测的通信网络和敌对行为的情况下也是如此。在未经许可的设置中，这些节点可以由未知玩家操作，每个玩家可以自由使用多个标识符，并且可以随时开始或停止运行协议。因此，要确定未经许可的共识协议是“安全的”，既需要分布式计算论证(该协议保证一致性和活性，除非敌方参与的比例足够大)，也需要经济论证(对攻击者来说，实施攻击的代价高得令人望而却步)。有一个成熟的工具箱用于组合前一种类型的参数；本文的目标是为后一种类型的参数奠定基础。一个理想的未经许可的协商一致协议，除了满足标准的一致性和活跃性保证外，还将使违反一致性的行为对攻击者来说代价高昂，而不会对诚实的参与者造成附带损害。我们用我们的EAAC(在没有崩溃的情况下攻击代价很高)属性的概念精确地表达了这个想法，并证明了以下结果：1.在同步和动态可用的环境中，如果对手控制了至少一半的总资源，则没有协议可以成为EAAC。2.在部分同步和准无许可的设置中，在对手控制至少三分之一的总资源的情况下，没有协议可以是EAAC。3.在同步和准无许可的设置中，存在一种风险证明协议，如果对手控制的总风险少于三分之二，则满足EAAC属性。就对手的规模而言，这三个结果都是最优的。



## **15. Dynamic Adversarial Attacks on Autonomous Driving Systems**

对自动驾驶系统的动态对抗攻击 cs.RO

**SubmitDate**: 2024-05-15    [abs](http://arxiv.org/abs/2312.06701v2) [paper-pdf](http://arxiv.org/pdf/2312.06701v2)

**Authors**: Amirhosein Chahe, Chenan Wang, Abhishek Jeyapratap, Kaidi Xu, Lifeng Zhou

**Abstract**: This paper introduces an attacking mechanism to challenge the resilience of autonomous driving systems. Specifically, we manipulate the decision-making processes of an autonomous vehicle by dynamically displaying adversarial patches on a screen mounted on another moving vehicle. These patches are optimized to deceive the object detection models into misclassifying targeted objects, e.g., traffic signs. Such manipulation has significant implications for critical multi-vehicle interactions such as intersection crossing and lane changing, which are vital for safe and efficient autonomous driving systems. Particularly, we make four major contributions. First, we introduce a novel adversarial attack approach where the patch is not co-located with its target, enabling more versatile and stealthy attacks. Moreover, our method utilizes dynamic patches displayed on a screen, allowing for adaptive changes and movement, enhancing the flexibility and performance of the attack. To do so, we design a Screen Image Transformation Network (SIT-Net), which simulates environmental effects on the displayed images, narrowing the gap between simulated and real-world scenarios. Further, we integrate a positional loss term into the adversarial training process to increase the success rate of the dynamic attack. Finally, we shift the focus from merely attacking perceptual systems to influencing the decision-making algorithms of self-driving systems. Our experiments demonstrate the first successful implementation of such dynamic adversarial attacks in real-world autonomous driving scenarios, paving the way for advancements in the field of robust and secure autonomous driving.

摘要: 本文介绍了一种攻击机制来挑战自动驾驶系统的弹性。具体地说，我们通过在安装在另一辆移动车辆上的屏幕上动态显示敌对补丁来操纵自动车辆的决策过程。这些补丁被优化以欺骗对象检测模型误分类目标对象，例如交通标志。这种操纵对交叉路口和换道等关键的多车辆相互作用具有重要影响，而这些对安全高效的自动驾驶系统至关重要。特别是，我们做出了四大贡献。首先，我们引入了一种新颖的对抗性攻击方法，其中补丁不与目标位于同一位置，从而实现了更多功能和隐蔽的攻击。此外，我们的方法利用屏幕上显示的动态补丁，允许自适应变化和移动，增强了攻击的灵活性和性能。为此，我们设计了一个屏幕图像转换网络(SIT-Net)，它模拟了环境对显示图像的影响，缩小了模拟场景和真实场景之间的差距。此外，我们还将位置损失项融入到对抗性训练过程中，以提高动态攻击的成功率。最后，我们将重点从仅仅攻击感知系统转移到影响自动驾驶系统的决策算法。我们的实验首次成功地在真实世界的自动驾驶场景中实现了这种动态对抗性攻击，为稳健和安全的自动驾驶领域的进步铺平了道路。



## **16. Optimizing Sensor Network Design for Multiple Coverage**

优化传感器网络设计以实现多覆盖 cs.LG

**SubmitDate**: 2024-05-15    [abs](http://arxiv.org/abs/2405.09096v1) [paper-pdf](http://arxiv.org/pdf/2405.09096v1)

**Authors**: Lukas Taus, Yen-Hsi Richard Tsai

**Abstract**: Sensor placement optimization methods have been studied extensively. They can be applied to a wide range of applications, including surveillance of known environments, optimal locations for 5G towers, and placement of missile defense systems. However, few works explore the robustness and efficiency of the resulting sensor network concerning sensor failure or adversarial attacks. This paper addresses this issue by optimizing for the least number of sensors to achieve multiple coverage of non-simply connected domains by a prescribed number of sensors. We introduce a new objective function for the greedy (next-best-view) algorithm to design efficient and robust sensor networks and derive theoretical bounds on the network's optimality. We further introduce a Deep Learning model to accelerate the algorithm for near real-time computations. The Deep Learning model requires the generation of training examples. Correspondingly, we show that understanding the geometric properties of the training data set provides important insights into the performance and training process of deep learning techniques. Finally, we demonstrate that a simple parallel version of the greedy approach using a simpler objective can be highly competitive.

摘要: 传感器布局优化方法得到了广泛的研究。它们可以应用于广泛的应用，包括对已知环境的监视，5G塔的最佳位置，以及导弹防御系统的布置。然而，很少有文献探讨传感器网络在传感器故障或敌意攻击下的健壮性和有效性。本文通过优化最少的传感器数量来解决这一问题，从而在规定的传感器数量下实现对非单连通区域的多次覆盖。为了设计高效、健壮的传感器网络，我们为贪婪(Next-Best-view)算法引入了一个新的目标函数，并给出了网络最优性的理论界。我们进一步引入了深度学习模型来加速算法，以实现近实时计算。深度学习模型需要生成训练实例。相应地，我们表明，理解训练数据集的几何属性可以为深度学习技术的性能和训练过程提供重要的见解。最后，我们证明了贪婪方法的简单并行版本使用更简单的目标可以具有很强的竞争力。



## **17. The Pitfalls and Promise of Conformal Inference Under Adversarial Attacks**

对抗性攻击下保形推理的陷阱和希望 cs.LG

ICML2024

**SubmitDate**: 2024-05-14    [abs](http://arxiv.org/abs/2405.08886v1) [paper-pdf](http://arxiv.org/pdf/2405.08886v1)

**Authors**: Ziquan Liu, Yufei Cui, Yan Yan, Yi Xu, Xiangyang Ji, Xue Liu, Antoni B. Chan

**Abstract**: In safety-critical applications such as medical imaging and autonomous driving, where decisions have profound implications for patient health and road safety, it is imperative to maintain both high adversarial robustness to protect against potential adversarial attacks and reliable uncertainty quantification in decision-making. With extensive research focused on enhancing adversarial robustness through various forms of adversarial training (AT), a notable knowledge gap remains concerning the uncertainty inherent in adversarially trained models. To address this gap, this study investigates the uncertainty of deep learning models by examining the performance of conformal prediction (CP) in the context of standard adversarial attacks within the adversarial defense community. It is first unveiled that existing CP methods do not produce informative prediction sets under the commonly used $l_{\infty}$-norm bounded attack if the model is not adversarially trained, which underpins the importance of adversarial training for CP. Our paper next demonstrates that the prediction set size (PSS) of CP using adversarially trained models with AT variants is often worse than using standard AT, inspiring us to research into CP-efficient AT for improved PSS. We propose to optimize a Beta-weighting loss with an entropy minimization regularizer during AT to improve CP-efficiency, where the Beta-weighting loss is shown to be an upper bound of PSS at the population level by our theoretical analysis. Moreover, our empirical study on four image classification datasets across three popular AT baselines validates the effectiveness of the proposed Uncertainty-Reducing AT (AT-UR).

摘要: 在医疗成像和自动驾驶等安全关键型应用中，决策对患者的健康和道路安全具有深远的影响，因此必须保持高度的对抗性健壮性，以防止潜在的对抗性攻击，并在决策中保持可靠的不确定性量化。随着广泛的研究集中在通过各种形式的对抗性训练(AT)来增强对抗性稳健性，关于对抗性训练模型中固有的不确定性的显著知识差距仍然存在。为了解决这一差距，本研究通过检验共形预测(CP)在对抗性防御社区内标准对抗性攻击的背景下的性能来调查深度学习模型的不确定性。首先揭示了在常用的L范数有界攻击下，如果模型没有经过对抗性训练，现有的对抗性预测方法不能产生有信息的预测集，这支持了对抗性训练对于对抗性预测的重要性。接下来，我们证明了使用带有AT变体的对抗性训练模型的预测集大小(PSS)往往比使用标准AT要差，这促使我们研究CP-Efficient AT以改进PSS。我们提出用最小熵正则化来优化AT过程中的Beta加权损失以提高CP效率，理论分析表明Beta加权损失是PSS在种群水平上的一个上界。此外，我们在三个流行的AT基线上的四个图像分类数据集上的实证研究验证了所提出的减少不确定性的AT(AT-UR)的有效性。



## **18. S3C2 Summit 2024-03: Industry Secure Supply Chain Summit**

S3 C2峰会2024-03：行业安全供应链峰会 cs.CR

This is our WIP paper on the Summit. More versions will be released  soon

**SubmitDate**: 2024-05-14    [abs](http://arxiv.org/abs/2405.08762v1) [paper-pdf](http://arxiv.org/pdf/2405.08762v1)

**Authors**: Greg Tystahl, Yasemin Acar, Michel Cukier, William Enck, Christian Kastner, Alexandros Kapravelos, Dominik Wermke, Laurie Williams

**Abstract**: Supply chain security has become a very important vector to consider when defending against adversary attacks. Due to this, more and more developers are keen on improving their supply chains to make them more robust against future threats. On March 7th, 2024 researchers from the Secure Software Supply Chain Center (S3C2) gathered 14 industry leaders, developers and consumers of the open source ecosystem to discuss the state of supply chain security. The goal of the summit is to share insights between companies and developers alike to foster new collaborations and ideas moving forward. Through this meeting, participants were questions on best practices and thoughts how to improve things for the future. In this paper we summarize the responses and discussions of the summit. The panel questions can be found in the appendix.

摘要: 供应链安全已成为防御对手攻击时需要考虑的一个非常重要的载体。因此，越来越多的开发商热衷于改善其供应链，使其更强大地应对未来的威胁。2024年3月7日，安全软件供应链中心（S3 C2）的研究人员聚集了开源生态系统的14位行业领导者、开发人员和消费者，讨论供应链安全状况。峰会的目标是在公司和开发人员之间分享见解，以促进新的合作和前进的想法。通过这次会议，与会者就最佳实践和如何为未来改进的想法提出了问题。本文总结了峰会的回应和讨论。小组问题可在附录中找到。



## **19. Design and Analysis of Resilient Vehicular Platoon Systems over Wireless Networks**

无线网络上弹性车辆排系统的设计与分析 eess.SY

6 pages, 4 figures, in submission of Globecom 2024

**SubmitDate**: 2024-05-14    [abs](http://arxiv.org/abs/2405.08706v1) [paper-pdf](http://arxiv.org/pdf/2405.08706v1)

**Authors**: Tingyu Shui, Walid Saad

**Abstract**: Connected vehicular platoons provide a promising solution to improve traffic efficiency and ensure road safety. Vehicles in a platoon utilize on-board sensors and wireless vehicle-to-vehicle (V2V) links to share traffic information for cooperative adaptive cruise control. To process real-time control and alert information, there is a need to ensure clock synchronization among the platoon's vehicles. However, adversaries can jeopardize the operation of the platoon by attacking the local clocks of vehicles, leading to clock offsets with the platoon's reference clock. In this paper, a novel framework is proposed for analyzing the resilience of vehicular platoons that are connected using V2V links. In particular, a resilient design based on a diffusion protocol is proposed to re-synchronize the attacked vehicle through wireless V2V links thereby mitigating the impact of variance of the transmission delay during recovery. Then, a novel metric named temporal conditional mean exceedance is defined and analyzed in order to characterize the resilience of the platoon. Subsequently, the conditions pertaining to the V2V links and recovery time needed for a resilient design are derived. Numerical results show that the proposed resilient design is feasible in face of a nine-fold increase in the variance of transmission delay compared to a baseline designed for reliability. Moreover, the proposed approach improves the reliability, defined as the probability of meeting a desired clock offset error requirement, by 45% compared to the baseline.

摘要: 联网的车辆排为提高交通效率和确保道路安全提供了一个很有前途的解决方案。排中的车辆利用车载传感器和无线车载(V2V)链路共享交通信息，以实现协作自适应巡航控制。为了处理实时的控制和警报信息，需要确保排内车辆之间的时钟同步。然而，敌人可以通过攻击车辆的本地时钟来危及排的运行，导致时钟与排的参考时钟发生偏差。本文提出了一种新的分析车辆排弹性的框架，这些排通过V2V链路连接。特别是，提出了一种基于扩散协议的弹性设计，通过无线V2V链路重新同步受攻击的车辆，从而减轻恢复过程中传输延迟变化的影响。然后，定义并分析了一种新的度量--时间条件平均超越度，以刻画排的抗弹能力。随后，推导出了弹性设计所需的V2V链路和恢复时间的条件。数值结果表明，当传输时延的方差比可靠性设计的基线增加9倍时，所提出的弹性设计是可行的。此外，与基线相比，所提出的方法将可靠性(定义为满足期望的时钟偏移误差要求的概率)提高了45%。



## **20. Quantum Oblivious LWE Sampling and Insecurity of Standard Model Lattice-Based SNARKs**

量子不经意LWE采样和标准模型基于网格的SNARK的不安全性 cs.CR

**SubmitDate**: 2024-05-14    [abs](http://arxiv.org/abs/2401.03807v2) [paper-pdf](http://arxiv.org/pdf/2401.03807v2)

**Authors**: Thomas Debris-Alazard, Pouria Fallahpour, Damien Stehlé

**Abstract**: The Learning With Errors ($\mathsf{LWE}$) problem asks to find $\mathbf{s}$ from an input of the form $(\mathbf{A}, \mathbf{b} = \mathbf{A}\mathbf{s}+\mathbf{e}) \in (\mathbb{Z}/q\mathbb{Z})^{m \times n} \times (\mathbb{Z}/q\mathbb{Z})^{m}$, for a vector $\mathbf{e}$ that has small-magnitude entries. In this work, we do not focus on solving $\mathsf{LWE}$ but on the task of sampling instances. As these are extremely sparse in their range, it may seem plausible that the only way to proceed is to first create $\mathbf{s}$ and $\mathbf{e}$ and then set $\mathbf{b} = \mathbf{A}\mathbf{s}+\mathbf{e}$. In particular, such an instance sampler knows the solution. This raises the question whether it is possible to obliviously sample $(\mathbf{A}, \mathbf{A}\mathbf{s}+\mathbf{e})$, namely, without knowing the underlying $\mathbf{s}$. A variant of the assumption that oblivious $\mathsf{LWE}$ sampling is hard has been used in a series of works to analyze the security of candidate constructions of Succinct Non interactive Arguments of Knowledge (SNARKs). As the assumption is related to $\mathsf{LWE}$, these SNARKs have been conjectured to be secure in the presence of quantum adversaries.   Our main result is a quantum polynomial-time algorithm that samples well-distributed $\mathsf{LWE}$ instances while provably not knowing the solution, under the assumption that $\mathsf{LWE}$ is hard. Moreover, the approach works for a vast range of $\mathsf{LWE}$ parametrizations, including those used in the above-mentioned SNARKs. This invalidates the assumptions used in their security analyses, although it does not yield attacks against the constructions themselves.

摘要: 错误学习($\mathbf{lwe}$)问题要求从以下形式的输入中查找$\mathbf{S}$，对于具有小震级条目的向量$\mathbf{e}$\mathbf{e}$/mathbf{e}$\mathbf{e}$\mathbf{e}$(mathbf{Z}/q\mathbb{Z})^{m}$。在这项工作中，我们关注的不是$\mathsf{LWE}$，而是采样实例的任务。因为它们在它们的范围内非常稀疏，所以似乎唯一的继续的方法是首先创建$\mathbf{S}$和$\mathbf{e}$，然后设置$\mathbf{b}=\mathbf{A}\mathbf{S}+\mathbf{e}$。特别是，这样的实例采样器知道解决方案。这就提出了一个问题：是否有可能在不知道潜在的$\mathbf{S}$的情况下，对$(\mathbf{A}，\mathbf{A}\mathbf{S}+\mathbf{e})$进行不经意的采样。在一系列工作中，忽略的抽样是困难的这一假设的变体被用于分析简明的非交互知识论元(SNARK)候选构造的安全性。由于该假设与$\mathsf{lwe}$有关，因此人们猜测这些snarks在量子对手的存在下是安全的。我们的主要结果是一个量子多项式时间算法，它在假设$\mathsf{LWE}$是困难的假设下，对均匀分布的$\mathsf{LWE}$实例进行采样，但可证明不知道解。此外，该方法适用于大量的$\mathsf{lwe}$参数化，包括在上述snarks中使用的那些。这使他们在安全分析中使用的假设无效，尽管它不会产生针对建筑本身的攻击。



## **21. PLeak: Prompt Leaking Attacks against Large Language Model Applications**

PLeak：针对大型语言模型应用程序的提示泄露攻击 cs.CR

To appear in the Proceedings of The ACM Conference on Computer and  Communications Security (CCS), 2024

**SubmitDate**: 2024-05-14    [abs](http://arxiv.org/abs/2405.06823v2) [paper-pdf](http://arxiv.org/pdf/2405.06823v2)

**Authors**: Bo Hui, Haolin Yuan, Neil Gong, Philippe Burlina, Yinzhi Cao

**Abstract**: Large Language Models (LLMs) enable a new ecosystem with many downstream applications, called LLM applications, with different natural language processing tasks. The functionality and performance of an LLM application highly depend on its system prompt, which instructs the backend LLM on what task to perform. Therefore, an LLM application developer often keeps a system prompt confidential to protect its intellectual property. As a result, a natural attack, called prompt leaking, is to steal the system prompt from an LLM application, which compromises the developer's intellectual property. Existing prompt leaking attacks primarily rely on manually crafted queries, and thus achieve limited effectiveness.   In this paper, we design a novel, closed-box prompt leaking attack framework, called PLeak, to optimize an adversarial query such that when the attacker sends it to a target LLM application, its response reveals its own system prompt. We formulate finding such an adversarial query as an optimization problem and solve it with a gradient-based method approximately. Our key idea is to break down the optimization goal by optimizing adversary queries for system prompts incrementally, i.e., starting from the first few tokens of each system prompt step by step until the entire length of the system prompt.   We evaluate PLeak in both offline settings and for real-world LLM applications, e.g., those on Poe, a popular platform hosting such applications. Our results show that PLeak can effectively leak system prompts and significantly outperforms not only baselines that manually curate queries but also baselines with optimized queries that are modified and adapted from existing jailbreaking attacks. We responsibly reported the issues to Poe and are still waiting for their response. Our implementation is available at this repository: https://github.com/BHui97/PLeak.

摘要: 大型语言模型(LLM)支持一个新的生态系统，该生态系统具有许多下游应用程序，称为LLM应用程序，具有不同的自然语言处理任务。LLM应用程序的功能和性能高度依赖于其系统提示符，系统提示符指示后端LLM执行什么任务。因此，LLM应用程序开发人员通常会对系统提示保密，以保护其知识产权。因此，一种称为提示泄漏的自然攻击是从LLM应用程序中窃取系统提示，这会损害开发人员的知识产权。现有的即时泄漏攻击主要依赖于手动创建的查询，因此效果有限。在本文中，我们设计了一个新颖的封闭盒提示泄漏攻击框架PLeak，用于优化敌意查询，使其在攻击者将其发送到目标LLM应用程序时，其响应显示其自己的系统提示。我们将寻找这样一个敌意查询描述为一个优化问题，并用基于梯度的方法近似求解。我们的核心思想是通过对系统提示的敌意查询进行增量优化来打破优化目标，即从每个系统提示的前几个令牌开始逐步优化，直到系统提示的整个长度。我们在离线设置和现实世界的LLM应用程序(例如，托管此类应用程序的流行平台PoE上的应用程序)中对PLeak进行评估。我们的结果表明，PLeak能够有效地泄露系统提示，不仅显著优于手动管理查询的基线，而且显著优于从现有越狱攻击中修改和调整的优化查询的基线。我们负责任地向PoE报告了这些问题，并仍在等待他们的回应。我们的实现可从以下存储库获得：https://github.com/BHui97/PLeak.



## **22. Certifying Robustness of Graph Convolutional Networks for Node Perturbation with Polyhedra Abstract Interpretation**

用多边形抽象解释证明图卷积网络对节点扰动的鲁棒性 cs.LG

**SubmitDate**: 2024-05-14    [abs](http://arxiv.org/abs/2405.08645v1) [paper-pdf](http://arxiv.org/pdf/2405.08645v1)

**Authors**: Boqi Chen, Kristóf Marussy, Oszkár Semeráth, Gunter Mussbacher, Dániel Varró

**Abstract**: Graph convolutional neural networks (GCNs) are powerful tools for learning graph-based knowledge representations from training data. However, they are vulnerable to small perturbations in the input graph, which makes them susceptible to input faults or adversarial attacks. This poses a significant problem for GCNs intended to be used in critical applications, which need to provide certifiably robust services even in the presence of adversarial perturbations. We propose an improved GCN robustness certification technique for node classification in the presence of node feature perturbations. We introduce a novel polyhedra-based abstract interpretation approach to tackle specific challenges of graph data and provide tight upper and lower bounds for the robustness of the GCN. Experiments show that our approach simultaneously improves the tightness of robustness bounds as well as the runtime performance of certification. Moreover, our method can be used during training to further improve the robustness of GCNs.

摘要: 图卷积神经网络(GCNS)是从训练数据中学习基于图的知识表示的有力工具。然而，它们容易受到输入图中的小扰动的影响，这使得它们容易受到输入错误或对手攻击的影响。这对希望用于关键应用的GCNS提出了一个重大问题，即使在存在对抗性扰动的情况下，GCNS也需要提供可证明的健壮性服务。针对存在节点特征扰动的节点分类问题，提出了一种改进的GCN健壮性认证技术。我们引入了一种新的基于多面体的抽象解释方法来解决图形数据的特定挑战，并为GCN的健壮性提供了严格的上下界。实验表明，我们的方法同时提高了健壮界的紧密性和认证的运行时性能。此外，我们的方法还可以在训练过程中使用，以进一步提高GCNS的鲁棒性。



## **23. HookChain: A new perspective for Bypassing EDR Solutions**

HookChain：询问EDR解决方案的新视角 cs.CR

46 pages, 22 figures, HookChain, Bypass EDR, Evading EDR, IAT Hook,  Halo's Gate

**SubmitDate**: 2024-05-14    [abs](http://arxiv.org/abs/2404.16856v2) [paper-pdf](http://arxiv.org/pdf/2404.16856v2)

**Authors**: Helvio Carvalho Junior

**Abstract**: In the current digital security ecosystem, where threats evolve rapidly and with complexity, companies developing Endpoint Detection and Response (EDR) solutions are in constant search for innovations that not only keep up but also anticipate emerging attack vectors. In this context, this article introduces the HookChain, a look from another perspective at widely known techniques, which when combined, provide an additional layer of sophisticated evasion against traditional EDR systems. Through a precise combination of IAT Hooking techniques, dynamic SSN resolution, and indirect system calls, HookChain redirects the execution flow of Windows subsystems in a way that remains invisible to the vigilant eyes of EDRs that only act on Ntdll.dll, without requiring changes to the source code of the applications and malwares involved. This work not only challenges current conventions in cybersecurity but also sheds light on a promising path for future protection strategies, leveraging the understanding that continuous evolution is key to the effectiveness of digital security. By developing and exploring the HookChain technique, this study significantly contributes to the body of knowledge in endpoint security, stimulating the development of more robust and adaptive solutions that can effectively address the ever-changing dynamics of digital threats. This work aspires to inspire deep reflection and advancement in the research and development of security technologies that are always several steps ahead of adversaries.   UNDER CONSTRUCTION RESEARCH: This paper is not the final version, as it is currently undergoing final tests against several EDRs. We expect to release the final version by August 2024.

摘要: 在当前的数字安全生态系统中，威胁发展迅速且复杂，开发终端检测和响应(EDR)解决方案的公司正在不断寻找创新，不仅要跟上形势，还要预测新出现的攻击媒介。在此背景下，本文介绍了HookChain，从另一个角度介绍了广为人知的技术，这些技术结合在一起时，提供了针对传统EDR系统的另一层复杂规避。通过IAT挂钩技术、动态SSN解析和间接系统调用的精确组合，HookChain以一种仅作用于Ntdll.dll的EDR保持警惕的眼睛看不到的方式重定向Windows子系统的执行流，而不需要更改所涉及的应用程序和恶意软件的源代码。这项工作不仅挑战了目前的网络安全惯例，而且还揭示了未来保护战略的一条有希望的道路，充分利用了对持续演变是数字安全有效性的关键的理解。通过开发和探索HookChain技术，这项研究对终端安全方面的知识体系做出了重大贡献，刺激了能够有效应对不断变化的数字威胁动态的更健壮和适应性更强的解决方案的开发。这项工作旨在激发人们对安全技术研究和开发的深刻反思和进步，这些技术总是领先于对手几步。正在进行的研究：这篇论文不是最终版本，因为它目前正在接受针对几个EDR的最终测试。我们预计在2024年8月之前发布最终版本。



## **24. Secure Aggregation Meets Sparsification in Decentralized Learning**

安全聚合遇到去中心化学习中的稀疏化 cs.LG

**SubmitDate**: 2024-05-14    [abs](http://arxiv.org/abs/2405.07708v2) [paper-pdf](http://arxiv.org/pdf/2405.07708v2)

**Authors**: Sayan Biswas, Anne-Marie Kermarrec, Rafael Pires, Rishi Sharma, Milos Vujasinovic

**Abstract**: Decentralized learning (DL) faces increased vulnerability to privacy breaches due to sophisticated attacks on machine learning (ML) models. Secure aggregation is a computationally efficient cryptographic technique that enables multiple parties to compute an aggregate of their private data while keeping their individual inputs concealed from each other and from any central aggregator. To enhance communication efficiency in DL, sparsification techniques are used, selectively sharing only the most crucial parameters or gradients in a model, thereby maintaining efficiency without notably compromising accuracy. However, applying secure aggregation to sparsified models in DL is challenging due to the transmission of disjoint parameter sets by distinct nodes, which can prevent masks from canceling out effectively. This paper introduces CESAR, a novel secure aggregation protocol for DL designed to be compatible with existing sparsification mechanisms. CESAR provably defends against honest-but-curious adversaries and can be formally adapted to counteract collusion between them. We provide a foundational understanding of the interaction between the sparsification carried out by the nodes and the proportion of the parameters shared under CESAR in both colluding and non-colluding environments, offering analytical insight into the working and applicability of the protocol. Experiments on a network with 48 nodes in a 3-regular topology show that with random subsampling, CESAR is always within 0.5% accuracy of decentralized parallel stochastic gradient descent (D-PSGD), while adding only 11% of data overhead. Moreover, it surpasses the accuracy on TopK by up to 0.3% on independent and identically distributed (IID) data.

摘要: 由于对机器学习(ML)模型的复杂攻击，分散学习(DL)面临着更多的隐私泄露漏洞。安全聚合是一种计算高效的加密技术，它使多方能够计算他们的私有数据的聚合，同时保持他们的个人输入对彼此和任何中央聚集器隐藏。为了提高DL中的通信效率，使用了稀疏化技术，选择性地仅共享模型中最关键的参数或梯度，从而在不显著影响精度的情况下保持效率。然而，将安全聚合应用于DL中的稀疏模型是具有挑战性的，这是因为不同的节点传输不相交的参数集，这会阻止掩码有效地抵消。本文介绍了一种新的面向下行链路的安全聚集协议--CESAR，该协议与现有的稀疏机制兼容。塞萨尔被证明可以防御诚实但好奇的对手，并可以正式修改以抵消他们之间的勾结。我们提供了对节点执行的稀疏化和在CESAR下在共谋和非共谋环境下共享的参数比例之间的交互的基础性理解，为协议的工作和适用性提供了分析洞察力。在一个具有48个节点的3正则拓扑网络上的实验表明，在随机子采样的情况下，CESAR算法的精度始终在分散并行随机梯度下降算法(D-PSGD)的0.5%以内，而增加的数据开销仅为11%。此外，在独立同分布(IID)数据上，它比TOPK的准确率高出0.3%。



## **25. UnMarker: A Universal Attack on Defensive Watermarking**

UnMarker：对防御性水印的普遍攻击 cs.CR

**SubmitDate**: 2024-05-14    [abs](http://arxiv.org/abs/2405.08363v1) [paper-pdf](http://arxiv.org/pdf/2405.08363v1)

**Authors**: Andre Kassis, Urs Hengartner

**Abstract**: Reports regarding the misuse of $\textit{Generative AI}$ ($\textit{GenAI}$) to create harmful deepfakes are emerging daily. Recently, defensive watermarking, which enables $\textit{GenAI}$ providers to hide fingerprints in their images to later use for deepfake detection, has been on the rise. Yet, its potential has not been fully explored. We present $\textit{UnMarker}$ -- the first practical $\textit{universal}$ attack on defensive watermarking. Unlike existing attacks, $\textit{UnMarker}$ requires no detector feedback, no unrealistic knowledge of the scheme or similar models, and no advanced denoising pipelines that may not be available. Instead, being the product of an in-depth analysis of the watermarking paradigm revealing that robust schemes must construct their watermarks in the spectral amplitudes, $\textit{UnMarker}$ employs two novel adversarial optimizations to disrupt the spectra of watermarked images, erasing the watermarks. Evaluations against the $\textit{SOTA}$ prove its effectiveness, not only defeating traditional schemes while retaining superior quality compared to existing attacks but also breaking $\textit{semantic}$ watermarks that alter the image's structure, reducing the best detection rate to $43\%$ and rendering them useless. To our knowledge, $\textit{UnMarker}$ is the first practical attack on $\textit{semantic}$ watermarks, which have been deemed the future of robust watermarking. $\textit{UnMarker}$ casts doubts on the very penitential of this countermeasure and exposes its paradoxical nature as designing schemes for robustness inevitably compromises other robustness aspects.

摘要: 关于滥用$\textit{生成性人工智能}$($\textit{GenAI}$)来创建有害的深度假冒的报告每天都在出现。最近，防御性水印正在兴起，它使$\textit{GenAI}$提供商能够隐藏他们图像中的指纹，以便以后用于深度假冒检测。然而，它的潜力还没有得到充分的开发。我们提出了第一个实用的针对防御性水印的$\textit{UnMarker}$攻击。与现有的攻击不同，$\textit{UnMarker}$不需要检测器反馈，不需要不切实际的方案或类似模型的知识，也不需要可能无法获得的高级去噪管道。相反，作为对水印范例的深入分析的产物，稳健方案必须在频谱幅度上构造水印，它采用了两种新的对抗性优化来扰乱水印图像的频谱，从而消除水印。对该算法的评估证明了该算法的有效性，该算法不仅能在保持现有攻击质量的同时击败传统方案，还能破解改变图像结构的水印，使最佳检测率降至43美元，并使其毫无用处。据我们所知，$\textit{UnMarker}$是针对被认为是未来稳健水印的$\textit{语义}$水印的第一个实用攻击。由于健壮性设计方案不可避免地会折衷于其他健壮性方面，因此对这一对策的悔过性提出了质疑，并暴露了它的悖论性质。



## **26. SpeechGuard: Exploring the Adversarial Robustness of Multimodal Large Language Models**

SpeechGuard：探索多模式大型语言模型的对抗鲁棒性 cs.CL

9+6 pages, Submitted to ACL 2024

**SubmitDate**: 2024-05-14    [abs](http://arxiv.org/abs/2405.08317v1) [paper-pdf](http://arxiv.org/pdf/2405.08317v1)

**Authors**: Raghuveer Peri, Sai Muralidhar Jayanthi, Srikanth Ronanki, Anshu Bhatia, Karel Mundnich, Saket Dingliwal, Nilaksh Das, Zejiang Hou, Goeric Huybrechts, Srikanth Vishnubhotla, Daniel Garcia-Romero, Sundararajan Srinivasan, Kyu J Han, Katrin Kirchhoff

**Abstract**: Integrated Speech and Large Language Models (SLMs) that can follow speech instructions and generate relevant text responses have gained popularity lately. However, the safety and robustness of these models remains largely unclear. In this work, we investigate the potential vulnerabilities of such instruction-following speech-language models to adversarial attacks and jailbreaking. Specifically, we design algorithms that can generate adversarial examples to jailbreak SLMs in both white-box and black-box attack settings without human involvement. Additionally, we propose countermeasures to thwart such jailbreaking attacks. Our models, trained on dialog data with speech instructions, achieve state-of-the-art performance on spoken question-answering task, scoring over 80% on both safety and helpfulness metrics. Despite safety guardrails, experiments on jailbreaking demonstrate the vulnerability of SLMs to adversarial perturbations and transfer attacks, with average attack success rates of 90% and 10% respectively when evaluated on a dataset of carefully designed harmful questions spanning 12 different toxic categories. However, we demonstrate that our proposed countermeasures reduce the attack success significantly.

摘要: 集成的语音和大型语言模型(SLM)可以遵循语音指令并生成相关的文本响应，最近得到了广泛的应用。然而，这些模型的安全性和稳健性在很大程度上仍不清楚。在这项工作中，我们调查了这种遵循指令的语音语言模型在对抗攻击和越狱时的潜在脆弱性。具体地说，我们设计的算法可以生成白盒和黑盒攻击环境下的越狱SLM的对抗性示例，而不需要人工参与。此外，我们还提出了挫败此类越狱攻击的对策。我们的模型在对话数据和语音指令上进行了训练，在口语问答任务中实现了最先进的性能，在安全性和有助性指标上都获得了80%以上的分数。尽管有安全护栏，但越狱实验证明了SLM在对抗性扰动和转移攻击中的脆弱性，当对12个不同有毒类别的精心设计的有害问题集进行评估时，平均攻击成功率分别为90%和10%。然而，我们证明我们提出的对策显著降低了攻击的成功率。



## **27. Adversarial Machine Learning Threats to Spacecraft**

对抗性机器学习对航天器的威胁 cs.LG

Preprint

**SubmitDate**: 2024-05-14    [abs](http://arxiv.org/abs/2405.08834v1) [paper-pdf](http://arxiv.org/pdf/2405.08834v1)

**Authors**: Rajiv Thummala, Shristi Sharma, Matteo Calabrese, Gregory Falco

**Abstract**: Spacecraft are among the earliest autonomous systems. Their ability to function without a human in the loop have afforded some of humanity's grandest achievements. As reliance on autonomy grows, space vehicles will become increasingly vulnerable to attacks designed to disrupt autonomous processes-especially probabilistic ones based on machine learning. This paper aims to elucidate and demonstrate the threats that adversarial machine learning (AML) capabilities pose to spacecraft. First, an AML threat taxonomy for spacecraft is introduced. Next, we demonstrate the execution of AML attacks against spacecraft through experimental simulations using NASA's Core Flight System (cFS) and NASA's On-board Artificial Intelligence Research (OnAIR) Platform. Our findings highlight the imperative for incorporating AML-focused security measures in spacecraft that engage autonomy.

摘要: 航天器是最早的自主系统之一。它们在没有人类参与的情况下发挥作用的能力为人类带来了一些最伟大的成就。随着对自主性依赖的增长，太空飞行器将越来越容易受到旨在破坏自主过程的攻击，尤其是基于机器学习的概率过程。本文旨在阐明和演示对抗性机器学习（ML）能力对航天器构成的威胁。首先，介绍了航天器的APL威胁分类。接下来，我们使用NASA的核心飞行系统（cFS）和NASA的机载人工智能研究（OnAir）平台，通过实验模拟来展示针对航天器的APL攻击的执行情况。我们的研究结果强调了在实现自主性的航天器中纳入以AML为重点的安全措施的必要性。



## **28. Adversarial Nibbler: An Open Red-Teaming Method for Identifying Diverse Harms in Text-to-Image Generation**

对抗性Nibbler：一种用于识别文本到图像生成中各种伤害的开放式红团队方法 cs.CY

10 pages, 6 figures

**SubmitDate**: 2024-05-14    [abs](http://arxiv.org/abs/2403.12075v3) [paper-pdf](http://arxiv.org/pdf/2403.12075v3)

**Authors**: Jessica Quaye, Alicia Parrish, Oana Inel, Charvi Rastogi, Hannah Rose Kirk, Minsuk Kahng, Erin van Liemt, Max Bartolo, Jess Tsang, Justin White, Nathan Clement, Rafael Mosquera, Juan Ciro, Vijay Janapa Reddi, Lora Aroyo

**Abstract**: With the rise of text-to-image (T2I) generative AI models reaching wide audiences, it is critical to evaluate model robustness against non-obvious attacks to mitigate the generation of offensive images. By focusing on ``implicitly adversarial'' prompts (those that trigger T2I models to generate unsafe images for non-obvious reasons), we isolate a set of difficult safety issues that human creativity is well-suited to uncover. To this end, we built the Adversarial Nibbler Challenge, a red-teaming methodology for crowdsourcing a diverse set of implicitly adversarial prompts. We have assembled a suite of state-of-the-art T2I models, employed a simple user interface to identify and annotate harms, and engaged diverse populations to capture long-tail safety issues that may be overlooked in standard testing. The challenge is run in consecutive rounds to enable a sustained discovery and analysis of safety pitfalls in T2I models.   In this paper, we present an in-depth account of our methodology, a systematic study of novel attack strategies and discussion of safety failures revealed by challenge participants. We also release a companion visualization tool for easy exploration and derivation of insights from the dataset. The first challenge round resulted in over 10k prompt-image pairs with machine annotations for safety. A subset of 1.5k samples contains rich human annotations of harm types and attack styles. We find that 14% of images that humans consider harmful are mislabeled as ``safe'' by machines. We have identified new attack strategies that highlight the complexity of ensuring T2I model robustness. Our findings emphasize the necessity of continual auditing and adaptation as new vulnerabilities emerge. We are confident that this work will enable proactive, iterative safety assessments and promote responsible development of T2I models.

摘要: 随着文本到图像(T2I)生成式人工智能模型的兴起，评估模型对非明显攻击的稳健性以减少攻击性图像的生成至关重要。通过关注“隐含的对抗性”提示(那些由于不明显的原因触发T2I模型生成不安全图像的提示)，我们隔离了一组人类创造力非常适合揭示的困难安全问题。为此，我们建立了对抗性Nibbler挑战赛，这是一种用于众包各种隐含对抗性提示的红团队方法论。我们组装了一套最先进的T2I模型，使用简单的用户界面来识别和注释危害，并让不同的人群参与捕获标准测试中可能被忽视的长尾安全问题。该挑战赛分连续几轮进行，以持续发现和分析T2I型号的安全隐患。在这篇文章中，我们介绍了我们的方法，对新的攻击策略进行了系统的研究，并讨论了挑战参与者揭示的安全故障。我们还发布了一个配套的可视化工具，用于轻松探索和从数据集获得洞察力。第一轮挑战赛产生了10000多个带有机器注释的提示图像对，以确保安全。1.5K样本的子集包含丰富的危害类型和攻击风格的人类注释。我们发现，在人类认为有害的图像中，14%被机器错误地贴上了“安全”的标签。我们已经确定了新的攻击策略，这些策略突出了确保T2I模型健壮性的复杂性。我们的发现强调了随着新漏洞的出现而持续审计和适应的必要性。我们相信，这项工作将使主动、迭代的安全评估成为可能，并促进负责任的T2I模型的开发。



## **29. RAID: A Shared Benchmark for Robust Evaluation of Machine-Generated Text Detectors**

RAGE：机器生成文本检测器稳健评估的共享基准 cs.CL

To appear at ACL 2024

**SubmitDate**: 2024-05-13    [abs](http://arxiv.org/abs/2405.07940v1) [paper-pdf](http://arxiv.org/pdf/2405.07940v1)

**Authors**: Liam Dugan, Alyssa Hwang, Filip Trhlik, Josh Magnus Ludan, Andrew Zhu, Hainiu Xu, Daphne Ippolito, Chris Callison-Burch

**Abstract**: Many commercial and open-source models claim to detect machine-generated text with very high accuracy (99\% or higher). However, very few of these detectors are evaluated on shared benchmark datasets and even when they are, the datasets used for evaluation are insufficiently challenging -- lacking variations in sampling strategy, adversarial attacks, and open-source generative models. In this work we present RAID: the largest and most challenging benchmark dataset for machine-generated text detection. RAID includes over 6 million generations spanning 11 models, 8 domains, 11 adversarial attacks and 4 decoding strategies. Using RAID, we evaluate the out-of-domain and adversarial robustness of 8 open- and 4 closed-source detectors and find that current detectors are easily fooled by adversarial attacks, variations in sampling strategies, repetition penalties, and unseen generative models. We release our dataset and tools to encourage further exploration into detector robustness.

摘要: 许多商业和开源模型声称可以以非常高的准确性（99%或更高）检测机器生成的文本。然而，这些检测器中很少有在共享基准数据集上进行评估，即使如此，用于评估的数据集也不够具有挑战性--缺乏采样策略、对抗性攻击和开源生成模型的变化。在这项工作中，我们介绍了RAIDA：用于机器生成文本检测的最大、最具挑战性的基准数据集。磁盘阵列包含超过600万代，涵盖11个模型、8个域、11种对抗性攻击和4种解码策略。使用RAIDGE，我们评估了8个开源检测器和4个开源检测器的域外和对抗稳健性，发现当前的检测器很容易被对抗攻击、采样策略的变化、重复惩罚和看不见的生成模型所愚弄。我们发布了我们的数据集和工具，以鼓励进一步探索检测器的稳健性。



## **30. On the Adversarial Robustness of Learning-based Image Compression Against Rate-Distortion Attacks**

基于学习的图像压缩对率失真攻击的对抗鲁棒性 eess.IV

**SubmitDate**: 2024-05-13    [abs](http://arxiv.org/abs/2405.07717v1) [paper-pdf](http://arxiv.org/pdf/2405.07717v1)

**Authors**: Chenhao Wu, Qingbo Wu, Haoran Wei, Shuai Chen, Lei Wang, King Ngi Ngan, Fanman Meng, Hongliang Li

**Abstract**: Despite demonstrating superior rate-distortion (RD) performance, learning-based image compression (LIC) algorithms have been found to be vulnerable to malicious perturbations in recent studies. Adversarial samples in these studies are designed to attack only one dimension of either bitrate or distortion, targeting a submodel with a specific compression ratio. However, adversaries in real-world scenarios are neither confined to singular dimensional attacks nor always have control over compression ratios. This variability highlights the inadequacy of existing research in comprehensively assessing the adversarial robustness of LIC algorithms in practical applications. To tackle this issue, this paper presents two joint rate-distortion attack paradigms at both submodel and algorithm levels, i.e., Specific-ratio Rate-Distortion Attack (SRDA) and Agnostic-ratio Rate-Distortion Attack (ARDA). Additionally, a suite of multi-granularity assessment tools is introduced to evaluate the attack results from various perspectives. On this basis, extensive experiments on eight prominent LIC algorithms are conducted to offer a thorough analysis of their inherent vulnerabilities. Furthermore, we explore the efficacy of two defense techniques in improving the performance under joint rate-distortion attacks. The findings from these experiments can provide a valuable reference for the development of compression algorithms with enhanced adversarial robustness.

摘要: 尽管基于学习的图像压缩(LIC)算法表现出优异的率失真(RD)性能，但在最近的研究中发现LIC算法容易受到恶意干扰。这些研究中的对抗性样本被设计为仅攻击比特率或失真的一个维度，以具有特定压缩比的子模型为目标。然而，现实世界场景中的对手既不限于单维攻击，也不总是控制压缩比。这种可变性突出了现有研究在全面评估LIC算法在实际应用中的对抗性稳健性方面的不足。针对这一问题，本文从子模型和算法两个层面提出了两种联合的率失真攻击范式，即特定比率率失真攻击(SRDA)和不可知率失真攻击(ARDA)。此外，还引入了一套多粒度评估工具，从不同角度对攻击结果进行评估。在此基础上，对8种重要的LIC算法进行了广泛的实验，深入分析了它们的固有漏洞。此外，我们还探讨了两种防御技术在提高联合码率失真攻击下性能的有效性。这些实验结果可以为开发具有增强对抗性的压缩算法提供有价值的参考。



## **31. DP-DCAN: Differentially Private Deep Contrastive Autoencoder Network for Single-cell Clustering**

DP-DCAN：用于单细胞集群的差异私有深度对比自动编码器网络 cs.LG

**SubmitDate**: 2024-05-13    [abs](http://arxiv.org/abs/2311.03410v2) [paper-pdf](http://arxiv.org/pdf/2311.03410v2)

**Authors**: Huifa Li, Jie Fu, Zhili Chen, Xiaomin Yang, Haitao Liu, Xinpeng Ling

**Abstract**: Single-cell RNA sequencing (scRNA-seq) is important to transcriptomic analysis of gene expression. Recently, deep learning has facilitated the analysis of high-dimensional single-cell data. Unfortunately, deep learning models may leak sensitive information about users. As a result, Differential Privacy (DP) is increasingly used to protect privacy. However, existing DP methods usually perturb whole neural networks to achieve differential privacy, and hence result in great performance overheads. To address this challenge, in this paper, we take advantage of the uniqueness of the autoencoder that it outputs only the dimension-reduced vector in the middle of the network, and design a Differentially Private Deep Contrastive Autoencoder Network (DP-DCAN) by partial network perturbation for single-cell clustering. Since only partial network is added with noise, the performance improvement is obvious and twofold: one part of network is trained with less noise due to a bigger privacy budget, and the other part is trained without any noise. Experimental results of six datasets have verified that DP-DCAN is superior to the traditional DP scheme with whole network perturbation. Moreover, DP-DCAN demonstrates strong robustness to adversarial attacks.

摘要: 单细胞RNA测序(scRNA-seq)对于基因表达的转录分析具有重要意义。最近，深度学习为高维单细胞数据的分析提供了便利。不幸的是，深度学习模型可能会泄露用户的敏感信息。因此，差异隐私(DP)越来越多地被用来保护隐私。然而，现有的DP方法通常会对整个神经网络进行扰动以实现差分隐私，从而导致很大的性能开销。为了应对这一挑战，本文利用自动编码器只输出网络中间降维向量的独特性，设计了一种基于部分网络扰动的差分私有深度对比自动编码器网络(DP-DCAN)，用于单小区聚类。由于只有部分网络添加了噪声，因此性能改善是明显的，而且是双重的：一部分网络的训练由于较大的隐私预算而噪声较小，而另一部分网络训练时没有任何噪声。在6个数据集上的实验结果表明，DP-DCAN算法优于传统的全网扰动下的DP算法。此外，DP-DCAN对敌方攻击表现出很强的鲁棒性。



## **32. CrossCert: A Cross-Checking Detection Approach to Patch Robustness Certification for Deep Learning Models**

CrossCert：一种交叉检查检测方法，为深度学习模型修补鲁棒性认证 cs.SE

23 pages, 2 figures, accepted by FSE 2024 (The ACM International  Conference on the Foundations of Software Engineering)

**SubmitDate**: 2024-05-13    [abs](http://arxiv.org/abs/2405.07668v1) [paper-pdf](http://arxiv.org/pdf/2405.07668v1)

**Authors**: Qilin Zhou, Zhengyuan Wei, Haipeng Wang, Bo Jiang, W. K. Chan

**Abstract**: Patch robustness certification is an emerging kind of defense technique against adversarial patch attacks with provable guarantees. There are two research lines: certified recovery and certified detection. They aim to label malicious samples with provable guarantees correctly and issue warnings for malicious samples predicted to non-benign labels with provable guarantees, respectively. However, existing certified detection defenders suffer from protecting labels subject to manipulation, and existing certified recovery defenders cannot systematically warn samples about their labels. A certified defense that simultaneously offers robust labels and systematic warning protection against patch attacks is desirable. This paper proposes a novel certified defense technique called CrossCert. CrossCert formulates a novel approach by cross-checking two certified recovery defenders to provide unwavering certification and detection certification. Unwavering certification ensures that a certified sample, when subjected to a patched perturbation, will always be returned with a benign label without triggering any warnings with a provable guarantee. To our knowledge, CrossCert is the first certified detection technique to offer this guarantee. Our experiments show that, with a slightly lower performance than ViP and comparable performance with PatchCensor in terms of detection certification, CrossCert certifies a significant proportion of samples with the guarantee of unwavering certification.

摘要: 补丁健壮性认证是一种新兴的防御恶意补丁攻击的技术，具有可证明的保证。有两条研究路线：认证的回收和认证的检测。它们的目标是正确地标记具有可证明保证的恶意样本，并分别对预测为具有可证明保证的非良性标签的恶意样本发出警告。然而，现有的认证检测防御者遭受着保护受操纵的标签的困扰，并且现有的认证恢复防御者不能系统地警告样本有关其标签的信息。同时提供坚固标签和针对补丁攻击的系统警告保护的认证防御是可取的。提出了一种新的认证防御技术CrossCert。CrossCert通过交叉检查两个经过认证的恢复防御者来制定一种新的方法，以提供坚定不移的认证和检测认证。坚定不移的认证确保了经过认证的样品在受到修补扰动时，始终会被退回带有良性标签的产品，而不会触发任何带有可证明保证的警告。据我们所知，CrossCert是第一个提供这一保证的认证检测技术。我们的实验表明，在检测认证方面，CrossCert的性能略低于VIP，但在检测认证方面与补丁检查器相当，可以在保证认证坚定不移的情况下认证相当比例的样本。



## **33. Backdoor Removal for Generative Large Language Models**

生成性大型语言模型的后门删除 cs.CR

**SubmitDate**: 2024-05-13    [abs](http://arxiv.org/abs/2405.07667v1) [paper-pdf](http://arxiv.org/pdf/2405.07667v1)

**Authors**: Haoran Li, Yulin Chen, Zihao Zheng, Qi Hu, Chunkit Chan, Heshan Liu, Yangqiu Song

**Abstract**: With rapid advances, generative large language models (LLMs) dominate various Natural Language Processing (NLP) tasks from understanding to reasoning. Yet, language models' inherent vulnerabilities may be exacerbated due to increased accessibility and unrestricted model training on massive textual data from the Internet. A malicious adversary may publish poisoned data online and conduct backdoor attacks on the victim LLMs pre-trained on the poisoned data. Backdoored LLMs behave innocuously for normal queries and generate harmful responses when the backdoor trigger is activated. Despite significant efforts paid to LLMs' safety issues, LLMs are still struggling against backdoor attacks. As Anthropic recently revealed, existing safety training strategies, including supervised fine-tuning (SFT) and Reinforcement Learning from Human Feedback (RLHF), fail to revoke the backdoors once the LLM is backdoored during the pre-training stage. In this paper, we present Simulate and Eliminate (SANDE) to erase the undesired backdoored mappings for generative LLMs. We initially propose Overwrite Supervised Fine-tuning (OSFT) for effective backdoor removal when the trigger is known. Then, to handle the scenarios where the trigger patterns are unknown, we integrate OSFT into our two-stage framework, SANDE. Unlike previous works that center on the identification of backdoors, our safety-enhanced LLMs are able to behave normally even when the exact triggers are activated. We conduct comprehensive experiments to show that our proposed SANDE is effective against backdoor attacks while bringing minimal harm to LLMs' powerful capability without any additional access to unbackdoored clean models. We will release the reproducible code.

摘要: 随着研究的深入，从理解到推理的各种自然语言处理任务都被生成性大语言模型(LLMS)所支配。然而，语言模型固有的脆弱性可能会因为可访问性的提高和对来自互联网的海量文本数据的不受限制的模型训练而加剧。恶意对手可能会在网上发布有毒数据，并对受害者LLM进行后门攻击，这些LLM预先训练了有毒数据。后门LLM在正常查询中的行为是无害的，并在激活后门触发器时生成有害的响应。尽管在LLMS的安全问题上付出了巨大的努力，但LLMS仍在努力应对后门攻击。正如人类最近揭示的那样，现有的安全培训策略，包括监督微调(SFT)和从人类反馈的强化学习(RLHF)，一旦LLM在培训前阶段后退，就无法取消后门。在这篇文章中，我们提出了模拟和消除(SANDE)来消除生成式LLMS中不需要的回溯映射。我们最初提出了覆盖监督精调(OSFT)，用于在已知触发器的情况下有效地删除后门。然后，为了处理触发模式未知的场景，我们将OSFT集成到我们的两阶段框架SANDE中。与以前以识别后门为中心的工作不同，我们的安全增强型LLM即使在准确的触发器被激活时也能够正常运行。我们进行了全面的实验，以表明我们提出的SANDE可以有效地抵御后门攻击，同时对LLMS的强大功能造成的损害最小，而不需要额外访问未后门的干净模型。我们将发布可重现的代码。



## **34. Environmental Matching Attack Against Unmanned Aerial Vehicles Object Detection**

针对无人机目标检测的环境匹配攻击 cs.CV

**SubmitDate**: 2024-05-13    [abs](http://arxiv.org/abs/2405.07595v1) [paper-pdf](http://arxiv.org/pdf/2405.07595v1)

**Authors**: Dehong Kong, Siyuan Liang, Wenqi Ren

**Abstract**: Object detection techniques for Unmanned Aerial Vehicles (UAVs) rely on Deep Neural Networks (DNNs), which are vulnerable to adversarial attacks. Nonetheless, adversarial patches generated by existing algorithms in the UAV domain pay very little attention to the naturalness of adversarial patches. Moreover, imposing constraints directly on adversarial patches makes it difficult to generate patches that appear natural to the human eye while ensuring a high attack success rate. We notice that patches are natural looking when their overall color is consistent with the environment. Therefore, we propose a new method named Environmental Matching Attack(EMA) to address the issue of optimizing the adversarial patch under the constraints of color. To the best of our knowledge, this paper is the first to consider natural patches in the domain of UAVs. The EMA method exploits strong prior knowledge of a pretrained stable diffusion to guide the optimization direction of the adversarial patch, where the text guidance can restrict the color of the patch. To better match the environment, the contrast and brightness of the patch are appropriately adjusted. Instead of optimizing the adversarial patch itself, we optimize an adversarial perturbation patch which initializes to zero so that the model can better trade off attacking performance and naturalness. Experiments conducted on the DroneVehicle and Carpk datasets have shown that our work can reach nearly the same attack performance in the digital attack(no greater than 2 in mAP$\%$), surpass the baseline method in the physical specific scenarios, and exhibit a significant advantage in terms of naturalness in visualization and color difference with the environment.

摘要: 无人机(UAV)的目标检测技术依赖于深度神经网络(DNN)，而DNN容易受到对手的攻击。尽管如此，无人机领域的现有算法生成的对抗性补丁很少关注对抗性补丁的自然性。此外，直接对敌方补丁施加限制，使得在确保高攻击成功率的同时，很难生成人眼看起来很自然的补丁。我们注意到，当补丁的整体颜色与环境一致时，它们看起来很自然。因此，我们提出了一种新的方法--环境匹配攻击(EMA)来解决颜色约束下敌方补丁的优化问题。据我们所知，本文是第一次考虑无人机领域中的自然斑块。EMA方法利用预先训练的稳定扩散的强先验知识来指导对抗性补丁的优化方向，其中文本指导可以限制补丁的颜色。为了更好地匹配环境，贴片的对比度和亮度进行了适当的调整。我们没有优化对抗性补丁本身，而是优化了一个初始化为零的对抗性扰动补丁，使模型能够更好地在攻击性能和自然性之间进行权衡。在DroneVehicle和Carpk数据集上进行的实验表明，我们的工作在数字攻击中可以达到几乎相同的攻击性能(MAP$不大于2)，在物理特定场景中超过基线方法，在可视化和与环境的色差方面显示出显著的优势。



## **35. Towards Rational Consensus in Honest Majority**

在诚实的多数中走向理性共识 cs.GT

**SubmitDate**: 2024-05-13    [abs](http://arxiv.org/abs/2405.07557v1) [paper-pdf](http://arxiv.org/pdf/2405.07557v1)

**Authors**: Varul Srivastava, Sujit Gujar

**Abstract**: Distributed consensus protocols reach agreement among $n$ players in the presence of $f$ adversaries; different protocols support different values of $f$. Existing works study this problem for different adversary types (captured by threat models). There are three primary threat models: (i) Crash fault tolerance (CFT), (ii) Byzantine fault tolerance (BFT), and (iii) Rational fault tolerance (RFT), each more general than the previous. Agreement in repeated rounds on both (1) the proposed value in each round and (2) the ordering among agreed-upon values across multiple rounds is called Atomic BroadCast (ABC). ABC is more generalized than consensus and is employed in blockchains.   This work studies ABC under the RFT threat model. We consider $t$ byzantine and $k$ rational adversaries among $n$ players. We also study different types of rational players based on their utility towards (1) liveness attack, (2) censorship or (3) disagreement (forking attack). We study the problem of ABC under this general threat model in partially-synchronous networks. We show (1) ABC is impossible for $n/3< (t+k) <n/2$ if rational players prefer liveness or censorship attacks and (2) the consensus protocol proposed by Ranchal-Pedrosa and Gramoli cannot be generalized to solve ABC due to insecure Nash equilibrium (resulting in disagreement). For ABC in partially synchronous network settings, we propose a novel protocol \textsf{pRFT}(practical Rational Fault Tolerance). We show \textsf{pRFT} achieves ABC if (a) rational players prefer only disagreement attacks and (b) $t < \frac{n}{4}$ and $(t + k) < \frac{n}{2}$. In \textsf{pRFT}, we incorporate accountability (capturing deviating players) within the protocol by leveraging honest players. We also show that the message complexity of \textsf{pRFT} is at par with the best consensus protocols that guarantee accountability.

摘要: 分布式共识协议在存在$f$对手的情况下，在$n$参与者之间达成协议；不同的协议支持不同的$f$值。现有的研究工作针对不同的对手类型(通过威胁模型捕获)来研究这一问题。有三种主要的威胁模型：(I)崩溃容错(CFT)，(Ii)拜占庭容错(BFT)和(Iii)理性容错(RFT)，每一种模型都比以前的模型更具一般性。在重复回合中就(1)每轮中的建议值和(2)多轮中商定的值之间的排序这两个方面达成一致称为原子广播(ABC)。ABC比共识更一般化，并被应用于区块链。本文研究了RFT威胁模型下的ABC。我们在$n$玩家中考虑$t$拜占庭和$k$理性对手。我们还研究了不同类型的理性玩家，根据他们对(1)活性攻击，(2)审查或(3)分歧(分叉攻击)的效用。我们研究了部分同步网络中这种一般威胁模型下的ABC问题。我们证明了：(1)如果理性参与者喜欢活跃度或审查攻击，则$n/3<(t+k)<n/2$时ABC是不可能的；(2)由于不安全的纳什均衡(导致不一致)，Ranchal-Pedrosa和Gramoli提出的共识协议不能推广到求解ABC。对于部分同步网络环境下的ABC，我们提出了一种新的协议-.我们证明了，如果(A)理性参与者只喜欢不一致攻击，并且(B)$t<\frac{n}{4}$和$(t+k)<\frac{n}{2}$，则文本sf{pRFT}达到ABC。在\extsf{pRFT}中，我们通过利用诚实的参与者将责任(捕获偏离规则的参与者)纳入到协议中。我们还证明了Textsf{pRFT}的消息复杂性与保证可问责性的最佳共识协议相当。



## **36. On Securing Analog Lagrange Coded Computing from Colluding Adversaries**

保护模拟拉格朗日编码计算免受共谋对手的侵害 cs.IT

To appear in the proceedings of IEEE ISIT 2024

**SubmitDate**: 2024-05-13    [abs](http://arxiv.org/abs/2405.07454v1) [paper-pdf](http://arxiv.org/pdf/2405.07454v1)

**Authors**: Rimpi Borah, J. Harshan

**Abstract**: Analog Lagrange Coded Computing (ALCC) is a recently proposed coded computing paradigm wherein certain computations over analog datasets can be efficiently performed using distributed worker nodes through floating point implementation. While ALCC is known to preserve privacy of data from the workers, it is not resilient to adversarial workers that return erroneous computation results. Pointing at this security vulnerability, we focus on securing ALCC from a wide range of non-colluding and colluding adversarial workers. As a foundational step, we make use of error-correction algorithms for Discrete Fourier Transform (DFT) codes to build novel algorithms to nullify the erroneous computations returned from the adversaries. Furthermore, when such a robust ALCC is implemented in practical settings, we show that the presence of precision errors in the system can be exploited by the adversaries to propose novel colluding attacks to degrade the computation accuracy. As the main takeaway, we prove a counter-intuitive result that not all the adversaries should inject noise in their computations in order to optimally degrade the accuracy of the ALCC framework. This is the first work of its kind to address the vulnerability of ALCC against colluding adversaries.

摘要: 模拟拉格朗日编码计算(ALCC)是最近提出的一种编码计算范例，其中模拟数据集上的某些计算可以通过浮点实现使用分布式工作节点来高效地执行。虽然众所周知，ALCC可以保护工作人员的数据隐私，但它对返回错误计算结果的敌意工作人员没有弹性。针对这一安全漏洞，我们专注于保护ALCC免受广泛的非串通和串通敌方工作人员的攻击。作为基础步骤，我们利用离散傅里叶变换(DFT)码的纠错算法来构建新的算法来抵消从对手返回的错误计算。此外，当这种健壮的ALCC在实际环境中实现时，我们证明了攻击者可以利用系统中存在的精度误差来提出新的合谋攻击来降低计算精度。作为主要的结论，我们证明了一个与直觉相反的结果，即并不是所有的对手都应该在他们的计算中注入噪声，以便最佳地降低ALCC框架的准确性。这是解决ALCC针对串通对手的脆弱性的第一项同类工作。



## **37. Universal Coding for Shannon Ciphers under Side-Channel Attacks**

侧通道攻击下香农密码的通用编码 cs.IT

6 pages, 3 figures. arXiv admin note: substantial text overlap with  arXiv:1801.02563, arXiv:2201.11670, arXiv:1901.05940

**SubmitDate**: 2024-05-13    [abs](http://arxiv.org/abs/2302.01314v3) [paper-pdf](http://arxiv.org/pdf/2302.01314v3)

**Authors**: Yasutada Oohama, Bagus Santoso

**Abstract**: We study the universal coding under side-channel attacks posed and investigated by Oohama and Santoso (2022). They proposed a theoretical security model for Shannon cipher system under side-channel attacks, where the adversary is not only allowed to collect ciphertexts by eavesdropping the public communication channel, but is also allowed to collect the physical information leaked by the devices where the cipher system is implemented on such as running time, power consumption, electromagnetic radiation, etc. For any distributions of the plain text, any noisy channels through which the adversary observe the corrupted version of the key, and any measurement device used for collecting the physical information, we can derive an achievable rate region for reliability and security such that if we compress the ciphertext with rate within the achievable rate region, then: (1) anyone with secret key will be able to decrypt and decode the ciphertext correctly, but (2) any adversary who obtains the ciphertext and also the side physical information will not be able to obtain any information about the hidden source as long as the leaked physical information is encoded with a rate within the rate region.

摘要: 我们研究了Oohama和Santoso(2022)提出和研究的边信道攻击下的通用编码。他们提出了一种侧信道攻击下Shannon密码系统的理论安全模型，其中不仅允许攻击者通过窃听公共通信信道来收集密文，还允许攻击者收集实现密码系统的设备泄露的物理信息，如运行时间、功耗、电磁辐射等。对于明文的任何分布、攻击者观察密钥被破坏的任何噪声信道以及用于收集物理信息的任何测量设备，我们可以推导出可靠性和安全性的可达速率区域，使得如果将密文的速率压缩在可达速率区域内，那么：(1)任何拥有秘密密钥的人都将能够正确地解密和解码密文，但是(2)只要以速率区域内的速率对泄漏的物理信息进行编码，任何获得密文以及侧物理信息的对手都将不能获得关于隐藏源的任何信息。



## **38. The Janus Interface: How Fine-Tuning in Large Language Models Amplifies the Privacy Risks**

Janus界面：大型语言模型中的微调如何放大隐私风险 cs.CR

**SubmitDate**: 2024-05-12    [abs](http://arxiv.org/abs/2310.15469v2) [paper-pdf](http://arxiv.org/pdf/2310.15469v2)

**Authors**: Xiaoyi Chen, Siyuan Tang, Rui Zhu, Shijun Yan, Lei Jin, Zihao Wang, Liya Su, Zhikun Zhang, XiaoFeng Wang, Haixu Tang

**Abstract**: The rapid advancements of large language models (LLMs) have raised public concerns about the privacy leakage of personally identifiable information (PII) within their extensive training datasets. Recent studies have demonstrated that an adversary could extract highly sensitive privacy data from the training data of LLMs with carefully designed prompts. However, these attacks suffer from the model's tendency to hallucinate and catastrophic forgetting (CF) in the pre-training stage, rendering the veracity of divulged PIIs negligible. In our research, we propose a novel attack, Janus, which exploits the fine-tuning interface to recover forgotten PIIs from the pre-training data in LLMs. We formalize the privacy leakage problem in LLMs and explain why forgotten PIIs can be recovered through empirical analysis on open-source language models. Based upon these insights, we evaluate the performance of Janus on both open-source language models and two latest LLMs, i.e., GPT-3.5-Turbo and LLaMA-2-7b. Our experiment results show that Janus amplifies the privacy risks by over 10 times in comparison with the baseline and significantly outperforms the state-of-the-art privacy extraction attacks including prefix attacks and in-context learning (ICL). Furthermore, our analysis validates that existing fine-tuning APIs provided by OpenAI and Azure AI Studio are susceptible to our Janus attack, allowing an adversary to conduct such an attack at a low cost.

摘要: 大型语言模型(LLM)的快速发展引起了公众对其广泛训练数据集中个人身份信息(PII)隐私泄露的担忧。最近的研究表明，攻击者可以通过精心设计的提示从LLMS的训练数据中提取高度敏感的隐私数据。然而，这些攻击受到模型在预训练阶段的幻觉和灾难性遗忘(CF)的倾向的影响，使得泄露的PII的真实性可以忽略不计。在我们的研究中，我们提出了一种新的攻击，Janus，它利用微调接口从LLMS的训练前数据中恢复被遗忘的PII。我们形式化地描述了LLMS中的隐私泄露问题，并通过对开源语言模型的实证分析解释了为什么被遗忘的PII可以恢复。基于这些见解，我们评估了Janus在开源语言模型和两个最新的LLMS上的性能，即GPT-3.5-Turbo和Llama-2-7b。我们的实验结果表明，Janus将隐私风险放大了10倍以上，并且显著优于目前最先进的隐私提取攻击，包括前缀攻击和上下文中学习(ICL)。此外，我们的分析验证了OpenAI和Azure AI Studio提供的现有微调API容易受到我们的Janus攻击，允许对手以低成本进行此类攻击。



## **39. Synthesizing Iris Images using Generative Adversarial Networks: Survey and Comparative Analysis**

使用生成对抗网络合成虹膜图像：调查和比较分析 cs.CV

**SubmitDate**: 2024-05-11    [abs](http://arxiv.org/abs/2404.17105v2) [paper-pdf](http://arxiv.org/pdf/2404.17105v2)

**Authors**: Shivangi Yadav, Arun Ross

**Abstract**: Biometric systems based on iris recognition are currently being used in border control applications and mobile devices. However, research in iris recognition is stymied by various factors such as limited datasets of bonafide irides and presentation attack instruments; restricted intra-class variations; and privacy concerns. Some of these issues can be mitigated by the use of synthetic iris data. In this paper, we present a comprehensive review of state-of-the-art GAN-based synthetic iris image generation techniques, evaluating their strengths and limitations in producing realistic and useful iris images that can be used for both training and testing iris recognition systems and presentation attack detectors. In this regard, we first survey the various methods that have been used for synthetic iris generation and specifically consider generators based on StyleGAN, RaSGAN, CIT-GAN, iWarpGAN, StarGAN, etc. We then analyze the images generated by these models for realism, uniqueness, and biometric utility. This comprehensive analysis highlights the pros and cons of various GANs in the context of developing robust iris matchers and presentation attack detectors.

摘要: 基于虹膜识别的生物识别系统目前正被用于边境管制应用和移动设备。然而，虹膜识别的研究受到各种因素的阻碍，例如真实虹膜和呈现攻击工具的数据集有限；类内变异有限；以及隐私问题。其中一些问题可以通过使用合成虹膜数据来缓解。本文对最新的基于GaN的合成虹膜图像生成技术进行了全面的综述，评价了它们在生成逼真和有用的虹膜图像方面的优势和局限性，这些图像可以用于虹膜识别系统和呈现攻击检测器的训练和测试。在这方面，我们首先综述了用于合成虹膜生成的各种方法，并具体考虑了基于StyleGAN、RaSGAN、CIT-GAN、iWarpGAN、StarGAN等的生成器。然后，我们分析了这些模型生成的图像的真实感、唯一性和生物特征实用价值。这一全面的分析强调了在开发健壮的虹膜匹配器和呈现攻击检测器的背景下各种GAN的优缺点。



## **40. Tree Proof-of-Position Algorithms**

树位置证明算法 cs.DS

**SubmitDate**: 2024-05-10    [abs](http://arxiv.org/abs/2405.06761v1) [paper-pdf](http://arxiv.org/pdf/2405.06761v1)

**Authors**: Aida Manzano Kharman, Pietro Ferraro, Homayoun Hamedmoghadam, Robert Shorten

**Abstract**: We present a novel class of proof-of-position algorithms: Tree-Proof-of-Position (T-PoP). This algorithm is decentralised, collaborative and can be computed in a privacy preserving manner, such that agents do not need to reveal their position publicly. We make no assumptions of honest behaviour in the system, and consider varying ways in which agents may misbehave. Our algorithm is therefore resilient to highly adversarial scenarios. This makes it suitable for a wide class of applications, namely those in which trust in a centralised infrastructure may not be assumed, or high security risk scenarios. Our algorithm has a worst case quadratic runtime, making it suitable for hardware constrained IoT applications. We also provide a mathematical model that summarises T-PoP's performance for varying operating conditions. We then simulate T-PoP's behaviour with a large number of agent-based simulations, which are in complete agreement with our mathematical model, thus demonstrating its validity. T-PoP can achieve high levels of reliability and security by tuning its operating conditions, both in high and low density environments. Finally, we also present a mathematical model to probabilistically detect platooning attacks.

摘要: 提出了一类新的位置证明算法：树位置证明算法(T-POP)。该算法是分散的、协作的，并且可以以保护隐私的方式进行计算，因此代理不需要公开透露他们的位置。我们没有对系统中的诚实行为做出假设，并考虑了代理人可能不当行为的各种方式。因此，我们的算法对高度对抗性的场景具有弹性。这使得它适合于广泛类别的应用程序，即那些可能不信任集中式基础设施的应用程序，或高安全风险场景。该算法具有最坏情况下的二次运行时间，适用于硬件受限的物联网应用。我们还提供了一个数学模型，总结了T-POP在不同工作条件下的性能。然后，我们用大量基于代理的模拟来模拟T-POP的行为，这与我们的数学模型完全一致，从而证明了它的有效性。T-POP可以通过调整其在高密度和低密度环境中的运行条件来实现高水平的可靠性和安全性。最后，我们还给出了一个概率检测排队攻击的数学模型。



## **41. Certified $\ell_2$ Attribution Robustness via Uniformly Smoothed Attributions**

通过均匀平滑的归因认证$\ell_2$归因稳健性 cs.LG

**SubmitDate**: 2024-05-10    [abs](http://arxiv.org/abs/2405.06361v1) [paper-pdf](http://arxiv.org/pdf/2405.06361v1)

**Authors**: Fan Wang, Adams Wai-Kin Kong

**Abstract**: Model attribution is a popular tool to explain the rationales behind model predictions. However, recent work suggests that the attributions are vulnerable to minute perturbations, which can be added to input samples to fool the attributions while maintaining the prediction outputs. Although empirical studies have shown positive performance via adversarial training, an effective certified defense method is eminently needed to understand the robustness of attributions. In this work, we propose to use uniform smoothing technique that augments the vanilla attributions by noises uniformly sampled from a certain space. It is proved that, for all perturbations within the attack region, the cosine similarity between uniformly smoothed attribution of perturbed sample and the unperturbed sample is guaranteed to be lower bounded. We also derive alternative formulations of the certification that is equivalent to the original one and provides the maximum size of perturbation or the minimum smoothing radius such that the attribution can not be perturbed. We evaluate the proposed method on three datasets and show that the proposed method can effectively protect the attributions from attacks, regardless of the architecture of networks, training schemes and the size of the datasets.

摘要: 模型归因是解释模型预测背后的理论基础的流行工具。然而，最近的工作表明，属性容易受到微小扰动的影响，可以将这些微小扰动添加到输入样本中，以在保持预测输出的同时愚弄属性。虽然实证研究表明，通过对抗性训练取得了积极的效果，但需要一种有效的认证防御方法来理解归因的稳健性。在这项工作中，我们提出使用均匀平滑技术，通过从特定空间均匀采样的噪声来增强香草属性。证明了对于攻击区域内的所有扰动，扰动样本的一致光滑属性与未扰动样本的余弦相似保证是下界的。我们还推导出了与原始证明等价的证明的替代公式，并且提供了最大扰动大小或最小光滑半径，使得属性不能被扰动。我们在三个数据集上对该方法进行了评估，结果表明，无论网络结构、训练方案和数据集的大小如何，该方法都能有效地保护属性免受攻击。



## **42. Evaluating Adversarial Robustness in the Spatial Frequency Domain**

空间频域中的对抗鲁棒性评估 cs.CV

14 pages

**SubmitDate**: 2024-05-10    [abs](http://arxiv.org/abs/2405.06345v1) [paper-pdf](http://arxiv.org/pdf/2405.06345v1)

**Authors**: Keng-Hsin Liao, Chin-Yuan Yeh, Hsi-Wen Chen, Ming-Syan Chen

**Abstract**: Convolutional Neural Networks (CNNs) have dominated the majority of computer vision tasks. However, CNNs' vulnerability to adversarial attacks has raised concerns about deploying these models to safety-critical applications. In contrast, the Human Visual System (HVS), which utilizes spatial frequency channels to process visual signals, is immune to adversarial attacks. As such, this paper presents an empirical study exploring the vulnerability of CNN models in the frequency domain. Specifically, we utilize the discrete cosine transform (DCT) to construct the Spatial-Frequency (SF) layer to produce a block-wise frequency spectrum of an input image and formulate Spatial Frequency CNNs (SF-CNNs) by replacing the initial feature extraction layers of widely-used CNN backbones with the SF layer. Through extensive experiments, we observe that SF-CNN models are more robust than their CNN counterparts under both white-box and black-box attacks. To further explain the robustness of SF-CNNs, we compare the SF layer with a trainable convolutional layer with identical kernel sizes using two mixing strategies to show that the lower frequency components contribute the most to the adversarial robustness of SF-CNNs. We believe our observations can guide the future design of robust CNN models.

摘要: 卷积神经网络(CNN)已经主导了计算机视觉的大部分任务。然而，CNN对对手攻击的脆弱性已经引起了人们对将这些模型部署到安全关键应用程序的担忧。相比之下，人类视觉系统(HVS)利用空间频率通道来处理视觉信号，不受对手攻击。因此，本文提出了一项实证研究，探索CNN模型在频域中的脆弱性。具体地说，我们利用离散余弦变换(DCT)来构造空间频率(SF)层来产生输入图像的块状频谱，并通过用SF层替换广泛使用的CNN骨干的初始特征提取层来构造空间频率CNN(SF-CNN)。通过大量的实验，我们观察到SF-CNN模型在白盒和黑盒攻击下都比CNN模型更健壮。为了进一步解释SF-CNN的健壮性，我们使用两种混合策略将SF层与具有相同核大小的可训练卷积层进行了比较，结果表明低频分量对SF-CNN的对抗健壮性贡献最大。我们相信，我们的观察可以指导未来稳健的CNN模型的设计。



## **43. Improving Transferable Targeted Adversarial Attack via Normalized Logit Calibration and Truncated Feature Mixing**

通过规范化Logit校准和截断特征混合改进可转移有针对性的对抗攻击 cs.CV

**SubmitDate**: 2024-05-10    [abs](http://arxiv.org/abs/2405.06340v1) [paper-pdf](http://arxiv.org/pdf/2405.06340v1)

**Authors**: Juanjuan Weng, Zhiming Luo, Shaozi Li

**Abstract**: This paper aims to enhance the transferability of adversarial samples in targeted attacks, where attack success rates remain comparatively low. To achieve this objective, we propose two distinct techniques for improving the targeted transferability from the loss and feature aspects. First, in previous approaches, logit calibrations used in targeted attacks primarily focus on the logit margin between the targeted class and the untargeted classes among samples, neglecting the standard deviation of the logit. In contrast, we introduce a new normalized logit calibration method that jointly considers the logit margin and the standard deviation of logits. This approach effectively calibrates the logits, enhancing the targeted transferability. Second, previous studies have demonstrated that mixing the features of clean samples during optimization can significantly increase transferability. Building upon this, we further investigate a truncated feature mixing method to reduce the impact of the source training model, resulting in additional improvements. The truncated feature is determined by removing the Rank-1 feature associated with the largest singular value decomposed from the high-level convolutional layers of the clean sample. Extensive experiments conducted on the ImageNet-Compatible and CIFAR-10 datasets demonstrate the individual and mutual benefits of our proposed two components, which outperform the state-of-the-art methods by a large margin in black-box targeted attacks.

摘要: 本文旨在提高攻击成功率相对较低的定向攻击中对抗性样本的可转移性。为了实现这一目标，我们从损失和特征两个方面提出了两种不同的技术来提高目标可转移性。首先，在以往的方法中，用于目标攻击的Logit校准主要集中在样本中目标类和非目标类之间的Logit差值，而忽略了Logit的标准差。相反，我们引入了一种新的归一化Logit校准方法，该方法同时考虑了Logit裕度和Logit的标准差。这种方法有效地校准了LOGITS，增强了目标可转移性。其次，以往的研究表明，在优化过程中混合清洁样本的特征可以显著提高可转移性。在此基础上，我们进一步研究了一种截断特征混合方法，以减少源训练模型的影响，从而得到进一步的改进。通过去除与从清洁样本的高级卷积层分解的最大奇异值相关联的Rank-1特征来确定截断特征。在ImageNet兼容和CIFAR-10数据集上进行的广泛实验表明，我们提出的两个组件具有单独和共同的好处，在黑盒定向攻击中远远超过最先进的方法。



## **44. PUMA: margin-based data pruning**

SEARCH A：基于利润的数据修剪 cs.LG

**SubmitDate**: 2024-05-10    [abs](http://arxiv.org/abs/2405.06298v1) [paper-pdf](http://arxiv.org/pdf/2405.06298v1)

**Authors**: Javier Maroto, Pascal Frossard

**Abstract**: Deep learning has been able to outperform humans in terms of classification accuracy in many tasks. However, to achieve robustness to adversarial perturbations, the best methodologies require to perform adversarial training on a much larger training set that has been typically augmented using generative models (e.g., diffusion models). Our main objective in this work, is to reduce these data requirements while achieving the same or better accuracy-robustness trade-offs. We focus on data pruning, where some training samples are removed based on the distance to the model classification boundary (i.e., margin). We find that the existing approaches that prune samples with low margin fails to increase robustness when we add a lot of synthetic data, and explain this situation with a perceptron learning task. Moreover, we find that pruning high margin samples for better accuracy increases the harmful impact of mislabeled perturbed data in adversarial training, hurting both robustness and accuracy. We thus propose PUMA, a new data pruning strategy that computes the margin using DeepFool, and prunes the training samples of highest margin without hurting performance by jointly adjusting the training attack norm on the samples of lowest margin. We show that PUMA can be used on top of the current state-of-the-art methodology in robustness, and it is able to significantly improve the model performance unlike the existing data pruning strategies. Not only PUMA achieves similar robustness with less data, but it also significantly increases the model accuracy, improving the performance trade-off.

摘要: 在许多任务中，深度学习在分类准确率方面已经能够超过人类。然而，为了实现对对抗性扰动的稳健性，最好的方法需要在通常使用生成模型(例如，扩散模型)扩充的大得多的训练集上执行对抗性训练。我们在这项工作中的主要目标是减少这些数据要求，同时实现相同或更好的精度-稳健性权衡。我们的重点是数据剪枝，即根据到模型分类边界的距离(即边界)来删除一些训练样本。我们发现，当我们添加大量的合成数据时，现有的对低边际样本进行剪枝的方法不能提高鲁棒性，并用感知器学习任务来解释这种情况。此外，我们发现，为了更好的准确性而修剪高边缘样本会增加错误标记的扰动数据在对抗性训练中的有害影响，损害稳健性和准确性。因此，我们提出了一种新的数据剪枝策略PUMA，它使用DeepFool计算差值，并在差值最小的样本上联合调整训练攻击范数，在不影响性能的情况下修剪差值最高的训练样本。我们表明，PUMA可以在当前最先进的方法的健壮性上使用，并且它能够显著提高模型的性能，而不是现有的数据剪枝策略。PUMA不仅用更少的数据实现了类似的稳健性，而且还显著提高了模型的精度，改善了性能权衡。



## **45. Exploring the Interplay of Interpretability and Robustness in Deep Neural Networks: A Saliency-guided Approach**

探索深度神经网络中可解释性和鲁棒性的相互作用：显着性引导的方法 cs.CV

**SubmitDate**: 2024-05-10    [abs](http://arxiv.org/abs/2405.06278v1) [paper-pdf](http://arxiv.org/pdf/2405.06278v1)

**Authors**: Amira Guesmi, Nishant Suresh Aswani, Muhammad Shafique

**Abstract**: Adversarial attacks pose a significant challenge to deploying deep learning models in safety-critical applications. Maintaining model robustness while ensuring interpretability is vital for fostering trust and comprehension in these models. This study investigates the impact of Saliency-guided Training (SGT) on model robustness, a technique aimed at improving the clarity of saliency maps to deepen understanding of the model's decision-making process. Experiments were conducted on standard benchmark datasets using various deep learning architectures trained with and without SGT. Findings demonstrate that SGT enhances both model robustness and interpretability. Additionally, we propose a novel approach combining SGT with standard adversarial training to achieve even greater robustness while preserving saliency map quality. Our strategy is grounded in the assumption that preserving salient features crucial for correctly classifying adversarial examples enhances model robustness, while masking non-relevant features improves interpretability. Our technique yields significant gains, achieving a 35\% and 20\% improvement in robustness against PGD attack with noise magnitudes of $0.2$ and $0.02$ for the MNIST and CIFAR-10 datasets, respectively, while producing high-quality saliency maps.

摘要: 对抗性攻击对在安全关键型应用中部署深度学习模型提出了重大挑战。在确保可解释性的同时保持模型的健壮性对于培养对这些模型的信任和理解至关重要。本研究调查显著引导训练(SGT)对模型稳健性的影响，这是一种旨在提高显著图的清晰度以加深对模型决策过程的理解的技术。实验在标准基准数据集上进行，使用各种深度学习体系结构，在有和没有SGT的情况下进行训练。研究结果表明，SGT既增强了模型的稳健性，又增强了模型的可解释性。此外，我们提出了一种结合SGT和标准对抗性训练的新方法，在保持显著图质量的同时获得更好的稳健性。我们的策略基于这样的假设，即保留对于正确分类对抗性示例至关重要的显著特征可以增强模型的稳健性，而屏蔽不相关的特征可以提高可解释性。我们的技术产生了显著的收益，在MNIST和CIFAR-10数据集的噪声幅度分别为0.2美元和0.02美元的情况下，对PGD攻击的稳健性分别提高了35%和20%，同时生成了高质量的显著图。



## **46. Disttack: Graph Adversarial Attacks Toward Distributed GNN Training**

区别：针对分布式GNN培训的图形对抗攻击 cs.LG

Accepted by 30th International European Conference on Parallel and  Distributed Computing(Euro-Par 2024)

**SubmitDate**: 2024-05-10    [abs](http://arxiv.org/abs/2405.06247v1) [paper-pdf](http://arxiv.org/pdf/2405.06247v1)

**Authors**: Yuxiang Zhang, Xin Liu, Meng Wu, Wei Yan, Mingyu Yan, Xiaochun Ye, Dongrui Fan

**Abstract**: Graph Neural Networks (GNNs) have emerged as potent models for graph learning. Distributing the training process across multiple computing nodes is the most promising solution to address the challenges of ever-growing real-world graphs. However, current adversarial attack methods on GNNs neglect the characteristics and applications of the distributed scenario, leading to suboptimal performance and inefficiency in attacking distributed GNN training.   In this study, we introduce Disttack, the first framework of adversarial attacks for distributed GNN training that leverages the characteristics of frequent gradient updates in a distributed system. Specifically, Disttack corrupts distributed GNN training by injecting adversarial attacks into one single computing node. The attacked subgraphs are precisely perturbed to induce an abnormal gradient ascent in backpropagation, disrupting gradient synchronization between computing nodes and thus leading to a significant performance decline of the trained GNN. We evaluate Disttack on four large real-world graphs by attacking five widely adopted GNNs. Compared with the state-of-the-art attack method, experimental results demonstrate that Disttack amplifies the model accuracy degradation by 2.75$\times$ and achieves speedup by 17.33$\times$ on average while maintaining unnoticeability.

摘要: 图神经网络(GNN)已经成为图学习的有力模型。将训练过程分布在多个计算节点上是解决不断增长的真实世界图的挑战的最有前途的解决方案。然而，目前针对GNN的对抗性攻击方法忽略了分布式场景的特点和应用，导致在攻击分布式GNN训练时性能不佳且效率低下。在这项研究中，我们介绍了Disttack，这是第一个用于分布式GNN训练的对抗性攻击框架，它利用了分布式系统中频繁梯度更新的特点。具体地说，Disttack通过将敌意攻击注入到单个计算节点来破坏分布式GNN训练。被攻击的子图被精确地扰动，导致反向传播中的异常梯度上升，扰乱了计算节点之间的梯度同步，从而导致训练后的GNN的性能显著下降。我们通过攻击五个广泛使用的GNN来评估四个大型真实世界图上的Disttack。实验结果表明，与最新的攻击方法相比，Disttack在保持不可察觉的情况下，使模型的准确率降低了2.75倍，平均加速比提高了17.33倍。



## **47. Concealing Backdoor Model Updates in Federated Learning by Trigger-Optimized Data Poisoning**

通过触发优化的数据中毒隐藏联邦学习中后门模型更新 cs.CR

**SubmitDate**: 2024-05-10    [abs](http://arxiv.org/abs/2405.06206v1) [paper-pdf](http://arxiv.org/pdf/2405.06206v1)

**Authors**: Yujie Zhang, Neil Gong, Michael K. Reiter

**Abstract**: Federated Learning (FL) is a decentralized machine learning method that enables participants to collaboratively train a model without sharing their private data. Despite its privacy and scalability benefits, FL is susceptible to backdoor attacks, where adversaries poison the local training data of a subset of clients using a backdoor trigger, aiming to make the aggregated model produce malicious results when the same backdoor condition is met by an inference-time input. Existing backdoor attacks in FL suffer from common deficiencies: fixed trigger patterns and reliance on the assistance of model poisoning. State-of-the-art defenses based on Byzantine-robust aggregation exhibit a good defense performance on these attacks because of the significant divergence between malicious and benign model updates. To effectively conceal malicious model updates among benign ones, we propose DPOT, a backdoor attack strategy in FL that dynamically constructs backdoor objectives by optimizing a backdoor trigger, making backdoor data have minimal effect on model updates. We provide theoretical justifications for DPOT's attacking principle and display experimental results showing that DPOT, via only a data-poisoning attack, effectively undermines state-of-the-art defenses and outperforms existing backdoor attack techniques on various datasets.

摘要: 联合学习(FL)是一种去中心化的机器学习方法，允许参与者在不共享私人数据的情况下协作训练模型。尽管FL具有隐私和可扩展性方面的优势，但它很容易受到后门攻击，即攻击者使用后门触发器毒化部分客户端的本地训练数据，目的是在推理时输入满足相同的后门条件时，使聚合模型产生恶意结果。FL中现有的后门攻击存在共同的缺陷：固定的触发模式和依赖模型中毒的辅助。由于恶意模型更新和良性模型更新之间的显著差异，基于拜占庭稳健聚合的最新防御技术在这些攻击中表现出良好的防御性能。为了有效地隐藏良性模型更新中的恶意模型更新，我们提出了一种FL中的后门攻击策略DPOT，它通过优化后门触发器来动态构建后门目标，使后门数据对模型更新的影响最小。我们为DPOT的攻击原理提供了理论依据，并展示了实验结果表明，DPOT仅通过一次数据中毒攻击就可以有效地破坏最先进的防御措施，并在各种数据集上优于现有的后门攻击技术。



## **48. Muting Whisper: A Universal Acoustic Adversarial Attack on Speech Foundation Models**

静音低语：对语音基础模型的通用声学对抗攻击 cs.CL

**SubmitDate**: 2024-05-09    [abs](http://arxiv.org/abs/2405.06134v1) [paper-pdf](http://arxiv.org/pdf/2405.06134v1)

**Authors**: Vyas Raina, Rao Ma, Charles McGhee, Kate Knill, Mark Gales

**Abstract**: Recent developments in large speech foundation models like Whisper have led to their widespread use in many automatic speech recognition (ASR) applications. These systems incorporate `special tokens' in their vocabulary, such as $\texttt{<endoftext>}$, to guide their language generation process. However, we demonstrate that these tokens can be exploited by adversarial attacks to manipulate the model's behavior. We propose a simple yet effective method to learn a universal acoustic realization of Whisper's $\texttt{<endoftext>}$ token, which, when prepended to any speech signal, encourages the model to ignore the speech and only transcribe the special token, effectively `muting' the model. Our experiments demonstrate that the same, universal 0.64-second adversarial audio segment can successfully mute a target Whisper ASR model for over 97\% of speech samples. Moreover, we find that this universal adversarial audio segment often transfers to new datasets and tasks. Overall this work demonstrates the vulnerability of Whisper models to `muting' adversarial attacks, where such attacks can pose both risks and potential benefits in real-world settings: for example the attack can be used to bypass speech moderation systems, or conversely the attack can also be used to protect private speech data.

摘要: 像Whisper这样的大型语音基础模型的最新发展导致它们在许多自动语音识别(ASR)应用中被广泛使用。这些系统在它们的词汇表中加入了“特殊记号”，如$\exttt{<endoftext>}$，以指导它们的语言生成过程。然而，我们证明了这些令牌可以被敌意攻击利用来操纵模型的行为。我们提出了一种简单而有效的方法来学习Whisper的$\exttt{<endoftext>}$标记的通用声学实现，当预先添加到任何语音信号时，鼓励模型忽略语音，只转录特殊的标记，从而有效地抑制了模型。我们的实验表明，相同的、通用的0.64秒的对抗性音频片段可以成功地使目标Whisper ASR模型在97%以上的语音样本上静音。此外，我们发现这种普遍的对抗性音频片段经常转移到新的数据集和任务。总体而言，这项工作证明了Whisper模型对“静音”对手攻击的脆弱性，在现实世界中，这种攻击既可以带来风险，也可以带来潜在的好处：例如，攻击可以用来绕过语音调节系统，或者反过来，攻击也可以用来保护私人语音数据。



## **49. Hard Work Does Not Always Pay Off: Poisoning Attacks on Neural Architecture Search**

努力工作并不总是有回报：对神经架构搜索的毒害攻击 cs.LG

**SubmitDate**: 2024-05-09    [abs](http://arxiv.org/abs/2405.06073v1) [paper-pdf](http://arxiv.org/pdf/2405.06073v1)

**Authors**: Zachary Coalson, Huazheng Wang, Qingyun Wu, Sanghyun Hong

**Abstract**: In this paper, we study the robustness of "data-centric" approaches to finding neural network architectures (known as neural architecture search) to data distribution shifts. To audit this robustness, we present a data poisoning attack, when injected to the training data used for architecture search that can prevent the victim algorithm from finding an architecture with optimal accuracy. We first define the attack objective for crafting poisoning samples that can induce the victim to generate sub-optimal architectures. To this end, we weaponize existing search algorithms to generate adversarial architectures that serve as our objectives. We also present techniques that the attacker can use to significantly reduce the computational costs of crafting poisoning samples. In an extensive evaluation of our poisoning attack on a representative architecture search algorithm, we show its surprising robustness. Because our attack employs clean-label poisoning, we also evaluate its robustness against label noise. We find that random label-flipping is more effective in generating sub-optimal architectures than our clean-label attack. Our results suggests that care must be taken for the data this emerging approach uses, and future work is needed to develop robust algorithms.

摘要: 在本文中，我们研究了“以数据为中心”的方法寻找神经网络结构(称为神经结构搜索)对数据分布变化的稳健性。为了检验这种健壮性，我们提出了一种数据中毒攻击，当注入用于体系结构搜索的训练数据时，可以阻止受害者算法以最佳精度找到体系结构。我们首先定义了制作中毒样本的攻击目标，这些样本可以诱导受害者生成次优的体系结构。为此，我们将现有的搜索算法武器化，以生成作为我们目标的对抗性架构。我们还提供了攻击者可以用来显著降低制作中毒样本的计算成本的技术。在对我们对一个典型架构搜索算法的毒化攻击的广泛评估中，我们展示了其惊人的健壮性。因为我们的攻击使用了干净标签中毒，所以我们还评估了它对标签噪声的稳健性。我们发现随机标签翻转在生成次优体系结构方面比我们的干净标签攻击更有效。我们的结果表明，必须注意这种新兴方法使用的数据，并且需要进一步的工作来开发健壮的算法。



## **50. BB-Patch: BlackBox Adversarial Patch-Attack using Zeroth-Order Optimization**

BB-patch：使用零阶优化的黑匣子对抗补丁攻击 cs.CV

**SubmitDate**: 2024-05-09    [abs](http://arxiv.org/abs/2405.06049v1) [paper-pdf](http://arxiv.org/pdf/2405.06049v1)

**Authors**: Satyadwyoom Kumar, Saurabh Gupta, Arun Balaji Buduru

**Abstract**: Deep Learning has become popular due to its vast applications in almost all domains. However, models trained using deep learning are prone to failure for adversarial samples and carry a considerable risk in sensitive applications. Most of these adversarial attack strategies assume that the adversary has access to the training data, the model parameters, and the input during deployment, hence, focus on perturbing the pixel level information present in the input image.   Adversarial Patches were introduced to the community which helped in bringing out the vulnerability of deep learning models in a much more pragmatic manner but here the attacker has a white-box access to the model parameters. Recently, there has been an attempt to develop these adversarial attacks using black-box techniques. However, certain assumptions such as availability large training data is not valid for a real-life scenarios. In a real-life scenario, the attacker can only assume the type of model architecture used from a select list of state-of-the-art architectures while having access to only a subset of input dataset. Hence, we propose an black-box adversarial attack strategy that produces adversarial patches which can be applied anywhere in the input image to perform an adversarial attack.

摘要: 深度学习由于其在几乎所有领域的广泛应用而变得流行起来。然而，使用深度学习训练的模型对于对抗性样本容易失败，并且在敏感应用中具有相当大的风险。这些对抗性攻击策略大多假设对手在部署过程中可以访问训练数据、模型参数和输入，因此，专注于干扰输入图像中存在的像素级信息。社区中引入了对抗性补丁，这有助于以更实用的方式暴露深度学习模型的漏洞，但在这里，攻击者可以通过白盒访问模型参数。最近，有人试图使用黑盒技术来开发这些对抗性攻击。然而，某些假设，如大量训练数据的可用性，对于现实生活场景是不成立的。在现实生活场景中，攻击者只能假定从最先进的体系结构的精选列表中使用的模型体系结构的类型，同时只能访问输入数据集的子集。因此，我们提出了一种黑盒对抗性攻击策略，该策略产生对抗性补丁，可以应用于输入图像中的任何位置来执行对抗性攻击。



