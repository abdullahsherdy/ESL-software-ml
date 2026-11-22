# ESL-software-ml

### Real-time Egyptian Sign Language recognition that runs on your own device

![Hero screenshot](docs/screenshots/hero.png)

> [  add a hero screenshot or a short GIF of the app recognizing a sign ]

---

## At a glance

| | |
|---|---|
| **Product** | ESL-software-ml: Egyptian Sign Language recognition |
| **What it does** | Reads signs from a camera or video and shows the sign name, a confidence score and the signer's emotion |
| **Works on** | Desktop (Windows, Linux, macOS) and phone or tablet browsers |
| **Internet needed** | No. Everything runs on your own computer |
| **Supported signs** | 56 |
| **Status** | Graduation project, working prototype |
| **Version** | 1 |
| **Team** |  |

---

## The problem

Many Deaf and hard-of-hearing people in Egypt depend on sign language interpreters to communicate with hearing people, for example at a clinic, a school or a government office. Interpreters are not always available, and written text is not a natural replacement for sign language.

## The solution

ESL-software-ml watches a person sign through an ordinary camera and tells the other person, on screen and in real time, which sign was made. It needs no special gloves, no sensors and no internet connection. It also reads the signer's facial emotion, because the same sign can carry a different feeling.

It is meant to support communication, not replace human interpreters.

---

## Who it is for

- **Deaf and hard-of-hearing people** who want to be understood in everyday situations
- **Families and friends** who are learning to communicate with a Deaf relative
- **Students and teachers** of Egyptian Sign Language, who can use it to practice and check their signs
- **Institutions** such as hospitals, schools and service desks that want a simple communication aid at the counter

---

## Key features

| Feature | What it means for you |
|---|---|
| **Real-time recognition** | Results appear while the person is still signing, not after a delay |
| **Works offline** | No internet connection is needed, and your video is not sent to any cloud service |
| **Emotion awareness** | Shows the signer's emotion next to the sign (desktop full mode) |
| **Confidence score** | Every result shows how sure the system is, so you know when to double-check |
| **Stable results** | The system waits for several consistent readings before showing a sign, so the text does not flicker |
| **Ignores idle hands** | It only predicts when the hands are moving, which avoids false results |
| **Phone-friendly web version** | Open it in a phone browser on the same Wi-Fi network |
| **No special equipment** | A normal webcam or phone camera is enough |

---

## How to use it

1. **Start the app** on the computer. Choose the desktop app or open the web version in a browser.
2. **Allow camera access** when asked.
3. **Stand or sit facing the camera** so that both hands and your face are visible.
4. **Sign naturally.** The screen shows the sign name, the confidence and the detected emotion.

### Tips for the best results

- Use good lighting, with the light in front of you rather than behind you
- Keep both hands fully inside the camera frame
- Sit about an arm's length or more from the camera, so your hands and face are visible together
- Use a plain background if possible
- Pause briefly between signs

The setup commands are in the [README](README(2).md).

---

## Supported signs

The model recognizes the 56 signs below. Labels are written in Latin letters exactly as the model uses them.

Signs marked with * have a meaning that still needs to be confirmed.

| # | Sign label | Arabic | English |
|---|---|---|---|
| 1 | 3amel eh | كيف حالك؟ | How are you? |
| 2 | 3ieb | عيب | Shame / that's not appropriate |
| 3 | 3rabia | عربية | Car |
| 4 | 3waz | عاوز (أريد) | I want |
| 5 | akhoia | أخويا | My brother |
| 6 | alam | ألم | Pain |
| 7 | bhab | بحب | I love |
| 8 | bkrh | بكرهه | I hate him/it |
| 9 | bsor3a | بسرعة | Quickly |
| 10 | delwaty | دلوقتي (الآن) | Now |
| 11 | docktor | دكتور | Doctor |
| 12 | efrdha | افردها | Spread it out / unfold it |
| 13 | ehtrem nfsk | احترم نفسك | Respect yourself |
| 14 | elnhrda | النهاردة | Today |
| 15 | ermy wra dhrk | ارمي ورا ظهرك | Throw it behind your back (let it go) |
| 16 | fatan | فتان | Troublemaker / tempter |
| 17 | fe sark | في سرك | Keep it to yourself / in secret |
| 18 | flos | فلوس | Money |
| 19 | fokk many | فك مني | Leave me alone * |
| 20 | gomma | جمعة | Friday * |
| 21 | harmy | حرامي | Thief |
| 22 | hba hba | حبة حبة | Little by little |
| 23 | helw awy | حلو أوي | Very nice |
| 24 | hta hta | حتة حتة | Piece by piece * |
| 25 | kdab | كداب | Liar |
| 26 | khaif | خايف | Afraid |
| 27 | khaly balk | خلي بالك | Be careful / watch out |
| 28 | khatbty | خطيبتي | My fiancee |
| 29 | kolya | كلية | College / faculty * |
| 30 | ktab | كتاب | Book |
| 31 | mafish | مافيش | There isn't / none |
| 32 | makar | مكار | Cunning / deceitful |
| 33 | manzel | منزل | House / home |
| 34 | mash mashy | مش ماشي | Not okay / not working * |
| 35 | mashy | ماشي | Okay |
| 36 | meraty | مراتي | My wife |
| 37 | mohands | مهندس | Engineer |
| 38 | moshwash | مشوش | Confused / disturbed * |
| 39 | msh 3waz | مش عاوز (لا أريد) | I don't want |
| 40 | oda | أوضة | Room |
| 41 | sabhl elkheer | صباح الخير | Good morning |
| 42 | sabhlala | سبحان الله | Glory be to God * |
| 43 | sadak | صدق | He told the truth / honest * |
| 44 | sahby | صاحبي | My friend |
| 45 | sbak | سباك | Plumber * |
| 46 | shar3 | شارع | Street |
| 47 | shbak | شباك | Window |
| 48 | t3ban | تعبان | Tired |
| 49 | tmam | تمام | Fine / perfect |
| 50 | whashny | وحشني | I miss you |
| 51 | whda whda | واحدة واحدة | One by one / slowly |
| 52 | y7takr | يحتكر | Monopolizes / hoards * |
| 53 | yabky | يبكي | He cries |
| 54 | yadhk | يضحك | He laughs |
| 55 | yallhwy | يا لهوي | Oh my! |
| 56 | yshtm | يشتم | He insults |

---

## Performance

Measured on the project's validation data with the default model (`model_v2`):

| Metric | Result |
|---|---|
| Validation accuracy | 91.7% |
| Macro F1 score | 0.906 |
| Improvement over the baseline model | from 81.8% to 91.7% accuracy, thanks to training with data augmentation |

| Other measurements | Result |
|---|---|
| Live display speed (FPS) on desktop |  |
| Round-trip time in the web version |  |
| Tested with a signer who was not in the training videos |  |

> Note: The validation set is small, about two to three samples per sign, so treat these numbers as an early indication rather than a final guarantee. Details are in `results/`.

---

## Technical specifications

| | |
|---|---|
| **Input** | Webcam, phone camera or video file |
| **Recognition method** | Hand and face landmarks are extracted from each frame, then classified by a neural network |
| **Default model** | Augmented MLP (`model_v2`) |
| **Other models included** | Baseline MLP (`model_v1`), Bi-LSTM (`model_v3`) |
| **Emotion detection** | DeepFace, desktop full mode only. The web version and sprint mode use a fixed neutral emotion |
| **Result format** | Sign label, confidence from 0.0 to 1.0, emotion |
| **Requirements** | Python 3.10 or 3.11, a camera, and about 500 MB of disk space for the emotion model on first run |
| **Integration** | One function, `predict_frame()`, for developers who want to embed it in another app |
| **License** | MIT |

---

## Privacy

- Recognition runs on your own computer. No video is sent to any cloud service.
- In the web version, the phone sends camera frames to the computer running the server on your own network. Nothing is uploaded elsewhere.
- Video frames are processed live and are not stored by the application. [ fill in: confirm this matches the final code ]

---

## Limitations

We want to be clear about what the product can and cannot do today:

- It recognizes **56 individual signs**. It does not translate full sentences or fingerspelling.
- Accuracy can drop for signers whose style differs from the people in the training videos.
- Hands must be visible to the camera. If they leave the frame, no sign is detected.
- Poor lighting, a busy background or fast movement can reduce accuracy.
- Emotion detection works only in the desktop full mode, and the first run downloads about 500 MB.
- The signs in this version were learned from a limited dataset. [ fill in: number of signers and videos ]

---

## Frequently asked questions

**Does it need internet?**
No. After installation, everything runs on your own computer.

**Is my video stored or uploaded?**
No video is sent to any cloud service. See the Privacy section.

**Can I use it on my phone?**
Yes, through the web version in your phone's browser, with the computer and phone on the same Wi-Fi network.

**Does it replace a sign language interpreter?**
No. It is a communication aid for simple, everyday signs. Interpreters are still essential for complex or sensitive conversations such as medical or legal ones.

**Can it learn new signs?**
Yes, but it needs new training data and retraining. [ fill in: describe the process if you plan to support it ]

**Can other apps use it?**
Yes. Developers can call `predict_frame()` and receive the sign, confidence and emotion.

---

## User feedback



---

## Roadmap



---

## Team and contact

| Name | Role |
|---|---|
| Abdullah Sherdy |  |
| Sara Yasser |  |

- **Supervisor:**
- **University / Faculty:**
- **Contact:**
- **Source code:** https://github.com/abdullahsherdy/ESL-software-ml
