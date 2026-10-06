"""
Builds a small, original seed dataset at datasets/combined_sample.csv.

IMPORTANT HONESTY NOTE (see README "Limitations"):
This is a hand-written *bootstrap* dataset of ~95 short texts, not a
research-grade corpus. It exists so the classifier has something real
to train on out of the box (zero cost, no downloads). Accuracy numbers
produced against it should be read as "does the pipeline work and learn
something sensible" rather than "state of the art AI detection".

For a serious model, replace/extend this with a much larger dataset,
e.g. a few thousand rows of real human writing (essays, emails, forum
posts you have rights to use) paired with generations from several
different AI models and prompting styles. Keep the same text,label
CSV schema.
"""
import csv
from pathlib import Path

OUT_PATH = Path(__file__).resolve().parent.parent / "datasets" / "combined_sample.csv"

# Human-written examples: intentionally messy, varied sentence length,
# personal, sometimes ungrammatical, genre-diverse.
HUMAN_TEXTS = [
    "so i tried fixing the leaky faucet myself last night and honestly it was a disaster. water everywhere, my roommate laughing at me from the doorway, and I still had to call a plumber this morning. lesson learned I guess.",
    "Dear team, quick update before the weekend — the staging server is back up, but please don't push anything new until Monday. I want to keep an eye on memory usage overnight. Thanks for your patience this week, it's been a lot.",
    "My grandmother used to say that a watched pot never boils, and honestly she was right about most things. I think about that a lot when I'm waiting on test results or job applications. Waiting is just part of life, isn't it?",
    "Okay hot take: pineapple on pizza is fine. Not great, not terrible, just fine. People need to calm down about it. There are worse crimes against food, like putting ketchup on a hot dog in Chicago.",
    "The hike took about four hours longer than we planned because Dave insisted the trail 'obviously' went left at the fork. It did not. We ended up bushwhacking through some genuinely unpleasant brambles before finding the actual path again.",
    "I've been putting off writing this cover letter for three days now. Every time I open the document I just stare at the blinking cursor and then go make tea. I think I'm afraid of sounding fake, honestly.",
    "Quarterly numbers are in and they're rough, not going to sugarcoat it. Revenue is down 8% from Q2, mostly because of the Denver contract falling through. We need to talk about the Q4 pipeline on Thursday.",
    "my cat knocked my coffee off the desk at 6am this morning and somehow looked proud of it. I have never been more tired and more annoyed at a small animal in my life. anyway here's the bug report you asked for.",
    "Honestly I don't think the movie was that bad? Like yeah the pacing dragged in the middle and the villain's motivation made zero sense, but the cinematography was gorgeous and I'd still recommend it for a rainy Sunday.",
    "We buried my dog Max on Tuesday under the oak tree in the backyard, the one he used to dig under every summer. Fifteen years is a long time to know someone, even if that someone had four legs and ate socks.",
    "Can you resend the invoice? Ours got flagged as spam apparently, which is annoying because now I have to chase down accounting again. Also — are we still on for the call Thursday at 2?",
    "The recipe called for two cups of flour but I only had one and a half so I just went for it, added some extra butter to compensate. Turned out fine actually, a little denser than usual but nobody complained.",
    "There's this weird sound my car makes only when I turn left at low speed, like a squeak but with attitude. The mechanic couldn't replicate it of course. Typical.",
    "I keep rereading your email trying to figure out if you're mad at me or if that's just how you write emails. Can we just hop on a call instead? Text is so hard to read tone in.",
    "Freshman year I failed calculus twice before it finally clicked, and honestly I think that struggle taught me more about persistence than any of the classes I aced. Grades don't always tell the whole story.",
    "The bus was twenty minutes late again, so I missed the first ten minutes of the meeting and had to awkwardly slide into a seat while everyone stared. Not my favorite way to start a Monday.",
    "Sent the draft over, let me know what you think. I wasn't totally sure about the ending so feel free to rip it apart, I can take it.",
    "Three kids, one bathroom, and a broken dishwasher — that's basically my week in a sentence. Somehow we're still surviving though, mostly on cereal and stubbornness.",
    "I don't really buy the argument that remote work kills creativity. My best ideas this year happened while walking my dog, not sitting in some glass conference room getting talked at.",
    "Turns out the printer just needed to be turned off and on again. Forty-five minutes of troubleshooting for that. I feel a specific kind of dumb right now.",
    "The garden's a mess this year, tomatoes got some kind of blight and the squirrels ate basically every strawberry before I could. Next year I'm building an actual fence, I swear.",
    "Look, I get that budgets are tight, but cutting the QA headcount right before a major release seems like a genuinely bad idea to me. Just my two cents before Friday's meeting.",
    "She showed up an hour late to her own birthday dinner because she got the date wrong, then insisted we'd all misremembered. Classic Sarah, honestly, we love her for it.",
    "Spent the whole afternoon debugging a null pointer that turned out to be a typo in a config file. Four hours. One typo. I need a nap.",
    "I don't think I'll ever fully understand offside rule in soccer and at this point I'm too embarrassed to ask anyone to explain it again.",
]

# Second batch: skewed toward content-marketing / SEO-adjacent genres
# (product descriptions, "best X" roundups, how-to intros, reviews),
# since that's a common real-world use case for this tool. Still
# original, hand-written examples — not scraped or copied.
HUMAN_TEXTS_SEO = [
    "Look, I've tried four different standing desks and this is the only one that hasn't wobbled when I lean on it. The motor's a little loud going up, not gonna lie, but at this price I can live with it.",
    "Quick tip if you're meal prepping chicken thighs: don't skip patting them dry first. I skipped it once, ended up steaming instead of searing, and the whole batch turned out sad and gray.",
    "Honestly the best budget noise-cancelling headphones I've owned, and I've owned a few. The bass is a bit much out of the box but two minutes in the EQ app fixes that.",
    "So you want to start a podcast. Cool — here's the thing nobody tells you: the mic matters less than your closet full of clothes, which is a genuinely great place to record because fabric eats echo.",
    "We switched our whole team to this project tool last spring and I'll admit I was skeptical. Six months in, the thing that actually stuck was the weekly digest email, not the fancy Gantt charts.",
    "I compared five air fryers for this review and honestly most of them are the same guts in different plastic. The one that stood out had a basket that actually goes in the dishwasher, which sounds small until you've hand-washed a greasy basket at 11pm.",
    "If your Wi-Fi keeps dropping in the back bedroom, don't buy a new router yet — try moving the one you have off the floor first. Fixed it for me and cost nothing.",
    "This blender crushed ice fine but choked on frozen mango every single time, which is annoying because that's literally what I bought it for.",
    "I've been running in these for about two months now, roughly 15 miles a week, and the cushioning held up better than I expected for the price.",
    "Here's what actually happened when I tried the 30-day decluttering challenge everyone's been posting about: I made it to day 11, got busy, and never went back. Still, those 11 days got rid of more stuff than the previous year combined.",
    "Picked this cast iron pan up secondhand and it took about three rounds of stripping and reseasoning before it stopped sticking. Worth it, but don't expect it to be nonstick on day one.",
    "The customer support on this was honestly the deciding factor for me — I emailed at midnight with a dumb question and got a real, useful answer back within the hour.",
    "Tried three different budgeting apps this year. This one won because it actually lets me edit a transaction without five confirmation screens.",
    "If you're choosing between these two laptops for video editing, get the one with more RAM, not the one with the slightly faster chip — I learned that the hard way exporting a ten-minute clip.",
    "I'll be honest, the setup instructions for this smart lock are rough. Budget 40 minutes, not the '5 minute install' on the box.",
]

AI_TEXTS_SEO = [
    "Are you looking for the best budget noise-cancelling headphones on the market today? In this comprehensive guide, we'll explore a wide range of top-rated options that deliver exceptional sound quality without breaking the bank.",
    "In today's competitive digital landscape, choosing the right project management tool plays a crucial role in driving team productivity. This comprehensive review will delve into the top solutions available.",
    "When it comes to meal prepping, having the right techniques can significantly enhance your culinary experience. It is important to note that proper preparation plays a vital role in achieving optimal results.",
    "Looking to elevate your home cooking game? Our comprehensive guide covers everything you need to know about choosing the perfect air fryer for your kitchen needs.",
    "Starting a podcast can seem like a daunting task, but with the right equipment and comprehensive planning, you can seamlessly launch a successful show. Let's delve into the essential gear you'll need.",
    "In an era where remote work continues to reshape the professional landscape, having a reliable standing desk plays a crucial role in maintaining productivity and well-being throughout the workday.",
    "Struggling with Wi-Fi connectivity issues? This comprehensive troubleshooting guide will walk you through a wide range of solutions to seamlessly optimize your home network performance.",
    "Choosing the right running shoes is essential for athletes at every level. In this comprehensive review, we'll explore a wide range of top-performing options that deliver exceptional comfort and durability.",
    "In today's fast-paced world, decluttering your living space plays a crucial role in fostering a sense of calm and productivity. This comprehensive guide will delve into proven strategies for lasting results.",
    "When it comes to cast iron cookware, proper seasoning is essential for achieving that coveted non-stick surface. It is worth noting that this process requires patience and a comprehensive understanding of the material.",
    "Selecting the right budgeting app can significantly enhance your financial management journey. Our comprehensive comparison delves into the top-rated options available in today's ever-evolving market.",
    "Choosing between laptops for video editing requires careful consideration of a wide range of technical specifications. This comprehensive guide will help you navigate the complexities of modern hardware.",
    "Smart home technology continues to revolutionize the way we secure our living spaces. In this comprehensive review, we'll explore the installation process and key features of leading smart lock solutions.",
    "In today's competitive e-commerce landscape, providing exceptional customer support plays a crucial role in building lasting brand loyalty and driving sustainable business growth.",
    "Blenders have become an essential kitchen appliance for health-conscious consumers. This comprehensive guide delves into a wide range of models to help you find the perfect fit for your lifestyle.",
]


HUMAN_TEXTS += HUMAN_TEXTS_SEO

# AI-style examples: written to reflect common patterns of generic,
# formulaic assistant-style output — consistent structure, hedging,
# transition-heavy, low personal specificity. These are original
# sentences written for this dataset, not copied from any real model
# output or copyrighted source.
AI_TEXTS = [
    "In today's fast-paced world, effective time management plays a crucial role in achieving both personal and professional success. It is important to note that individuals who prioritize their tasks tend to experience significantly lower levels of stress.",
    "Artificial intelligence has emerged as a transformative technology with the potential to revolutionize numerous industries. Furthermore, it is worth noting that the integration of AI systems can lead to substantial improvements in operational efficiency.",
    "When it comes to maintaining a healthy lifestyle, it is essential to consider a wide range of factors, including diet, exercise, and mental well-being. Moreover, a comprehensive approach to wellness can significantly enhance overall quality of life.",
    "Climate change represents one of the most pressing challenges of our time, and it is important to note that its effects are multifaceted, touching nearly every aspect of modern society. Additionally, addressing this ever-evolving crisis requires a robust and coordinated global response.",
    "Effective communication plays a vital role in the workplace, fostering a sense of collaboration and mutual understanding among team members. In conclusion, organizations that leverage clear communication strategies tend to achieve greater levels of productivity.",
    "The advent of remote work has fundamentally reshaped the modern professional landscape. It is worth noting that this shift has paved the way for greater flexibility, while also introducing a wide range of new challenges related to work-life balance.",
    "In the realm of digital marketing, businesses must navigate the complexities of an ever-evolving landscape to remain competitive. Leveraging data-driven strategies can seamlessly enhance customer engagement and drive measurable results.",
    "Education serves as a cornerstone of societal development, and it is important to note that access to quality education plays a crucial role in reducing inequality. Furthermore, comprehensive educational reform stands as a testament to a nation's long-term priorities.",
    "The importance of cybersecurity cannot be overstated in an era where digital threats are becoming increasingly sophisticated. Organizations must adopt a holistic approach to safeguard sensitive data and maintain the trust of their stakeholders.",
    "Sustainable practices are essential for businesses aiming to reduce their environmental footprint. Additionally, it is worth noting that companies that leverage sustainable innovation often experience a wide range of long-term financial benefits.",
    "In conclusion, the ongoing evolution of technology underscores the importance of adaptability in both personal and professional contexts. Moreover, individuals who embrace lifelong learning are better positioned to navigate an increasingly complex world.",
    "Mental health awareness plays a crucial role in fostering supportive communities. It is important to note that early intervention and comprehensive support systems can significantly improve outcomes for individuals facing psychological challenges.",
    "The rise of e-commerce has seamlessly transformed the retail industry, offering consumers a wide range of convenient purchasing options. Furthermore, businesses that leverage robust digital infrastructure are better equipped to meet evolving customer expectations.",
    "Effective leadership requires a multifaceted skill set, including strong communication, strategic vision, and emotional intelligence. In today's ever-evolving business environment, leaders who foster a culture of innovation tend to drive superior organizational outcomes.",
    "Renewable energy sources, such as solar and wind power, are playing an increasingly crucial role in the global transition toward sustainability. It is worth noting that this shift paves the way for a more resilient and comprehensive energy infrastructure.",
    "The integration of technology in classrooms has revolutionized modern education, offering students access to a wide range of interactive learning tools. Moreover, this ever-evolving landscape underscores the importance of digital literacy in today's society.",
    "Building strong customer relationships is essential for long-term business success. Additionally, companies that leverage personalized engagement strategies can seamlessly enhance customer loyalty and drive sustainable growth.",
    "The gig economy has fundamentally reshaped traditional employment models, offering workers greater flexibility while introducing a wide range of new challenges. It is important to note that policymakers must adopt a comprehensive approach to address these emerging issues.",
    "Innovation serves as a critical driver of economic growth, and it is worth noting that companies that foster a culture of creativity often outperform their competitors. Furthermore, this comprehensive approach to innovation stands as a testament to long-term strategic vision.",
    "In the realm of healthcare, technological advancements have paved the way for more personalized and effective treatment options. It is essential to note that this ever-evolving field continues to unlock the potential for improved patient outcomes.",
    "Social media platforms have seamlessly integrated into daily life, playing a crucial role in shaping public discourse. Additionally, it is worth noting that the multifaceted nature of these platforms presents both opportunities and challenges for society.",
    "Financial literacy is an essential skill in today's complex economic landscape. Moreover, individuals who leverage sound budgeting practices are better positioned to navigate an increasingly uncertain financial future.",
    "The importance of biodiversity cannot be overstated, as ecosystems rely on a wide range of interconnected species to maintain ecological balance. In conclusion, comprehensive conservation efforts are essential for preserving our planet's natural heritage.",
    "Remote collaboration tools have revolutionized the way teams operate, fostering a sense of connectivity despite geographical distance. Furthermore, organizations that leverage these robust platforms often experience seamless improvements in overall productivity.",
    "As we navigate an increasingly digital world, data privacy has emerged as a critical concern for individuals and organizations alike. It is important to note that a comprehensive, multifaceted approach is essential to safeguarding personal information.",
]

AI_TEXTS += AI_TEXTS_SEO

# Third batch: lightly-edited / paraphrased AI text — no stock phrases
# ("furthermore", "delve into", etc.), simpler sentences, but still
# carrying the subtler patterns adversarial testing showed the model
# was missing: safe/hedgy generalizations, low personal specificity,
# fairly uniform sentence length, generic transitions ("also", "this
# means", "because of this") instead of concrete examples or opinions.
# This is what real "humanized" AI output tends to look like, and it's
# exactly the category evaluation/adversarial.py found the model
# failing on before this batch was added.
AI_TEXTS_PARAPHRASED = [
    "Companies today face a lot of pressure to keep up with changing technology. Falling behind on modern tools can hurt a business over time. Taking a broad view of new technology tends to lead to steadier results. This means firms that plan carefully are usually better prepared for what comes next.",
    "Getting enough sleep matters more than most people realize. Poor sleep affects mood, focus, and even physical health over time. Building a consistent bedtime routine can help a lot. Small changes like this often make a bigger difference than people expect.",
    "Remote teams need to work a bit differently than teams in an office. Clear written communication becomes more important when people aren't in the same room. Regular check-ins help keep everyone on the same page. Over time, this leads to fewer misunderstandings and smoother projects.",
    "Eating more vegetables is one of the simplest ways to improve your diet. Most people don't get enough fiber from their meals. Adding just one extra serving a day can help close that gap. This kind of small change tends to be easier to stick with long term.",
    "Good customer service can make or break a small business. When a customer has a bad experience, they often tell other people about it. Responding quickly to complaints shows that a company cares. This builds trust, which matters more than any single sale.",
    "Learning a new skill takes time, and most people underestimate how long that takes. Progress often feels slow at first, especially in the early stages. Sticking with it past that point is usually where real improvement happens. This is why consistency tends to matter more than raw talent.",
    "Budgeting doesn't have to be complicated to be effective. Tracking spending for even a single month can reveal a lot. Once you see where the money goes, it's easier to make changes. This kind of awareness is often the first step toward saving more.",
    "Good writing usually comes from multiple rounds of editing, not a single draft. The first version is rarely the clearest one. Reading a piece out loud can help catch awkward phrasing. Over time, this habit tends to improve the overall quality of the work.",
    "A reliable morning routine can set the tone for the rest of the day. Starting with a clear task list often reduces stress later on. This doesn't need to be complicated to work well. Even a short routine can create a sense of structure.",
    "Choosing the right tool for a job depends on the specific situation. What works well for one team might not work for another. It helps to think about long-term needs, not just what seems easiest right now. This kind of planning tends to save time later.",
]

AI_TEXTS += AI_TEXTS_PARAPHRASED

# A few more formal/technical HUMAN examples, added alongside the batch
# above so the model learns "plainer sentences + generic transitions"
# as an AI-leaning pattern specifically, rather than overfitting to
# flag all careful, low-personality writing as AI. Real human writing
# is often plain and impersonal too (reports, documentation, instructions).
HUMAN_TEXTS_FORMAL = [
    "The quarterly audit identified three discrepancies in the expense reports submitted by the marketing department. Two were attributed to rounding errors in the currency conversion tool. The third remains under review pending further documentation from the vendor.",
    "To reset the router, hold the recessed button on the back panel for ten seconds until the lights flash. The device will reboot and restore factory settings. Note that this will remove any custom Wi-Fi passwords you've configured.",
    "The study followed 214 participants over an eighteen-month period, tracking changes in resting heart rate after the introduction of a structured walking program. Results varied considerably by age group, with the most pronounced improvement observed in participants over fifty.",
    "Please submit timesheets by 5pm on the last Friday of each month. Late submissions may delay payroll processing by up to one billing cycle. Contact HR directly if you're missing login access to the portal.",
    "The bridge's east span will be closed for resurfacing between March 3rd and March 17th. Traffic will be rerouted via the north detour during this period. Pedestrian access will remain open throughout construction.",
]

HUMAN_TEXTS += HUMAN_TEXTS_FORMAL


def main() -> None:
    rows = [("text", "label")]
    rows += [(t, "human") for t in HUMAN_TEXTS]
    rows += [(t, "ai") for t in AI_TEXTS]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    print(f"Wrote {len(rows) - 1} rows to {OUT_PATH}")


if __name__ == "__main__":
    main()
