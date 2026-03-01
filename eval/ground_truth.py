"""Sample queries, ground truth answers, and metadata for evaluation."""

SAMPLE_QUERIES = [
    {
        "id": 1,
        "query": "What is the notice period for terminating the NDA?",
        "reference": "Either party may terminate with thirty (30) days written notice (NDA:3).",
        "expected_docs": ["nda_acme_vendor"],
        "expected_intent": "fact",
        "key_phrases": ["30 days", "thirty", "written notice"],
        "expect_risks": False,
    },
    {
        "id": 2,
        "query": "What is the uptime commitment in the SLA?",
        "reference": "Vendor shall ensure 99.5% monthly uptime (SLA:1).",
        "expected_docs": ["service_level_agreement"],
        "expected_intent": "fact",
        "key_phrases": ["99.5%", "monthly uptime"],
        "expect_risks": False,
    },
    {
        "id": 3,
        "query": "Which law governs the Vendor Services Agreement?",
        "reference": "The Vendor Services Agreement is governed by the laws of England and Wales (:6).",
        "expected_docs": ["vendor_services_agreement"],
        "expected_intent": "fact",
        "key_phrases": ["England and Wales"],
        "expect_risks": False,
    },
    {
        "id": 4,
        "query": "Do confidentiality obligations survive termination of the NDA?",
        "reference": "Yes, confidentiality obligations survive termination for five (5) years (NDA:3).",
        "expected_docs": ["nda_acme_vendor"],
        "expected_intent": "fact",
        "key_phrases": ["five", "5 years", "survive"],
        "expect_risks": False,
    },
    {
        "id": 5,
        "query": "Is liability capped for breach of confidentiality?",
        "reference": (
            "The NDA (:4) specifies no explicit limitation of liability. "
            "The Vendor Services Agreement (:4) caps total liability at fees paid in the preceding 12 months, "
            "except for indemnification and gross negligence."
        ),
        "expected_docs": ["nda_acme_vendor", "vendor_services_agreement"],
        "expected_intent": "fact",
        "key_phrases": ["no explicit limitation", "12 months"],
        "expect_risks": True,
    },
    {
        "id": 6,
        "query": "What remedies are available if the SLA uptime is not met?",
        "reference": (
            "Vendor shall provide service credits as defined in Schedule B. "
            "Service credits are the sole and exclusive remedy (SLA:2)."
        ),
        "expected_docs": ["service_level_agreement"],
        "expected_intent": "fact",
        "key_phrases": ["service credits", "sole and exclusive remedy", "Schedule B"],
        "expect_risks": True,
    },
    {
        "id": 7,
        "query": "Is Vendor XYZ's liability capped for data breaches?",
        "reference": (
            "The DPA (:5) does not specify an independent liability cap and relies on the Vendor Services Agreement. "
            "The VSA (:4) caps total liability at fees paid in the preceding 12 months, "
            "except for indemnification and gross negligence."
        ),
        "expected_docs": ["data_processing_agreement", "vendor_services_agreement"],
        "expected_intent": "risk",
        "key_phrases": ["12 months", "relies on", "Vendor Services Agreement"],
        "expect_risks": True,
    },
    {
        "id": 8,
        "query": "Which agreement governs data breach notification timelines?",
        "reference": (
            "The Data Processing Agreement (:3) requires notification within 72 hours of becoming aware of a breach."
        ),
        "expected_docs": ["data_processing_agreement"],
        "expected_intent": "fact",
        "key_phrases": ["72 hours", "Data Processing Agreement"],
        "expect_risks": False,
    },
    {
        "id": 9,
        "query": "Are there conflicting governing laws across agreements?",
        "reference": (
            "Yes. The NDA is governed by California law (:5), "
            "the Vendor Services Agreement by England and Wales (:6), "
            "and the DPA by EU/GDPR law (:6)."
        ),
        "expected_docs": ["nda_acme_vendor", "vendor_services_agreement", "data_processing_agreement"],
        "expected_intent": "cross_document",
        "key_phrases": ["California", "England and Wales", "GDPR"],
        "expect_risks": True,
    },
    {
        "id": 10,
        "query": "Are there any legal risks related to liability exposure?",
        "reference": (
            "The NDA has no explicit liability cap (:4). "
            "The SLA limits remedies to service credits only (:2). "
            "The SLA excludes indirect and consequential damages (:4)."
        ),
        "expected_docs": ["nda_acme_vendor", "service_level_agreement", "vendor_services_agreement"],
        "expected_intent": "risk",
        "key_phrases": ["no explicit", "unlimited", "service credits", "sole"],
        "expect_risks": True,
    },
    {
        "id": 11,
        "query": "Identify any clauses that could pose financial risk to Acme Corp.",
        "reference": (
            "Financial risks include: fixed monthly fees with late payment interest (VSA:2), "
            "sole remedy of service credits for downtime (SLA:2), "
            "and no liability cap in the NDA (:4)."
        ),
        "expected_docs": ["vendor_services_agreement", "service_level_agreement", "nda_acme_vendor"],
        "expected_intent": "risk",
        "key_phrases": ["late payment", "service credits", "no", "liability"],
        "expect_risks": True,
    },
    {
        "id": 12,
        "query": "Is there any unlimited liability in these agreements?",
        "reference": (
            "The NDA (:4) states the Receiving Party is liable for damages with no explicit limitation of liability."
        ),
        "expected_docs": ["nda_acme_vendor"],
        "expected_intent": "risk",
        "key_phrases": ["no explicit limitation", "NDA", "unlimited"],
        "expect_risks": True,
    },
    {
        "id": 13,
        "query": "Can Vendor XYZ share Acme's confidential data with subcontractors?",
        "reference": (
            "The DPA (:4) allows the Processor to engage subprocessors with prior written authorization from the Controller."
        ),
        "expected_docs": ["data_processing_agreement"],
        "expected_intent": "fact",
        "key_phrases": ["prior written authorization", "subprocessor"],
        "expect_risks": False,
    },
    {
        "id": 14,
        "query": "What happens if Vendor delays breach notification beyond 72 hours?",
        "reference": (
            "Notification must occur within 72 hours (DPA:3). "
            "A delay would breach the DPA obligation and may violate GDPR requirements (DPA:6)."
        ),
        "expected_docs": ["data_processing_agreement"],
        "expected_intent": "risk",
        "key_phrases": ["72 hours", "breach", "GDPR"],
        "expect_risks": True,
    },
    {
        "id": 15,
        "query": "Summarize all risks for Acme Corp in one paragraph.",
        "reference": (
            "Acme Corp faces risks from conflicting governing laws across three jurisdictions, "
            "unlimited liability in the NDA, limited remedies (service credits only) for SLA failures, "
            "and strict 72-hour data breach reporting obligations."
        ),
        "expected_docs": ["nda_acme_vendor", "vendor_services_agreement", "service_level_agreement", "data_processing_agreement"],
        "expected_intent": "summary",
        "key_phrases": ["governing law", "unlimited", "service credits", "72 hours"],
        "expect_risks": True,
    },
    {
        "id": 16,
        "query": "Can you draft a better NDA for me?",
        "reference": "Out of scope — the system should not draft documents.",
        "expected_docs": [],
        "expected_intent": "out_of_scope",
        "key_phrases": [],
        "expect_risks": False,
    },
    {
        "id": 17,
        "query": "What legal strategy should Acme take against Vendor XYZ?",
        "reference": "Out of scope — the system should not provide legal strategy.",
        "expected_docs": [],
        "expected_intent": "out_of_scope",
        "key_phrases": [],
        "expect_risks": False,
    },
]
