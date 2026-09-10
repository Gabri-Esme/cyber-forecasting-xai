import sqlite3
import os
from dotenv import load_dotenv
from util import *

ATTACK_LIST = [
    {"target": "DDoS", "accepted": ["DDoS", "Distributed Denial of Service", "Network Flooding"]},
    {"target": "Phishing", "accepted": ["Phishing", "Spear Phishing", "Social Engineering", "Smishing", "Vishing"]},
    {"target": "Ransomware", "accepted": ["Ransomware", "Crypto-malware", "Extortionware"]},
    {"target": "Password Attack", "accepted": ["Password Attack", "Brute Force", "Credential Stuffing", "Dictionary Attack", "Password Cracking"]},
    {"target": "Account Hijacking", "accepted": ["Account Hijacking", "Credential Theft", "Account Takeover", "ATO"]},
    {"target": "Trojan", "accepted": ["Trojan", "Trojan Horse", "Remote Access Trojan", "RAT"]},
    {"target": "Vulnerability", "accepted": ["Vulnerability", "Exploit", "Security Flaw", "Weakness"]},
    {"target": "Zero-day", "accepted": ["Zero-day", "0-day", "Zero-day Exploit", "Unpatched Vulnerability"]},
    {"target": "APT", "accepted": ["Advanced Persistent Threat", "APT", "State-sponsored Attack", "Targeted Intrusion"]},
    {"target": "Malware", "accepted": ["Malware", "Malicious Software", "Computer Virus", "Computer Worm", "Spyware", "Adware"]},
    {"target": "Disinformation", "accepted": ["Disinformation", "Misinformation", "Fake News", "Propaganda", "Information Operation", "Influence Campaign"]},
    {"target": "Targeted Attack", "accepted": ["Targeted Attack", "Spear Attack", "Watering Hole Attack", "BEC", "Business Email Compromise"]},
    {"target": "Backdoor", "accepted": ["Backdoor", "Trapdoor", "Hidden Access", "Web Shell"]},
    {"target": "Botnet", "accepted": ["Botnet", "Zombie Network", "Command and Control", "C2", "Bot Network"]},
    {"target": "Dropper", "accepted": ["Dropper", "Malware Installer", "Stage-1 Loader", "Stager"]},
    {"target": "Insider Threat", "accepted": ["Insider Threat", "Malicious Insider", "Insider Attack", "Privileged User Misuse"]},
    {"target": "Session Hijacking", "accepted": ["Session Hijacking", "Cookie Stealing", "Man-in-the-Middle", "MitM", "Sidejacking"]}
]

PATS = [
    {"target": "Access Control", "accepted": ["Access Control", "Privilege Management", "Authorization Mechanism"]},
    {"target": "Anomaly Detection", "accepted": ["Anomaly Detection", "Outlier Detection", "Network Deviation Detection"]},
    {"target": "Activity Monitoring", "accepted": ["Activity Monitoring", "User Activity Monitoring", "System Monitoring"]},
    {"target": "Attack Tree", "accepted": ["Attack Tree", "Threat Modeling", "Attack Path Analysis"]},
    {"target": "Application Whitelisting", "accepted": ["Application Whitelisting", "App Allowlisting", "Executable Control"]},
    {"target": "Behaviour-based Detection", "accepted": ["Behaviour-based Detection", "Heuristic-based Detection", "Behavioral Intrusion Detection"]},
    {"target": "Blockchain", "accepted": ["Blockchain", "Distributed Ledger", "Smart Contract Security"]},
    {"target": "Blackholing", "accepted": ["Blackholing", "Remote Triggered Black Hole", "Null Routing"]},
    {"target": "Blacklisting", "accepted": ["Blacklisting", "Deny-listing", "IP Reputation Filtering"]},
    {"target": "Continuous Authentication", "accepted": ["Continuous Authentication", "Zero Trust Authentication", "Active Authentication"]},
    {"target": "CAPTCHA", "accepted": ["CAPTCHA", "Completely Automated Public Turing test", "Human Interaction Proof"]},
    {"target": "Control Flow Integrity", "accepted": ["Control Flow Integrity", "CFI Security", "Code Flow Protection"]},
    {"target": "Cryptography/Encryption", "accepted": ["Cryptography", "Data Encryption", "Cryptographic Protocols"]},
    {"target": "Code Signing", "accepted": ["Code Signing", "Digital Signature Verification", "Binary Integrity"]},
    {"target": "Dynamic Analysis", "accepted": ["Dynamic Analysis", "Dynamic Malware Analysis", "Runtime Analysis"]},
    {"target": "Data Backups", "accepted": ["Data Backups", "Disaster Recovery Planning", "Data Redundancy"]},
    {"target": "Data Leakage Detection/Prevention", "accepted": ["Data Leakage Detection", "Information Leakage Prevention"]},
    {"target": "Data Loss Prevention", "accepted": ["Data Loss Prevention", "DLP System", "Data Exfiltration Protection"]},
    {"target": "Distributed Ledgers Technology", "accepted": ["Distributed Ledgers Technology", "DLT Framework", "Consensus Protocol"]},
    {"target": "Darknet Monitoring", "accepted": ["Darknet Monitoring", "Dark Web Intelligence", "Onion Network Monitoring"]},
    {"target": "Data Provenance", "accepted": ["Data Provenance", "Data Lineage", "Information Origins"]},
    {"target": "Dynamic Resource Management", "accepted": ["Dynamic Resource Management", "Adaptive Resource Allocation"]},
    {"target": "Deception Technology", "accepted": ["Deception Technology", "Honeytokens", "Cyber Decoy"]},
    {"target": "File Integrity Monitoring", "accepted": ["File Integrity Monitoring", "FIM Software", "System File Audit"]},
    {"target": "Formal Verification", "accepted": ["Formal Verification", "Model Checking", "Mathematical Verification of Software"]},
    {"target": "Graphical Authentication", "accepted": ["Graphical Authentication", "Visual Passwords", "Image-based Login"]},
    {"target": "Graphical Model", "accepted": ["Graphical Model", "Probabilistic Graphical Model", "Bayesian Cybersecurity Model"]},
    {"target": "Game Theory", "accepted": ["Game Theory", "Cyber Game Theory", "Adversarial Modeling"]},
    {"target": "Hypergame", "accepted": ["Hypergame Theory", "Hypergame Modeling", "Strategic Deception"]},
    {"target": "Honeypot", "accepted": ["Honeypot", "Honey-network", "Decoy System"]},
    {"target": "HTTPS", "accepted": ["Hypertext Transfer Protocol Secure", "HTTP over TLS", "Secure HTTP"]},
    {"target": "Identity-based Encryption", "accepted": ["Identity-based Encryption", "ID-based Cryptography"]},
    {"target": "Identity Management", "accepted": ["Identity Management", "IAM Framework", "Identity Governance"]},
    {"target": "Intrusion Detection/Prevention System", "accepted": ["Intrusion Detection System", "Intrusion Prevention System", "Host-based Intrusion Detection"]},
    {"target": "Image Recognition", "accepted": ["Image Recognition", "Computer Vision Security", "Visual Pattern Recognition"]},
    {"target": "Keystroke Dynamics", "accepted": ["Keystroke Dynamics", "Typing Biometrics", "Keystroke Behavioral Analysis"]},
    {"target": "Least Privilege", "accepted": ["Least Privilege", "Principle of Least Privilege", "Access Minimization"]},
    {"target": "Mutual Authentication", "accepted": ["Mutual Authentication", "Two-way Authentication", "Client-Server Authentication"]},
    {"target": "Multi-factor Authentication", "accepted": ["Multi-factor Authentication", "Two-factor Authentication", "MFA Protocol"]},
    {"target": "Machine Learning/Deep Learning", "accepted": ["Machine Learning", "Deep Learning", "Neural Network Security"]},
    {"target": "Moving Target Defence", "accepted": ["Moving Target Defence", "Cyber Agility", "Dynamic Defense Mechanism"]},
    {"target": "NLP/LLM", "accepted": ["Natural Language Processing", "Large Language Model Security", "Transformer Models"]},
    {"target": "Network Segmentation", "accepted": ["Network Segmentation", "Micro-segmentation", "VLAN Isolation"]},
    {"target": "One Time Password", "accepted": ["One Time Password", "Time-based OTP", "TOTP"]},
    {"target": "Packet Filtering", "accepted": ["Packet Filtering", "Stateful Inspection", "Network Filter"]},
    {"target": "Password Hashing", "accepted": ["Password Hashing", "Password Salting", "Cryptographic Hashing"]},
    {"target": "Public Key Infrastructure", "accepted": ["Public Key Infrastructure", "PKI Certificate", "Certificate Authority"]},
    {"target": "Password Management", "accepted": ["Password Management", "Credential Vault", "Password Manager"]},
    {"target": "Patch Management", "accepted": ["Patch Management", "Software Update Management", "Security Patching"]},
    {"target": "Password Policy", "accepted": ["Password Policy", "Password Complexity Rule", "Credential Policy"]},
    {"target": "Privacy Preserving", "accepted": ["Privacy Preserving", "Differential Privacy", "Homomorphic Encryption"]},
    {"target": "Password Strength Meters", "accepted": ["Password Strength Meters", "Password Complexity Checker"]},
    {"target": "Penetration Testing", "accepted": ["Penetration Testing", "Ethical Hacking", "Red Teaming"]},
    {"target": "Risk Assessment", "accepted": ["Risk Assessment", "Cyber Risk Evaluation", "Security Risk Analysis"]},
    {"target": "Rank Correlation", "accepted": ["Rank Correlation", "Spearman Correlation", "Kendall Rank Correlation"]},
    {"target": "Rate Limiting", "accepted": ["Rate Limiting", "API Throttling", "Request Limiting"]},
    {"target": "Static Analysis", "accepted": ["Static Analysis", "Static Code Analysis", "SAST"]},
    {"target": "Strong Authentication", "accepted": ["Strong Authentication", "High-assurance Authentication"]},
    {"target": "Secure Boot", "accepted": ["Secure Boot", "Verified Boot", "Root of Trust"]},
    {"target": "Sandboxing", "accepted": ["Sandboxing", "Malware Sandbox", "Isolated Execution Environment"]},
    {"target": "Standardised Communication", "accepted": ["Standardised Communication", "Interoperability Protocol"]},
    {"target": "Supply Chain Risk Management", "accepted": ["Supply Chain Risk Management", "Cyber Supply Chain", "Vendor Security"]},
    {"target": "Software Defined Network", "accepted": ["Software Defined Network", "SDN Security", "Network Programmability"]},
    {"target": "Statistical Hidden Markov Model", "accepted": ["Statistical Hidden Markov Model", "HMM Security Analysis"]},
    {"target": "Source Identification", "accepted": ["Source Identification", "Attribution Analysis", "IP Attribution"]},
    {"target": "SIEM", "accepted": ["Security Information and Event Management", "Security Log Analysis"]},
    {"target": "Session Management", "accepted": ["Session Management", "Session Hijacking Protection", "Token Management"]},
    {"target": "Split Manufacturing", "accepted": ["Split Manufacturing", "Hardware Obfuscation", "IC Security"]},
    {"target": "SSL/TLS", "accepted": ["Transport Layer Security", "Secure Sockets Layer", "TLS Protocol"]},
    {"target": "Traffic Shaping", "accepted": ["Traffic Shaping", "Bandwidth Throttling", "Traffic Policing"]},
    {"target": "User Behaviour Analytics", "accepted": ["User Behaviour Analytics", "UEBA", "User Behavior Profiling"]},
    {"target": "Vulnerability Assessment", "accepted": ["Vulnerability Assessment", "Vulnerability Scanning", "Security Auditing"]},
    {"target": "Virtual Keyboards", "accepted": ["Virtual Keyboards", "On-screen Keyboards", "Soft Keyboard Security"]},
    {"target": "Vulnerability Management", "accepted": ["Vulnerability Management", "Vuln Lifecycle Management"]},
    {"target": "VPN", "accepted": ["Virtual Private Network", "Secure Tunneling", "IPsec VPN"]},
    {"target": "Vulnerability Scanner", "accepted": ["Vulnerability Scanner", "Nessus Scan", "Automated Security Scanning"]}
]

YEARS = [2026]

if __name__ == "__main__":
    load_dotenv("01_data_preparation/secrets.env")
    db_name = os.getenv("DB_PATH")
    api_key = os.getenv("ELS_API_KEY")
    db = init_db(db_name="data.db") 

    # Get monthly results for each
    with sqlite3.connect(db_name) as conn:
        attack_m_results = get_counts(conn, api_key, YEARS, ATTACK_LIST, mode='monthly', is_pat=False)
        pat_m_results = get_counts(conn, api_key, YEARS, PATS, mode='monthly', is_pat=True)
        attack_y_results = get_counts(conn, api_key, YEARS, ATTACK_LIST, mode='yearly', is_pat=False)
        pat_y_results = get_counts(conn, api_key, YEARS, PATS, mode='yearly', is_pat=True)


