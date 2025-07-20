"""
Comprehensive Skill Detection System
Uses weighted keyword matching + context analysis to determine skill confidence scores
Covers all major technical domains with subcategories
"""

import re
from typing import Dict, List, Any

class SkillDetector:
    def __init__(self):
        """Initialize with comprehensive skill categories and weights"""
        self.skill_categories = self._load_skill_categories()
        self.experience_keywords = {
            'years': r'(\d+)\s*(?:\+)?\s*years?\s*(?:of\s*)?(?:experience)?',
            'months': r'(\d+)\s*months?\s*(?:of\s*)?(?:experience)?',
            'projects': r'(\d+)\s*projects?',
            'level_indicators': {
                'senior': ['senior', 'lead', 'principal', 'architect', 'expert', 'advanced'],
                'mid': ['mid', 'intermediate', 'experienced', 'proficient'],
                'junior': ['junior', 'entry', 'beginner', 'basic', 'learning', 'fresher']
            }
        }
    
    def _load_skill_categories(self) -> Dict[str, Dict]:
        """Load comprehensive skill categories with weights and subcategories"""
        return {
            # PROGRAMMING LANGUAGES
            "Python Programming": {
                "keywords": [
                    "python", "py", "django", "flask", "fastapi", "tornado", "pyramid",
                    "pandas", "numpy", "scipy", "matplotlib", "seaborn", "plotly",
                    "jupyter", "ipython", "conda", "pip", "virtualenv", "poetry",
                    "pytest", "unittest", "asyncio", "celery", "gunicorn", "uvicorn",
                    "pydantic", "sqlalchemy", "alembic", "pycharm", "black", "flake8"
                ],
                "weights": {
                    "python": 1.0, "django": 0.9, "flask": 0.9, "fastapi": 0.9,
                    "pandas": 0.8, "numpy": 0.8, "scipy": 0.7, "pytest": 0.7,
                    "sqlalchemy": 0.8, "celery": 0.7, "asyncio": 0.8
                },
                "subcategories": ["Web Development", "Data Science", "Backend", "Testing", "Automation"]
            },
            
            "JavaScript Programming": {
                "keywords": [
                    "javascript", "js", "typescript", "ts", "node.js", "nodejs", "npm", "yarn",
                    "react", "reactjs", "vue", "vuejs", "angular", "angularjs", "svelte",
                    "express", "expressjs", "koa", "hapi", "nestjs", "next.js", "nuxt",
                    "webpack", "babel", "eslint", "prettier", "jest", "mocha", "cypress",
                    "electron", "deno", "bun", "pnpm", "turborepo", "vite", "rollup"
                ],
                "weights": {
                    "javascript": 1.0, "typescript": 0.9, "react": 0.9, "node.js": 0.9,
                    "vue": 0.8, "angular": 0.8, "express": 0.8, "next.js": 0.8,
                    "electron": 0.7, "webpack": 0.7, "vite": 0.7
                },
                "subcategories": ["Frontend", "Backend", "Full Stack", "Testing", "Desktop Apps"]
            },
            
            "Java Programming": {
                "keywords": [
                    "java", "spring", "spring boot", "hibernate", "maven", "gradle",
                    "junit", "mockito", "tomcat", "jetty", "jvm", "jdk", "jre",
                    "android", "kotlin", "scala", "groovy", "jsp", "servlets",
                    "micronaut", "quarkus", "dropwizard", "struts", "jsf", "tapestry"
                ],
                "weights": {
                    "java": 1.0, "spring": 0.9, "spring boot": 0.9, "hibernate": 0.8,
                    "maven": 0.7, "gradle": 0.7, "junit": 0.7, "kotlin": 0.8,
                    "quarkus": 0.7, "micronaut": 0.7
                },
                "subcategories": ["Enterprise", "Android", "Backend", "Testing", "Microservices"]
            },
            
            "C/C++ Programming": {
                "keywords": [
                    "c++", "cpp", "c programming", "gcc", "clang", "cmake", "make",
                    "stl", "boost", "qt", "opencv", "cuda", "openmp", "mpi",
                    "visual studio", "code::blocks", "dev-c++", "mingw", "msvc",
                    "embedded c", "arduino", "raspberry pi", "stm32", "pic"
                ],
                "weights": {
                    "c++": 1.0, "cpp": 1.0, "c programming": 1.0, "stl": 0.8,
                    "boost": 0.7, "qt": 0.8, "opencv": 0.8, "cuda": 0.9,
                    "embedded c": 0.8, "arduino": 0.6, "openmp": 0.7
                },
                "subcategories": ["Systems", "Graphics", "Performance", "Embedded", "Game Development"]
            },
            
            "C# Programming": {
                "keywords": [
                    "c#", "csharp", ".net", "dotnet", "asp.net", "mvc", "web api",
                    "entity framework", "linq", "xamarin", "unity", "wpf", "winforms",
                    "blazor", "maui", "azure functions", "visual studio", "nuget"
                ],
                "weights": {
                    "c#": 1.0, "csharp": 1.0, ".net": 0.9, "asp.net": 0.9,
                    "entity framework": 0.8, "unity": 0.8, "blazor": 0.7,
                    "xamarin": 0.7, "wpf": 0.6
                },
                "subcategories": ["Web Development", "Desktop Apps", "Mobile", "Game Development", "Enterprise"]
            },
            
            "Go Programming": {
                "keywords": [
                    "golang", "go programming", "goroutines", "channels", "gin", "echo",
                    "buffalo", "beego", "revel", "go modules", "dep", "fiber",
                    "gorm", "cobra", "viper", "testify", "grpc-go", "protobuf"
                ],
                "weights": {
                    "golang": 1.0, "go programming": 1.0, "goroutines": 0.8,
                    "gin": 0.7, "echo": 0.7, "fiber": 0.7, "gorm": 0.6,
                    "grpc-go": 0.7
                },
                "subcategories": ["Backend", "Microservices", "Concurrency", "CLI Tools"]
            },
            
            "Rust Programming": {
                "keywords": [
                    "rust", "cargo", "rustc", "tokio", "async", "actix", "rocket",
                    "serde", "rayon", "wasm", "webassembly", "diesel", "sea-orm",
                    "clap", "yew", "tauri", "bevy", "nom"
                ],
                "weights": {
                    "rust": 1.0, "cargo": 0.8, "tokio": 0.7, "actix": 0.7,
                    "serde": 0.6, "wasm": 0.8, "tauri": 0.7, "bevy": 0.6
                },
                "subcategories": ["Systems", "WebAssembly", "Performance", "CLI Tools", "Game Development"]
            },
            
            "PHP Programming": {
                "keywords": [
                    "php", "laravel", "symfony", "codeigniter", "yii", "zend",
                    "composer", "phpunit", "wordpress", "drupal", "magento",
                    "eloquent", "twig", "blade", "doctrine", "psr", "phpstorm"
                ],
                "weights": {
                    "php": 1.0, "laravel": 0.9, "symfony": 0.8, "wordpress": 0.7,
                    "composer": 0.7, "phpunit": 0.6, "eloquent": 0.6
                },
                "subcategories": ["Web Development", "CMS", "E-commerce", "Testing", "Frameworks"]
            },
            
            "Ruby Programming": {
                "keywords": [
                    "ruby", "rails", "ruby on rails", "sinatra", "gem", "bundler",
                    "rspec", "minitest", "rake", "activerecord", "haml", "erb",
                    "capistrano", "sidekiq", "puma", "unicorn"
                ],
                "weights": {
                    "ruby": 1.0, "rails": 0.9, "ruby on rails": 0.9, "sinatra": 0.6,
                    "gem": 0.6, "rspec": 0.7, "activerecord": 0.7
                },
                "subcategories": ["Web Development", "Testing", "Backend", "DevOps"]
            },
            
            "Swift Programming": {
                "keywords": [
                    "swift", "ios", "xcode", "cocoa", "uikit", "swiftui", "core data",
                    "alamofire", "rxswift", "combine", "vapor", "perfect", "kitura",
                    "objective-c", "foundation", "app store"
                ],
                "weights": {
                    "swift": 1.0, "ios": 0.9, "xcode": 0.8, "swiftui": 0.8,
                    "uikit": 0.7, "core data": 0.6, "vapor": 0.6
                },
                "subcategories": ["iOS Development", "macOS", "Backend", "UI/UX"]
            },
            
            "Kotlin Programming": {
                "keywords": [
                    "kotlin", "android", "jetpack compose", "coroutines", "ktor",
                    "spring kotlin", "kotlin multiplatform", "gradle kotlin",
                    "anko", "room", "retrofit", "dagger", "hilt"
                ],
                "weights": {
                    "kotlin": 1.0, "android": 0.9, "jetpack compose": 0.8,
                    "coroutines": 0.7, "ktor": 0.6, "kotlin multiplatform": 0.7
                },
                "subcategories": ["Android Development", "Backend", "Multiplatform", "Coroutines"]
            },
            
            "Scala Programming": {
                "keywords": [
                    "scala", "akka", "play framework", "sbt", "cats", "scalaz",
                    "spark", "kafka", "slick", "scalatest", "circe", "http4s"
                ],
                "weights": {
                    "scala": 1.0, "akka": 0.8, "play framework": 0.7, "spark": 0.8,
                    "kafka": 0.7, "sbt": 0.6, "cats": 0.5
                },
                "subcategories": ["Big Data", "Functional Programming", "Backend", "Concurrency"]
            },
            
            "R Programming": {
                "keywords": [
                    "r programming", "rstudio", "ggplot2", "dplyr", "tidyr", "shiny",
                    "caret", "randomforest", "knitr", "rmarkdown", "plotly r",
                    "bioconductor", "devtools", "roxygen2"
                ],
                "weights": {
                    "r programming": 1.0, "ggplot2": 0.8, "shiny": 0.8, "dplyr": 0.7,
                    "caret": 0.7, "rstudio": 0.6, "tidyr": 0.6
                },
                "subcategories": ["Data Science", "Statistics", "Visualization", "Bioinformatics"]
            },
            
            # MACHINE LEARNING & AI
            "Machine Learning": {
                "keywords": [
                    "machine learning", "ml", "artificial intelligence", "ai",
                    "tensorflow", "pytorch", "scikit-learn", "sklearn", "keras",
                    "xgboost", "lightgbm", "catboost", "random forest", "svm",
                    "neural networks", "deep learning", "cnn", "rnn", "lstm", "gru",
                    "transformer", "bert", "gpt", "attention", "embeddings",
                    "automl", "mlops", "feature engineering", "hyperparameter tuning",
                    "model deployment", "a/b testing", "mlflow", "kubeflow", "weights & biases"
                ],
                "weights": {
                    "machine learning": 1.0, "tensorflow": 0.9, "pytorch": 0.9,
                    "scikit-learn": 0.8, "keras": 0.8, "deep learning": 0.9,
                    "neural networks": 0.8, "transformer": 0.8, "bert": 0.7,
                    "xgboost": 0.7, "lightgbm": 0.7, "mlops": 0.8
                },
                "subcategories": ["Deep Learning", "Classical ML", "NLP", "Computer Vision", "MLOps"]
            },
            
            "Deep Learning": {
                "keywords": [
                    "deep learning", "neural networks", "cnn", "convolutional", "rnn", "lstm",
                    "gru", "transformer", "attention", "gan", "vae", "autoencoder",
                    "resnet", "vgg", "inception", "densenet", "mobilenet", "efficientnet",
                    "yolo", "rcnn", "unet", "bert", "gpt", "t5", "roberta",
                    "tensorflow", "pytorch", "keras", "jax", "flax", "huggingface"
                ],
                "weights": {
                    "deep learning": 1.0, "neural networks": 0.9, "cnn": 0.8,
                    "transformer": 0.9, "gan": 0.8, "bert": 0.8, "gpt": 0.8,
                    "pytorch": 0.9, "tensorflow": 0.9, "huggingface": 0.8
                },
                "subcategories": ["Computer Vision", "NLP", "Generative AI", "Model Architecture"]
            },
            
            "Data Science": {
                "keywords": [
                    "data science", "data analysis", "data mining", "statistics",
                    "pandas", "numpy", "scipy", "matplotlib", "seaborn", "plotly",
                    "jupyter", "r programming", "stata", "spss", "tableau", "power bi",
                    "data visualization", "exploratory data analysis", "eda",
                    "feature engineering", "data cleaning", "etl", "data pipeline",
                    "statistical modeling", "hypothesis testing", "regression", "classification"
                ],
                "weights": {
                    "data science": 1.0, "data analysis": 0.9, "pandas": 0.8,
                    "numpy": 0.8, "matplotlib": 0.7, "tableau": 0.8,
                    "data visualization": 0.8, "feature engineering": 0.8,
                    "statistical modeling": 0.7
                },
                "subcategories": ["Analytics", "Visualization", "Statistics", "Tools", "Pipeline"]
            },
            
            "Computer Vision": {
                "keywords": [
                    "computer vision", "cv", "opencv", "image processing", "video processing",
                    "object detection", "image segmentation", "facial recognition",
                    "yolo", "rcnn", "unet", "gan", "vae", "image classification",
                    "feature extraction", "edge detection", "contour detection",
                    "optical character recognition", "ocr", "image augmentation",
                    "pillow", "pil", "imageio", "scikit-image", "albumentations"
                ],
                "weights": {
                    "computer vision": 1.0, "opencv": 0.9, "object detection": 0.8,
                    "image processing": 0.8, "yolo": 0.7, "gan": 0.7,
                    "image segmentation": 0.8, "facial recognition": 0.7
                },
                "subcategories": ["Image Processing", "Object Detection", "Deep Learning CV", "OCR"]
            },
            
            "Natural Language Processing": {
                "keywords": [
                    "natural language processing", "nlp", "text mining", "sentiment analysis",
                    "named entity recognition", "ner", "text classification", "tokenization",
                    "stemming", "lemmatization", "tf-idf", "word2vec", "glove", "fasttext",
                    "bert", "gpt", "transformer", "huggingface", "spacy", "nltk",
                    "language models", "chatbots", "conversational ai", "text generation",
                    "machine translation", "question answering", "text summarization"
                ],
                "weights": {
                    "natural language processing": 1.0, "nlp": 1.0, "bert": 0.9,
                    "transformer": 0.9, "spacy": 0.8, "nltk": 0.8,
                    "sentiment analysis": 0.7, "chatbots": 0.7, "gpt": 0.9
                },
                "subcategories": ["Text Processing", "Language Models", "Text Analytics", "Conversational AI"]
            },
            
            "Reinforcement Learning": {
                "keywords": [
                    "reinforcement learning", "rl", "q-learning", "deep q-network", "dqn",
                    "policy gradient", "actor-critic", "ppo", "a3c", "ddpg", "td3",
                    "multi-agent", "openai gym", "stable baselines", "ray rllib",
                    "markov decision process", "mdp", "monte carlo", "temporal difference"
                ],
                "weights": {
                    "reinforcement learning": 1.0, "rl": 1.0, "q-learning": 0.8,
                    "dqn": 0.7, "policy gradient": 0.7, "openai gym": 0.6
                },
                "subcategories": ["Q-Learning", "Policy Gradient", "Multi-Agent", "Game AI"]
            },
            
            "Generative AI": {
                "keywords": [
                    "generative ai", "large language models", "llm", "gpt", "claude",
                    "palm", "llama", "mistral", "gemini", "prompt engineering",
                    "fine-tuning", "lora", "qlora", "rag", "retrieval augmented generation",
                    "langchain", "llamaindex", "vector databases", "embeddings",
                    "diffusion models", "stable diffusion", "dalle", "midjourney"
                ],
                "weights": {
                    "generative ai": 1.0, "large language models": 1.0, "llm": 1.0,
                    "gpt": 0.9, "prompt engineering": 0.8, "fine-tuning": 0.8,
                    "rag": 0.8, "langchain": 0.7, "vector databases": 0.7
                },
                "subcategories": ["Language Models", "Prompt Engineering", "Fine-tuning", "RAG"]
            },
            
            # CLOUD & INFRASTRUCTURE
            "Amazon Web Services": {
                "keywords": [
                    "aws", "amazon web services", "ec2", "s3", "rds", "lambda",
                    "cloudformation", "cloudwatch", "iam", "vpc", "elb", "auto scaling",
                    "eks", "ecs", "fargate", "api gateway", "dynamodb", "redshift",
                    "sagemaker", "glue", "kinesis", "sns", "sqs", "elasticache",
                    "route 53", "cloudfront", "elastic beanstalk", "emr", "athena",
                    "step functions", "eventbridge", "cognito", "secrets manager",
                    "parameter store", "codecommit", "codebuild", "codepipeline", "codedeploy"
                ],
                "weights": {
                    "aws": 1.0, "ec2": 0.8, "s3": 0.8, "lambda": 0.8,
                    "cloudformation": 0.7, "eks": 0.7, "sagemaker": 0.8,
                    "dynamodb": 0.7, "redshift": 0.6, "api gateway": 0.6
                },
                "subcategories": ["Compute", "Storage", "Database", "ML Services", "Networking", "Security"]
            },
            
            "Microsoft Azure": {
                "keywords": [
                    "azure", "microsoft azure", "azure functions", "azure sql",
                    "azure storage", "azure vm", "azure kubernetes", "aks",
                    "azure devops", "azure ml", "cognitive services", "power platform",
                    "azure active directory", "azure key vault", "azure monitor",
                    "application insights", "azure data factory", "azure synapse",
                    "azure cosmos db", "azure service bus", "azure logic apps"
                ],
                "weights": {
                    "azure": 1.0, "azure functions": 0.8, "aks": 0.7,
                    "azure ml": 0.8, "azure devops": 0.7, "cognitive services": 0.7,
                    "azure active directory": 0.6, "azure data factory": 0.6
                },
                "subcategories": ["Compute", "AI Services", "DevOps", "Database", "Analytics"]
            },
            
            "Google Cloud Platform": {
                "keywords": [
                    "gcp", "google cloud", "compute engine", "app engine", "cloud functions",
                    "gke", "kubernetes engine", "cloud storage", "bigquery", "cloud sql",
                    "cloud ml", "vertex ai", "cloud run", "firebase", "cloud firestore",
                    "cloud pub/sub", "cloud dataflow", "cloud dataproc", "cloud composer",
                    "cloud endpoints", "cloud armor", "cloud cdn", "cloud load balancing"
                ],
                "weights": {
                    "gcp": 1.0, "google cloud": 1.0, "bigquery": 0.8,
                    "gke": 0.7, "vertex ai": 0.8, "firebase": 0.7,
                    "cloud run": 0.6, "cloud functions": 0.7
                },
                "subcategories": ["Compute", "Data Analytics", "ML Platform", "Mobile", "Serverless"]
            },
            
            "Multi-Cloud & Hybrid": {
                "keywords": [
                    "multi-cloud", "hybrid cloud", "cloud migration", "cloud architecture",
                    "cloud strategy", "cloud governance", "cloud cost optimization",
                    "cloud security", "disaster recovery", "backup strategies",
                    "cloud automation", "infrastructure as code", "iac"
                ],
                "weights": {
                    "multi-cloud": 1.0, "hybrid cloud": 0.9, "cloud migration": 0.8,
                    "cloud architecture": 0.8, "infrastructure as code": 0.7,
                    "cloud security": 0.7
                },
                "subcategories": ["Architecture", "Migration", "Security", "Governance"]
            },
            
            "DevOps & CI/CD": {
                "keywords": [
                    "devops", "ci/cd", "continuous integration", "continuous deployment",
                    "jenkins", "gitlab ci", "github actions", "azure pipelines",
                    "terraform", "ansible", "puppet", "chef", "vagrant",
                    "monitoring", "logging", "prometheus", "grafana", "elk stack",
                    "circleci", "travis ci", "bamboo", "teamcity", "octopus deploy",
                    "infrastructure as code", "gitops", "argocd", "flux"
                ],
                "weights": {
                    "devops": 1.0, "ci/cd": 0.9, "jenkins": 0.8, "terraform": 0.8,
                    "ansible": 0.7, "prometheus": 0.7, "grafana": 0.7,
                    "github actions": 0.8, "gitlab ci": 0.7, "gitops": 0.6
                },
                "subcategories": ["Automation", "Monitoring", "Infrastructure", "Deployment", "GitOps"]
            },
            
            "Containerization & Orchestration": {
                "keywords": [
                    "docker", "containers", "kubernetes", "k8s", "helm", "istio",
                    "docker compose", "dockerfile", "container orchestration",
                    "pods", "services", "ingress", "configmap", "secrets",
                    "containerd", "cri-o", "podman", "buildah", "skopeo",
                    "openshift", "rancher", "nomad", "docker swarm"
                ],
                "weights": {
                    "docker": 1.0, "kubernetes": 1.0, "k8s": 1.0, "helm": 0.7,
                    "docker compose": 0.8, "istio": 0.6, "openshift": 0.7,
                    "rancher": 0.5, "podman": 0.5
                },
                "subcategories": ["Docker", "Kubernetes", "Orchestration", "Service Mesh", "Security"]
            },
            
            "Infrastructure as Code": {
                "keywords": [
                    "infrastructure as code", "iac", "terraform", "cloudformation",
                    "arm templates", "bicep", "ansible", "puppet", "chef",
                    "saltstack", "pulumi", "crossplane", "cdk", "serverless framework"
                ],
                "weights": {
                    "infrastructure as code": 1.0, "iac": 1.0, "terraform": 0.9,
                    "cloudformation": 0.8, "ansible": 0.8, "pulumi": 0.6,
                    "cdk": 0.6, "serverless framework": 0.5
                },
                "subcategories": ["Terraform", "CloudFormation", "Configuration Management", "Serverless"]
            },
            
            "Monitoring & Observability": {
                "keywords": [
                    "monitoring", "observability", "prometheus", "grafana", "elk stack",
                    "elasticsearch", "logstash", "kibana", "splunk", "datadog",
                    "new relic", "dynatrace", "appdynamics", "jaeger", "zipkin",
                    "opentelemetry", "metrics", "logging", "tracing", "alerting",
                    "sla", "slo", "sli", "incident management", "pagerduty"
                ],
                "weights": {
                    "monitoring": 1.0, "observability": 0.9, "prometheus": 0.8,
                    "grafana": 0.8, "elk stack": 0.8, "datadog": 0.7,
                    "jaeger": 0.6, "opentelemetry": 0.7, "incident management": 0.6
                },
                "subcategories": ["Metrics", "Logging", "Tracing", "Alerting", "Incident Response"]
            },
            
            # DATABASES & DATA ENGINEERING
            "SQL Databases": {
                "keywords": [
                    "sql", "mysql", "postgresql", "postgres", "oracle", "sql server",
                    "sqlite", "mariadb", "database design", "normalization",
                    "indexing", "query optimization", "stored procedures", "triggers",
                    "views", "materialized views", "partitioning", "replication",
                    "database administration", "dba", "backup", "recovery"
                ],
                "weights": {
                    "sql": 1.0, "mysql": 0.8, "postgresql": 0.8, "postgres": 0.8,
                    "oracle": 0.7, "sql server": 0.7, "database design": 0.8,
                    "query optimization": 0.7, "stored procedures": 0.6
                },
                "subcategories": ["Database Design", "Query Optimization", "Administration", "Performance"]
            },
            
            "NoSQL Databases": {
                "keywords": [
                    "nosql", "mongodb", "cassandra", "redis", "elasticsearch",
                    "dynamodb", "couchdb", "neo4j", "graph database", "document database",
                    "key-value store", "column family", "time series database",
                    "influxdb", "prometheus", "hbase", "couchbase", "riak",
                    "arangodb", "orientdb", "amazon documentdb"
                ],
                "weights": {
                    "nosql": 1.0, "mongodb": 0.9, "redis": 0.8, "elasticsearch": 0.8,
                    "cassandra": 0.7, "neo4j": 0.7, "dynamodb": 0.7,
                    "influxdb": 0.6, "couchdb": 0.5
                },
                "subcategories": ["Document", "Graph", "Key-Value", "Search", "Time Series"]
            },
            
            "Big Data Technologies": {
                "keywords": [
                    "big data", "hadoop", "hdfs", "mapreduce", "yarn", "hive",
                    "pig", "hbase", "spark", "scala", "pyspark", "kafka",
                    "storm", "flink", "beam", "dataflow", "databricks",
                    "snowflake", "redshift", "bigquery", "data lake", "data warehouse",
                    "parquet", "avro", "orc", "delta lake", "iceberg"
                ],
                "weights": {
                    "big data": 1.0, "hadoop": 0.8, "spark": 0.9, "kafka": 0.8,
                    "databricks": 0.7, "snowflake": 0.8, "flink": 0.7,
                    "data lake": 0.7, "data warehouse": 0.8
                },
                "subcategories": ["Hadoop Ecosystem", "Stream Processing", "Data Lakes", "Data Warehouses"]
            },
            
            "Data Engineering": {
                "keywords": [
                    "data engineering", "etl", "elt", "data pipeline", "data integration",
                    "airflow", "luigi", "prefect", "dagster", "nifi", "talend",
                    "informatica", "pentaho", "data quality", "data governance",
                    "data lineage", "data catalog", "metadata management",
                    "real-time processing", "batch processing", "stream processing"
                ],
                "weights": {
                    "data engineering": 1.0, "etl": 0.9, "data pipeline": 0.9,
                    "airflow": 0.8, "data quality": 0.7, "data governance": 0.7,
                    "stream processing": 0.8, "batch processing": 0.7
                },
                "subcategories": ["ETL/ELT", "Orchestration", "Data Quality", "Governance", "Real-time"]
            },
            
            "Data Analytics & BI": {
                "keywords": [
                    "data analytics", "business intelligence", "bi", "tableau", "power bi",
                    "qlik", "looker", "metabase", "superset", "grafana",
                    "data studio", "quicksight", "pentaho", "cognos",
                    "dax", "mdx", "olap", "data modeling", "dimensional modeling",
                    "star schema", "snowflake schema", "fact tables", "dimension tables"
                ],
                "weights": {
                    "data analytics": 1.0, "business intelligence": 0.9, "tableau": 0.8,
                    "power bi": 0.8, "looker": 0.7, "data modeling": 0.8,
                    "olap": 0.6, "dimensional modeling": 0.7
                },
                "subcategories": ["Visualization", "Modeling", "Reporting", "Self-Service BI"]
            },
            
            "Search & Information Retrieval": {
                "keywords": [
                    "elasticsearch", "solr", "lucene", "search engine", "full-text search",
                    "information retrieval", "indexing", "relevance scoring",
                    "search analytics", "faceted search", "autocomplete",
                    "spell checking", "synonyms", "stemming", "lemmatization",
                    "opensearch", "algolia", "swiftype"
                ],
                "weights": {
                    "elasticsearch": 1.0, "solr": 0.8, "search engine": 0.8,
                    "full-text search": 0.7, "information retrieval": 0.7,
                    "relevance scoring": 0.6, "search analytics": 0.6
                },
                "subcategories": ["Full-text Search", "Analytics", "Relevance", "Performance"]
            },
            
            # WEB DEVELOPMENT & FRONTEND
            "Frontend Development": {
                "keywords": [
                    "frontend", "front-end", "html", "css", "sass", "scss", "less",
                    "bootstrap", "tailwind", "material ui", "ant design", "chakra ui",
                    "responsive design", "mobile first", "progressive web app", "pwa",
                    "spa", "single page application", "webpack", "parcel", "vite",
                    "rollup", "esbuild", "swc", "babel", "postcss", "autoprefixer",
                    "css grid", "flexbox", "css modules", "styled components", "emotion"
                ],
                "weights": {
                    "frontend": 1.0, "html": 0.8, "css": 0.8, "sass": 0.7,
                    "bootstrap": 0.7, "tailwind": 0.7, "responsive design": 0.8,
                    "pwa": 0.6, "webpack": 0.7, "vite": 0.6
                },
                "subcategories": ["Styling", "Frameworks", "Build Tools", "Architecture", "Performance"]
            },
            
            "React Ecosystem": {
                "keywords": [
                    "react", "reactjs", "jsx", "tsx", "hooks", "context", "redux",
                    "mobx", "zustand", "react router", "react query", "apollo client",
                    "next.js", "gatsby", "create react app", "react testing library",
                    "enzyme", "storybook", "react native", "expo", "react helmet"
                ],
                "weights": {
                    "react": 1.0, "reactjs": 1.0, "hooks": 0.8, "redux": 0.8,
                    "next.js": 0.9, "react router": 0.7, "react query": 0.6,
                    "react native": 0.8, "gatsby": 0.6
                },
                "subcategories": ["Core React", "State Management", "Routing", "SSR/SSG", "Mobile"]
            },
            
            "Vue.js Ecosystem": {
                "keywords": [
                    "vue", "vuejs", "vue 3", "composition api", "options api", "vuex",
                    "pinia", "vue router", "nuxt", "nuxt.js", "vuepress", "quasar",
                    "vuetify", "vue cli", "vite vue", "vue test utils"
                ],
                "weights": {
                    "vue": 1.0, "vuejs": 1.0, "vue 3": 0.9, "composition api": 0.8,
                    "vuex": 0.7, "pinia": 0.7, "nuxt": 0.8, "vuetify": 0.6
                },
                "subcategories": ["Core Vue", "State Management", "SSR", "UI Libraries", "Testing"]
            },
            
            "Angular Ecosystem": {
                "keywords": [
                    "angular", "angularjs", "typescript", "rxjs", "observables",
                    "angular cli", "angular material", "ngrx", "angular router",
                    "angular forms", "reactive forms", "template driven forms",
                    "angular universal", "ionic angular", "nx workspace"
                ],
                "weights": {
                    "angular": 1.0, "angularjs": 0.6, "rxjs": 0.8, "ngrx": 0.7,
                    "angular cli": 0.6, "angular material": 0.6, "ionic angular": 0.5
                },
                "subcategories": ["Core Angular", "State Management", "Forms", "SSR", "Mobile"]
            },
            
            "UI/UX Design": {
                "keywords": [
                    "ui design", "ux design", "user interface", "user experience",
                    "figma", "sketch", "adobe xd", "invision", "principle",
                    "wireframing", "prototyping", "user research", "usability testing",
                    "accessibility", "a11y", "wcag", "design systems", "style guides",
                    "interaction design", "visual design", "typography", "color theory"
                ],
                "weights": {
                    "ui design": 1.0, "ux design": 1.0, "figma": 0.8, "sketch": 0.7,
                    "prototyping": 0.7, "accessibility": 0.8, "design systems": 0.8,
                    "usability testing": 0.7, "wireframing": 0.6
                },
                "subcategories": ["Design Tools", "Prototyping", "Research", "Accessibility", "Systems"]
            },
            
            "Mobile Development": {
                "keywords": [
                    "mobile development", "android", "ios", "flutter", "dart",
                    "react native", "xamarin", "ionic", "cordova", "swift",
                    "kotlin", "objective-c", "java android", "mobile ui", "mobile ux",
                    "app store optimization", "aso", "push notifications", "in-app purchases",
                    "mobile analytics", "firebase", "expo", "capacitor"
                ],
                "weights": {
                    "mobile development": 1.0, "android": 0.9, "ios": 0.9,
                    "flutter": 0.8, "react native": 0.8, "swift": 0.8, "kotlin": 0.8,
                    "firebase": 0.7, "expo": 0.6
                },
                "subcategories": ["Native", "Cross-Platform", "UI/UX", "Performance", "Analytics"]
            },
            
            # BACKEND & ARCHITECTURE
            "Backend Development": {
                "keywords": [
                    "backend", "back-end", "server-side", "api development",
                    "rest api", "restful", "graphql", "grpc", "microservices",
                    "monolith", "serverless", "lambda", "functions",
                    "authentication", "authorization", "jwt", "oauth", "saml",
                    "rate limiting", "caching", "session management", "cors"
                ],
                "weights": {
                    "backend": 1.0, "api development": 0.9, "rest api": 0.8,
                    "graphql": 0.7, "microservices": 0.8, "serverless": 0.7,
                    "authentication": 0.7, "jwt": 0.6, "oauth": 0.6
                },
                "subcategories": ["API Design", "Architecture", "Security", "Performance", "Authentication"]
            },
            
            "Microservices Architecture": {
                "keywords": [
                    "microservices", "service oriented architecture", "soa", "api gateway",
                    "service mesh", "istio", "linkerd", "consul", "service discovery",
                    "circuit breaker", "bulkhead pattern", "saga pattern",
                    "event sourcing", "cqrs", "distributed tracing", "service registry"
                ],
                "weights": {
                    "microservices": 1.0, "service mesh": 0.8, "api gateway": 0.8,
                    "service discovery": 0.7, "circuit breaker": 0.6, "event sourcing": 0.7,
                    "cqrs": 0.6, "istio": 0.7
                },
                "subcategories": ["Architecture Patterns", "Service Mesh", "Resilience", "Communication"]
            },
            
            "System Architecture": {
                "keywords": [
                    "system architecture", "software architecture", "distributed systems",
                    "scalability", "high availability", "load balancing", "caching",
                    "message queues", "event driven", "publish subscribe",
                    "design patterns", "solid principles", "clean architecture",
                    "hexagonal architecture", "onion architecture", "event storming",
                    "domain driven design", "ddd", "twelve-factor app"
                ],
                "weights": {
                    "system architecture": 1.0, "distributed systems": 0.9,
                    "scalability": 0.8, "load balancing": 0.7, "caching": 0.7,
                    "design patterns": 0.8, "clean architecture": 0.7,
                    "domain driven design": 0.8, "ddd": 0.8
                },
                "subcategories": ["Design Patterns", "Scalability", "Performance", "Reliability", "DDD"]
            },
            
            "Message Queues & Event Streaming": {
                "keywords": [
                    "message queues", "rabbitmq", "kafka", "apache kafka", "pulsar",
                    "activemq", "amazon sqs", "azure service bus", "google pub/sub",
                    "event streaming", "event sourcing", "event driven architecture",
                    "publish subscribe", "message brokers", "dead letter queues",
                    "kafka streams", "ksqldb", "confluent", "schema registry"
                ],
                "weights": {
                    "message queues": 1.0, "kafka": 0.9, "rabbitmq": 0.8,
                    "event streaming": 0.8, "event driven architecture": 0.8,
                    "publish subscribe": 0.7, "kafka streams": 0.6
                },
                "subcategories": ["Message Brokers", "Event Streaming", "Architecture", "Stream Processing"]
            },
            
            # SECURITY & COMPLIANCE
            "Cybersecurity": {
                "keywords": [
                    "cybersecurity", "security", "penetration testing", "ethical hacking",
                    "vulnerability assessment", "security audit", "encryption",
                    "ssl", "tls", "https", "firewall", "ids", "ips",
                    "owasp", "sql injection", "xss", "csrf", "security compliance",
                    "iso 27001", "nist", "gdpr", "hipaa", "sox", "pci dss",
                    "threat modeling", "risk assessment", "incident response"
                ],
                "weights": {
                    "cybersecurity": 1.0, "penetration testing": 0.8, "encryption": 0.8,
                    "owasp": 0.7, "security audit": 0.7, "ssl": 0.6,
                    "vulnerability assessment": 0.8, "threat modeling": 0.7
                },
                "subcategories": ["Penetration Testing", "Compliance", "Encryption", "Web Security", "Risk Management"]
            },
            
            "Cloud Security": {
                "keywords": [
                    "cloud security", "aws security", "azure security", "gcp security",
                    "iam", "identity management", "access control", "zero trust",
                    "secrets management", "key management", "encryption at rest",
                    "encryption in transit", "network security", "security groups",
                    "nacl", "waf", "ddos protection", "security monitoring"
                ],
                "weights": {
                    "cloud security": 1.0, "iam": 0.8, "zero trust": 0.7,
                    "secrets management": 0.7, "encryption at rest": 0.6,
                    "network security": 0.7, "waf": 0.6
                },
                "subcategories": ["Identity Management", "Network Security", "Data Protection", "Monitoring"]
            },
            
            "Application Security": {
                "keywords": [
                    "application security", "secure coding", "code review", "sast", "dast",
                    "static analysis", "dynamic analysis", "dependency scanning",
                    "container security", "devsecops", "security testing",
                    "threat modeling", "security by design", "secure sdlc"
                ],
                "weights": {
                    "application security": 1.0, "secure coding": 0.8, "sast": 0.7,
                    "dast": 0.7, "devsecops": 0.8, "threat modeling": 0.7,
                    "secure sdlc": 0.6
                },
                "subcategories": ["Secure Coding", "Testing", "DevSecOps", "Architecture"]
            },
            
            # TESTING & QUALITY ASSURANCE
            "Software Testing": {
                "keywords": [
                    "testing", "quality assurance", "qa", "unit testing", "integration testing",
                    "system testing", "acceptance testing", "test automation",
                    "selenium", "cypress", "playwright", "jest", "mocha", "junit", "pytest",
                    "test driven development", "tdd", "behavior driven development", "bdd",
                    "cucumber", "gherkin", "testng", "rspec", "jasmine"
                ],
                "weights": {
                    "testing": 1.0, "test automation": 0.9, "selenium": 0.8,
                    "unit testing": 0.8, "tdd": 0.7, "cypress": 0.7,
                    "playwright": 0.7, "pytest": 0.6, "jest": 0.6
                },
                "subcategories": ["Automation", "Manual Testing", "TDD/BDD", "Tools", "Frameworks"]
            },
            
            "Performance Testing": {
                "keywords": [
                    "performance testing", "load testing", "stress testing", "volume testing",
                    "endurance testing", "jmeter", "gatling", "locust", "k6",
                    "performance monitoring", "apm", "profiling", "benchmarking",
                    "scalability testing", "capacity planning"
                ],
                "weights": {
                    "performance testing": 1.0, "load testing": 0.9, "jmeter": 0.8,
                    "gatling": 0.7, "stress testing": 0.8, "k6": 0.6,
                    "scalability testing": 0.7
                },
                "subcategories": ["Load Testing", "Tools", "Monitoring", "Analysis"]
            },
            
            "Test Management": {
                "keywords": [
                    "test management", "test planning", "test strategy", "test cases",
                    "test execution", "defect tracking", "test metrics", "test reporting",
                    "jira", "testlink", "testrail", "zephyr", "qtest", "xray"
                ],
                "weights": {
                    "test management": 1.0, "test planning": 0.8, "test strategy": 0.8,
                    "defect tracking": 0.7, "test metrics": 0.6, "jira": 0.6
                },
                "subcategories": ["Planning", "Execution", "Reporting", "Tools"]
            },
            
            # SPECIALIZED DOMAINS
            "Game Development": {
                "keywords": [
                    "game development", "unity", "unreal engine", "godot", "gamemaker",
                    "c# unity", "c++ unreal", "game design", "game programming",
                    "3d graphics", "2d graphics", "shaders", "physics simulation",
                    "game ai", "multiplayer", "networking", "optimization",
                    "mobile games", "console games", "pc games", "vr games", "ar games"
                ],
                "weights": {
                    "game development": 1.0, "unity": 0.9, "unreal engine": 0.9,
                    "game design": 0.8, "3d graphics": 0.7, "shaders": 0.6,
                    "game ai": 0.7, "multiplayer": 0.6
                },
                "subcategories": ["Engines", "Graphics", "AI", "Networking", "Platforms"]
            },
            
            "Blockchain & Web3": {
                "keywords": [
                    "blockchain", "web3", "cryptocurrency", "smart contracts", "solidity",
                    "ethereum", "bitcoin", "defi", "nft", "dao", "dapp",
                    "web3.js", "ethers.js", "truffle", "hardhat", "metamask",
                    "ipfs", "polygon", "binance smart chain", "avalanche", "cardano"
                ],
                "weights": {
                    "blockchain": 1.0, "web3": 0.9, "smart contracts": 0.9,
                    "solidity": 0.8, "ethereum": 0.8, "defi": 0.7,
                    "nft": 0.6, "dapp": 0.7, "web3.js": 0.6
                },
                "subcategories": ["Smart Contracts", "DeFi", "NFTs", "Tools", "Platforms"]
            },
            
            "IoT & Embedded Systems": {
                "keywords": [
                    "iot", "internet of things", "embedded systems", "arduino", "raspberry pi",
                    "esp32", "esp8266", "stm32", "pic microcontroller", "arm cortex",
                    "real-time systems", "rtos", "freertos", "embedded c", "embedded linux",
                    "sensors", "actuators", "wireless communication", "bluetooth", "wifi",
                    "zigbee", "lora", "mqtt", "coap", "edge computing"
                ],
                "weights": {
                    "iot": 1.0, "embedded systems": 1.0, "arduino": 0.7,
                    "raspberry pi": 0.7, "esp32": 0.6, "rtos": 0.8,
                    "embedded c": 0.8, "mqtt": 0.6, "edge computing": 0.7
                },
                "subcategories": ["Microcontrollers", "Communication", "RTOS", "Sensors", "Edge Computing"]
            },
            
            "Robotics & Automation": {
                "keywords": [
                    "robotics", "automation", "ros", "robot operating system", "gazebo",
                    "computer vision robotics", "slam", "navigation", "path planning",
                    "machine learning robotics", "ai robotics", "industrial automation",
                    "plc programming", "scada", "hmi", "servo motors", "stepper motors"
                ],
                "weights": {
                    "robotics": 1.0, "automation": 0.9, "ros": 0.8,
                    "slam": 0.7, "path planning": 0.6, "plc programming": 0.7,
                    "industrial automation": 0.8
                },
                "subcategories": ["ROS", "Navigation", "Industrial", "AI Robotics", "Hardware"]
            },
            
            "Fintech & Finance": {
                "keywords": [
                    "fintech", "financial technology", "algorithmic trading", "quantitative finance",
                    "risk management", "compliance", "kyc", "aml", "payment processing",
                    "digital banking", "robo advisory", "credit scoring", "fraud detection",
                    "regulatory technology", "regtech", "open banking", "api banking"
                ],
                "weights": {
                    "fintech": 1.0, "algorithmic trading": 0.8, "quantitative finance": 0.8,
                    "risk management": 0.8, "payment processing": 0.7,
                    "fraud detection": 0.7, "regtech": 0.6
                },
                "subcategories": ["Trading", "Risk Management", "Payments", "Compliance", "Banking"]
            },
            
            "Healthcare & Biotech": {
                "keywords": [
                    "healthcare technology", "health tech", "medical devices", "telemedicine",
                    "electronic health records", "ehr", "fhir", "hl7", "dicom",
                    "medical imaging", "bioinformatics", "genomics", "proteomics",
                    "clinical trials", "pharmaceutical", "drug discovery", "precision medicine"
                ],
                "weights": {
                    "healthcare technology": 1.0, "telemedicine": 0.8, "ehr": 0.8,
                    "medical imaging": 0.8, "bioinformatics": 0.9, "genomics": 0.8,
                    "drug discovery": 0.7, "precision medicine": 0.7
                },
                "subcategories": ["Medical Devices", "Healthcare IT", "Bioinformatics", "Imaging", "Clinical"]
            },
            
            # PROJECT MANAGEMENT & SOFT SKILLS
            "Project Management": {
                "keywords": [
                    "project management", "agile", "scrum", "kanban", "waterfall",
                    "jira", "confluence", "trello", "asana", "monday.com", "notion",
                    "sprint planning", "retrospectives", "stand-ups", "product owner",
                    "scrum master", "stakeholder management", "risk management",
                    "project planning", "resource allocation", "timeline management",
                    "pmp", "prince2", "certified scrum master", "csm", "safe"
                ],
                "weights": {
                    "project management": 1.0, "agile": 0.9, "scrum": 0.8,
                    "kanban": 0.7, "jira": 0.7, "sprint planning": 0.6,
                    "pmp": 0.7, "scrum master": 0.8, "stakeholder management": 0.7
                },
                "subcategories": ["Agile", "Tools", "Leadership", "Planning", "Certification"]
            },
            
            "Technical Leadership": {
                "keywords": [
                    "technical leadership", "team lead", "tech lead", "engineering manager",
                    "architect", "principal engineer", "staff engineer", "senior engineer",
                    "mentoring", "coaching", "code review", "technical strategy",
                    "technology roadmap", "technical debt", "performance optimization",
                    "team building", "hiring", "interviews", "onboarding"
                ],
                "weights": {
                    "technical leadership": 1.0, "team lead": 0.9, "tech lead": 0.9,
                    "engineering manager": 0.9, "architect": 0.8, "mentoring": 0.8,
                    "code review": 0.7, "technical strategy": 0.8
                },
                "subcategories": ["Leadership", "Mentoring", "Strategy", "Team Management", "Architecture"]
            },
            
            "Communication & Collaboration": {
                "keywords": [
                    "communication", "presentation", "documentation", "technical writing",
                    "knowledge sharing", "training", "workshops", "public speaking",
                    "cross-functional collaboration", "stakeholder communication",
                    "client communication", "remote work", "distributed teams",
                    "confluence", "wiki", "slack", "microsoft teams", "zoom"
                ],
                "weights": {
                    "communication": 1.0, "technical writing": 0.8, "presentation": 0.7,
                    "knowledge sharing": 0.7, "public speaking": 0.6,
                    "cross-functional collaboration": 0.8, "documentation": 0.7
                },
                "subcategories": ["Technical Writing", "Presentation", "Collaboration", "Remote Work"]
            },
            
            "Product Management": {
                "keywords": [
                    "product management", "product strategy", "product roadmap", "user stories",
                    "product requirements", "market research", "competitive analysis",
                    "user research", "product analytics", "a/b testing", "mvp",
                    "product launch", "go-to-market", "product metrics", "kpis",
                    "product owner", "business analysis", "requirements gathering"
                ],
                "weights": {
                    "product management": 1.0, "product strategy": 0.9, "user stories": 0.7,
                    "market research": 0.7, "a/b testing": 0.6, "mvp": 0.7,
                    "product owner": 0.8, "business analysis": 0.7
                },
                "subcategories": ["Strategy", "Research", "Analytics", "Requirements", "Launch"]
            },
            
            "Business & Strategy": {
                "keywords": [
                    "business strategy", "digital transformation", "innovation", "entrepreneurship",
                    "startup", "venture capital", "business development", "partnerships",
                    "market analysis", "business model", "revenue optimization",
                    "cost optimization", "roi analysis", "business intelligence",
                    "consulting", "change management", "process improvement"
                ],
                "weights": {
                    "business strategy": 1.0, "digital transformation": 0.8, "innovation": 0.7,
                    "entrepreneurship": 0.7, "startup": 0.6, "business development": 0.8,
                    "consulting": 0.7, "process improvement": 0.7
                },
                "subcategories": ["Strategy", "Innovation", "Development", "Analysis", "Transformation"]
            },
            
            "Sales & Marketing": {
                "keywords": [
                    "sales", "marketing", "digital marketing", "content marketing", "seo",
                    "sem", "social media marketing", "email marketing", "crm",
                    "lead generation", "conversion optimization", "sales funnel",
                    "customer acquisition", "customer retention", "salesforce",
                    "hubspot", "google analytics", "marketing automation"
                ],
                "weights": {
                    "sales": 1.0, "marketing": 1.0, "digital marketing": 0.8,
                    "seo": 0.7, "crm": 0.7, "lead generation": 0.6,
                    "salesforce": 0.6, "google analytics": 0.6
                },
                "subcategories": ["Digital Marketing", "CRM", "Analytics", "Automation", "Strategy"]
            }
        }
    
    def analyze_resume(self, resume_text: str) -> Dict[str, Any]:
        """
        Analyze resume and return comprehensive skill confidence scores
        """
        if not resume_text or len(resume_text.strip()) < 10:
            return self._empty_analysis("Resume text is empty or too short")
        
        resume_lower = resume_text.lower()
        results = {}
        
        # First pass: normal detection with lower threshold
        for category, data in self.skill_categories.items():
            score_data = self._calculate_category_score(resume_lower, data)
            if score_data['confidence'] > 0.05:  # Lowered threshold from 0.1 to 0.05
                results[category] = score_data
        
        # If no skills found, try aggressive detection
        if not results:
            print("⚠️ No skills found with normal detection, trying aggressive detection...")
            results = self._aggressive_skill_detection(resume_lower)
        
        # Sort by confidence score
        results = dict(sorted(results.items(), key=lambda x: x[1]['confidence'], reverse=True))
        
        return {
            'skill_scores': results,
            'top_skills': list(results.keys())[:10],
            'total_categories_detected': len(results),
            'overall_technical_level': self._determine_overall_level(results),
            'summary': self._generate_skill_summary(results),
            'debug_info': {
                'text_length': len(resume_text),
                'has_content': bool(resume_text.strip()),
                'detection_method': 'aggressive' if not results else 'normal'
            }
        }
    
    def _calculate_category_score(self, resume_text: str, data: Dict) -> Dict[str, Any]:
        """Calculate confidence score for a specific skill category"""
        keywords = data['keywords']
        weights = data.get('weights', {})
        subcategories = data.get('subcategories', [])
        
        found_keywords = []
        total_score = 0.0
        evidence = []
        
        for keyword in keywords:
            if keyword in resume_text:
                found_keywords.append(keyword)
                weight = weights.get(keyword, 0.5)  # Default weight
                total_score += weight
                evidence.append(keyword)
        
        # Calculate experience boost
        experience_multiplier = self._calculate_experience_multiplier(resume_text)
        
        # Calculate confidence (normalize by number of possible keywords)
        max_possible_score = sum(weights.get(kw, 0.5) for kw in keywords)
        base_confidence = min(total_score / max_possible_score, 1.0) if max_possible_score > 0 else 0.0
        
        # Apply experience multiplier
        final_confidence = min(base_confidence * experience_multiplier, 1.0)
        
        # Determine experience level
        experience_level = self._determine_experience_level(resume_text, evidence, final_confidence)
        
        return {
            'confidence': round(final_confidence, 3),
            'evidence': evidence,
            'keyword_count': len(found_keywords),
            'experience_level': experience_level,
            'subcategories': subcategories,
            'experience_multiplier': round(experience_multiplier, 2)
        }
    
    def _calculate_experience_multiplier(self, resume_text: str) -> float:
        """Calculate experience multiplier based on years, projects, and level indicators"""
        multiplier = 1.0
        
        # Look for years of experience
        years_pattern = self.experience_keywords['years']
        years_matches = re.findall(years_pattern, resume_text, re.IGNORECASE)
        if years_matches:
            max_years = max(int(year) for year in years_matches)
            if max_years >= 5:
                multiplier += 0.3
            elif max_years >= 3:
                multiplier += 0.2
            elif max_years >= 1:
                multiplier += 0.1
        
        # Look for project count
        projects_pattern = self.experience_keywords['projects']
        project_matches = re.findall(projects_pattern, resume_text, re.IGNORECASE)
        if project_matches:
            max_projects = max(int(proj) for proj in project_matches)
            if max_projects >= 5:
                multiplier += 0.2
            elif max_projects >= 3:
                multiplier += 0.1
        
        # Look for level indicators
        for level, indicators in self.experience_keywords['level_indicators'].items():
            for indicator in indicators:
                if indicator in resume_text:
                    if level == 'senior':
                        multiplier += 0.3
                    elif level == 'mid':
                        multiplier += 0.1
                    # Junior indicators don't add multiplier
                    break
        
        return min(multiplier, 2.0)  # Cap at 2x
    
    def _determine_experience_level(self, resume_text: str, evidence: List[str], confidence: float) -> str:
        """Determine experience level based on evidence and confidence"""
        # Check for explicit level indicators
        for level, indicators in self.experience_keywords['level_indicators'].items():
            for indicator in indicators:
                if indicator in resume_text:
                    return level.title()
        
        # Infer from confidence and evidence count
        if confidence >= 0.8 and len(evidence) >= 5:
            return "Senior"
        elif confidence >= 0.6 and len(evidence) >= 3:
            return "Mid-level"
        elif confidence >= 0.3:
            return "Junior"
        else:
            return "Beginner"
    
    def _determine_overall_level(self, results: Dict) -> str:
        """Determine overall technical level"""
        if not results:
            return "Entry Level"
        
        high_confidence_skills = sum(1 for data in results.values() if data['confidence'] >= 0.7)
        total_skills = len(results)
        
        if high_confidence_skills >= 5 and total_skills >= 10:
            return "Senior"
        elif high_confidence_skills >= 3 and total_skills >= 6:
            return "Mid-level"
        elif total_skills >= 3:
            return "Junior"
        else:
            return "Entry Level"
    
    def _generate_skill_summary(self, results: Dict) -> str:
        """Generate a human-readable summary of skills"""
        if not results:
            return "No significant technical skills detected."
        
        top_skills = list(results.keys())[:5]
        skill_count = len(results)
        
        summary = f"Detected {skill_count} technical skill categories. "
        summary += f"Strongest areas: {', '.join(top_skills[:3])}. "
        
        # Count by domain
        domains = {
            'Programming Languages': [
                'Python Programming', 'JavaScript Programming', 'Java Programming', 
                'C/C++ Programming', 'C# Programming', 'Go Programming', 'Rust Programming',
                'PHP Programming', 'Ruby Programming', 'Swift Programming', 'Kotlin Programming',
                'Scala Programming', 'R Programming'
            ],
            'AI/ML & Data Science': [
                'Machine Learning', 'Deep Learning', 'Data Science', 'Computer Vision', 
                'Natural Language Processing', 'Reinforcement Learning', 'Generative AI',
                'Big Data Technologies', 'Data Engineering', 'Data Analytics & BI'
            ],
            'Cloud & Infrastructure': [
                'Amazon Web Services', 'Microsoft Azure', 'Google Cloud Platform',
                'Multi-Cloud & Hybrid', 'DevOps & CI/CD', 'Containerization & Orchestration',
                'Infrastructure as Code', 'Monitoring & Observability'
            ],
            'Database & Storage': [
                'SQL Databases', 'NoSQL Databases', 'Search & Information Retrieval'
            ],
            'Web & Mobile Development': [
                'Frontend Development', 'React Ecosystem', 'Vue.js Ecosystem', 'Angular Ecosystem',
                'UI/UX Design', 'Mobile Development', 'Backend Development'
            ],
            'Architecture & Systems': [
                'Microservices Architecture', 'System Architecture', 'Message Queues & Event Streaming'
            ],
            'Security & Testing': [
                'Cybersecurity', 'Cloud Security', 'Application Security', 'Software Testing',
                'Performance Testing', 'Test Management'
            ],
            'Specialized Domains': [
                'Game Development', 'Blockchain & Web3', 'IoT & Embedded Systems',
                'Robotics & Automation', 'Fintech & Finance', 'Healthcare & Biotech'
            ],
            'Business & Management': [
                'Project Management', 'Technical Leadership', 'Communication & Collaboration',
                'Product Management', 'Business & Strategy', 'Sales & Marketing'
            ]
        }
        
        domain_counts = {}
        for domain, categories in domains.items():
            count = sum(1 for cat in categories if cat in results)
            if count > 0:
                domain_counts[domain] = count
        
        if domain_counts:
            summary += f"Domain expertise in: {', '.join(domain_counts.keys())}."
        
        return summary
    
    def _empty_analysis(self, reason: str = "No skills detected") -> Dict[str, Any]:
        """Return empty analysis result"""
        return {
            'skill_scores': {},
            'top_skills': [],
            'total_categories_detected': 0,
            'overall_technical_level': 'Entry Level',
            'summary': f'No technical skills detected. {reason}',
            'debug_info': {
                'reason': reason,
                'detection_method': 'none'
            }
        }
    
    def _aggressive_skill_detection(self, resume_text: str) -> Dict[str, Any]:
        """More aggressive skill detection for hard-to-parse resumes"""
        results = {}
        
        # Common programming terms that might indicate technical skills
        basic_tech_indicators = {
            'programming': ['programming', 'coding', 'development', 'software', 'developer', 'engineer'],
            'web': ['web', 'website', 'html', 'css', 'frontend', 'backend'],
            'database': ['database', 'sql', 'data', 'mysql', 'postgres'],
            'languages': ['python', 'java', 'javascript', 'c++', 'c#', 'php', 'ruby'],
            'frameworks': ['react', 'angular', 'vue', 'django', 'flask', 'spring'],
            'tools': ['git', 'docker', 'linux', 'aws', 'azure', 'jenkins']
        }
        
        for category, keywords in basic_tech_indicators.items():
            found_count = sum(1 for keyword in keywords if keyword in resume_text)
            if found_count > 0:
                confidence = min(found_count * 0.1, 0.8)  # Max 0.8 confidence
                results[f"General {category.title()}"] = {
                    'confidence': confidence,
                    'evidence': [kw for kw in keywords if kw in resume_text][:5],
                    'experience_level': 'Mid-level' if confidence > 0.3 else 'Entry Level',
                    'subcategories': ['General']
                }
        
        return results
