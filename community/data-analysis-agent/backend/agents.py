# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import re
import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from openai import OpenAI
from typing import List, Any, Optional, Dict, Tuple
import warnings
import base64
warnings.filterwarnings('ignore')

# Try to load from .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, continue without it

# Configuration
API_BASE_URL = "https://integrate.api.nvidia.com/v1"
API_KEY = os.environ.get("NVIDIA_API_KEY")

DEFAULT_FIGSIZE = (6, 4)
DEFAULT_DPI = 100
MAX_RESULT_DISPLAY_LENGTH = 300

class ModelConfig:
    """Configuration class for different models."""

    def __init__(self, model_name: str, model_url: str, model_print_name: str,
                 query_understanding_temperature: float = 0.1,
                 query_understanding_max_tokens: int = 5,
                 code_generation_temperature: float = 0.2,
                 code_generation_max_tokens: int = 1024,
                 reasoning_temperature: float = 0.2,
                 reasoning_max_tokens: int = 1024,
                 insights_temperature: float = 0.2,
                 insights_max_tokens: int = 512,
                 reasoning_false: str = "detailed thinking off",
                 reasoning_true: str = "detailed thinking on"):
        self.MODEL_NAME = model_name
        self.MODEL_URL = model_url
        self.MODEL_PRINT_NAME = model_print_name
        self.QUERY_UNDERSTANDING_TEMPERATURE = query_understanding_temperature
        self.QUERY_UNDERSTANDING_MAX_TOKENS = query_understanding_max_tokens
        self.CODE_GENERATION_TEMPERATURE = code_generation_temperature
        self.CODE_GENERATION_MAX_TOKENS = code_generation_max_tokens
        self.REASONING_TEMPERATURE = reasoning_temperature
        self.REASONING_MAX_TOKENS = reasoning_max_tokens
        self.INSIGHTS_TEMPERATURE = insights_temperature
        self.INSIGHTS_MAX_TOKENS = insights_max_tokens
        self.REASONING_FALSE = reasoning_false
        self.REASONING_TRUE = reasoning_true

MODEL_CONFIGS = {
    "llama-3-1-nemotron-ultra-v1": ModelConfig(
        model_name="nvidia/llama-3.1-nemotron-ultra-253b-v1",
        model_url="https://build.nvidia.com/nvidia/llama-3_1-nemotron-ultra-253b-v1",
        model_print_name="NVIDIA Llama 3.1 Nemotron Ultra 253B v1",
        query_understanding_temperature=0.1,
        query_understanding_max_tokens=5,
        code_generation_temperature=0.2,
        code_generation_max_tokens=1024,
        reasoning_temperature=0.6,
        reasoning_max_tokens=1024,
        insights_temperature=0.2,
        insights_max_tokens=512,
        reasoning_false="detailed thinking off",
        reasoning_true="detailed thinking on"
    ),
    "llama-3-3-nemotron-super-v1-5": ModelConfig(
        model_name="nvidia/llama-3.3-nemotron-super-49b-v1.5",
        model_url="https://build.nvidia.com/nvidia/llama-3_3-nemotron-super-49b-v1_5",
        model_print_name="NVIDIA Llama 3.3 Nemotron Super 49B v1.5",
        query_understanding_temperature=0.1,
        query_understanding_max_tokens=5,
        code_generation_temperature=0.0,
        code_generation_max_tokens=1024,
        reasoning_temperature=0.6,
        reasoning_max_tokens=2048,
        insights_temperature=0.2,
        insights_max_tokens=512,
        reasoning_false="/no_think",
        reasoning_true=""
    )
}

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

class DataAnalysisAgents:
    """Main class containing all data analysis agents."""

    def __init__(self, model_key: str = "llama-3-1-nemotron-ultra-v1"):
        self.config = MODEL_CONFIGS.get(model_key, MODEL_CONFIGS["llama-3-1-nemotron-ultra-v1"])

    def query_understanding_tool(self, query: str) -> bool:
        """Determine if query requests visualization."""
        full_prompt = f"""You are a query classifier. Your task is to determine if a user query is requesting a data visualization.

IMPORTANT: Respond with ONLY 'true' or 'false' (lowercase, no quotes, no punctuation).

Classify as 'true' ONLY if the query explicitly asks for:
- A plot, chart, graph, visualization, or figure
- To "show" or "display" data visually
- To "create" or "generate" a visual representation

Classify as 'false' for:
- Data analysis without visualization requests
- Statistical calculations, aggregations, filtering, sorting
- Questions about data content, counts, summaries

User query: {query}"""

        messages = [
            {"role": "system", "content": self.config.REASONING_FALSE},
            {"role": "user", "content": full_prompt}
        ]

        response = client.chat.completions.create(
            model=self.config.MODEL_NAME,
            messages=messages,
            temperature=self.config.QUERY_UNDERSTANDING_TEMPERATURE,
            max_tokens=self.config.QUERY_UNDERSTANDING_MAX_TOKENS
        )

        intent_response = response.choices[0].message.content.strip().lower()
        return intent_response == "true"

    def code_writing_tool(self, cols: List[str], query: str) -> str:
        """Generate prompt for pandas-only code."""
        return f"""
Given DataFrame `df` with columns: {', '.join(cols)}

Write Python code (pandas **only**, no plotting) to answer: "{query}"

Rules:
1. Use pandas operations on `df` only.
2. Assign the final result to `result`.
3. Return your answer inside a single markdown fence that starts with ```python and ends with ```.
4. Do not include any explanations outside the code block.
5. Handle missing values (`dropna`) before aggregations.

Example:
```python
result = df.groupby("some_column")["a_numeric_col"].mean().sort_values(ascending=False)
```
"""

    def plot_code_generator_tool(self, cols: List[str], query: str) -> str:
        """Generate prompt for matplotlib plotting code."""
        return f"""
Given DataFrame `df` with columns: {', '.join(cols)}

Write Python code using pandas **and matplotlib** (as plt) to answer: "{query}"

Rules:
1. Use pandas for data manipulation and matplotlib.pyplot (as plt) for plotting.
2. Assign the final result to a variable named `result`.
3. Create only ONE relevant plot. Set `figsize={DEFAULT_FIGSIZE}`, add title/labels.
4. Return your answer inside a single markdown fence that starts with ```python and ends with ```.
5. Handle missing values (`dropna`) before plotting.
"""

    def code_generation_agent(self, query: str, df: pd.DataFrame, chat_context: Optional[str] = None):
        """Generate code for user's query."""
        should_plot = self.query_understanding_tool(query)

        prompt = self.plot_code_generator_tool(df.columns.tolist(), query) if should_plot else self.code_writing_tool(df.columns.tolist(), query)

        context_section = f"\nConversation context:\n{chat_context}\n" if chat_context else ""

        full_prompt = f"""You are a senior Python data analyst who writes clean, efficient code.
Your response must contain ONLY a properly-closed ```python code block with no explanations. {context_section}{prompt}"""

        messages = [
            {"role": "system", "content": self.config.REASONING_FALSE},
            {"role": "user", "content": full_prompt}
        ]

        response = client.chat.completions.create(
            model=self.config.MODEL_NAME,
            messages=messages,
            temperature=self.config.CODE_GENERATION_TEMPERATURE,
            max_tokens=self.config.CODE_GENERATION_MAX_TOKENS
        )

        full_response = response.choices[0].message.content
        code = self.extract_first_code_block(full_response)
        return code, should_plot

    def execution_agent(self, code: str, df: pd.DataFrame, should_plot: bool):
        """Execute generated code."""
        env = {"pd": pd, "df": df}

        if should_plot:
            plt.rcParams["figure.dpi"] = DEFAULT_DPI
            env["plt"] = plt
            env["io"] = io

        try:
            exec(code, {}, env)
            result = env.get("result", None)

            if result is None:
                return {"error": "No result was assigned to 'result' variable"}

            # Convert plot to base64 if it's a figure
            if should_plot and isinstance(result, (plt.Figure, plt.Axes)):
                fig = result.figure if isinstance(result, plt.Axes) else result
                buf = io.BytesIO()
                fig.savefig(buf, format='png', bbox_inches='tight')
                buf.seek(0)
                img_base64 = base64.b64encode(buf.read()).decode('utf-8')
                plt.close(fig)
                return {"type": "plot", "data": img_base64}

            # Convert pandas objects to JSON
            if isinstance(result, pd.DataFrame):
                return {"type": "dataframe", "data": result.to_dict('records'), "shape": result.shape}
            elif isinstance(result, pd.Series):
                return {"type": "series", "data": result.to_dict(), "length": len(result)}
            else:
                return {"type": "value", "data": str(result)}

        except Exception as exc:
            return {"error": f"Error executing code: {exc}"}

    def reasoning_agent(self, query: str, result: Dict[str, Any]):
        """Generate reasoning about the result."""
        is_error = "error" in result

        if is_error:
            desc = result["error"]
        elif result.get("type") == "plot":
            desc = "[Plot Object]"
        else:
            desc = str(result.get("data", ""))[:MAX_RESULT_DISPLAY_LENGTH]

        if result.get("type") == "plot":
            prompt = f'The user asked: "{query}". Explain in 2-3 sentences what the chart shows.'
        else:
            prompt = f'The user asked: "{query}". The result is: {desc}. Explain in 2-3 sentences what this tells about the data.'

        response = client.chat.completions.create(
            model=self.config.MODEL_NAME,
            messages=[
                {"role": "system", "content": self.config.REASONING_TRUE},
                {"role": "user", "content": "You are an insightful data analyst. " + prompt}
            ],
            temperature=self.config.REASONING_TEMPERATURE,
            max_tokens=self.config.REASONING_MAX_TOKENS,
            stream=True
        )

        full_response = ""
        thinking_content = ""
        in_think = False

        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                token = chunk.choices[0].delta.content
                full_response += token

                if "<think>" in token:
                    in_think = True
                    token = token.split("<think>", 1)[1]
                if "</think>" in token:
                    token = token.split("</think>", 1)[0]
                    in_think = False
                if in_think or ("<think>" in full_response and not "</think>" in full_response):
                    thinking_content += token

        cleaned = re.sub(r"<think>.*?</think>", "", full_response, flags=re.DOTALL).strip()
        return thinking_content, cleaned

    def data_insight_agent(self, df: pd.DataFrame) -> str:
        """Generate insights for uploaded dataset."""
        prompt = f"""
Given a dataset with {len(df)} rows and {len(df.columns)} columns:
Columns: {', '.join(df.columns)}
Data types: {df.dtypes.to_dict()}
Missing values: {df.isnull().sum().to_dict()}

Provide:
1. A brief description of what this dataset contains
2. 3-4 possible data analysis questions that could be explored
Keep it concise and focused."""

        try:
            response = client.chat.completions.create(
                model=self.config.MODEL_NAME,
                messages=[
                    {"role": "system", "content": self.config.REASONING_FALSE},
                    {"role": "user", "content": "You are a data analyst providing brief insights. " + prompt}
                ],
                temperature=self.config.INSIGHTS_TEMPERATURE,
                max_tokens=self.config.INSIGHTS_MAX_TOKENS
            )
            return response.choices[0].message.content
        except Exception as exc:
            return f"Error generating insights: {exc}"

    def comprehensive_eda_agent(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive EDA analysis."""
        try:
            eda_results = {
                "statistical_summary": self.statistical_summary_tool(df),
                "data_profiling": self.data_profiling_tool(df),
                "correlation_analysis": self.correlation_analysis_tool(df),
                "distribution_analysis": self.distribution_analysis_tool(df)
            }
            return eda_results
        except Exception as exc:
            raise Exception(f"Error generating comprehensive EDA: {exc}")

    def statistical_summary_tool(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate statistical summary."""
        summary = {
            "basic_info": {
                "shape": df.shape,
                "memory_usage": df.memory_usage(deep=True).sum() / 1024**2,
                "dtypes": {str(k): int(v) for k, v in df.dtypes.value_counts().to_dict().items()}
            },
            "missing_data": {
                "total_missing": int(df.isnull().sum().sum()),
                "missing_percentage": float((df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100),
                "columns_with_missing": {k: int(v) for k, v in df.isnull().sum()[df.isnull().sum() > 0].to_dict().items()}
            }
        }

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            summary["numeric_summary"] = df[numeric_cols].describe().to_dict()

        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            cat_summary = {}
            for col in categorical_cols:
                cat_summary[col] = {
                    "unique_count": int(df[col].nunique()),
                    "most_frequent": str(df[col].mode().iloc[0]) if not df[col].mode().empty else None,
                    "frequency": int(df[col].value_counts().iloc[0]) if not df[col].empty else 0
                }
            summary["categorical_summary"] = cat_summary

        return summary

    def data_profiling_tool(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate data profiling information."""
        profile = {
            "data_quality": {
                "duplicate_rows": int(df.duplicated().sum()),
                "duplicate_percentage": float((df.duplicated().sum() / len(df)) * 100),
                "constant_columns": [col for col in df.columns if df[col].nunique() <= 1],
                "high_cardinality": [col for col in df.columns if df[col].nunique() > len(df) * 0.8]
            },
            "column_analysis": {}
        }

        for col in df.columns:
            col_analysis = {
                "dtype": str(df[col].dtype),
                "null_count": int(df[col].isnull().sum()),
                "null_percentage": float((df[col].isnull().sum() / len(df)) * 100),
                "unique_count": int(df[col].nunique()),
                "unique_percentage": float((df[col].nunique() / len(df)) * 100)
            }

            if df[col].dtype in ['object', 'category']:
                col_analysis["most_common"] = {str(k): int(v) for k, v in df[col].value_counts().head(3).to_dict().items()}
            elif df[col].dtype in [np.number]:
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                outliers = len(df[(df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)])
                col_analysis["outliers"] = int(outliers)

            profile["column_analysis"][col] = col_analysis

        return profile

    def correlation_analysis_tool(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate correlation analysis."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) < 2:
            return {"message": "Not enough numeric columns for correlation analysis"}

        correlation_matrix = df[numeric_cols].corr()

        high_correlations = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_val = correlation_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:
                    high_correlations.append({
                        "var1": correlation_matrix.columns[i],
                        "var2": correlation_matrix.columns[j],
                        "correlation": float(corr_val)
                    })

        return {
            "correlation_matrix": correlation_matrix.to_dict(),
            "high_correlations": high_correlations,
            "numeric_columns": list(numeric_cols)
        }

    def distribution_analysis_tool(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze distributions of numeric columns."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) == 0:
            return {"message": "No numeric columns found"}

        distributions = {}
        for col in numeric_cols:
            col_data = df[col].dropna()
            distributions[col] = {
                "skewness": float(col_data.skew()),
                "kurtosis": float(col_data.kurtosis()),
                "normality_test": "Normal" if abs(col_data.skew()) < 0.5 and abs(col_data.kurtosis()) < 0.5 else "Non-normal"
            }

        return distributions

    @staticmethod
    def extract_first_code_block(text: str) -> str:
        """Extract first Python code block from markdown."""
        start = text.find("```python")
        if start == -1:
            return ""
        start += len("```python")
        end = text.find("```", start)
        if end == -1:
            return ""
        return text[start:end].strip()
