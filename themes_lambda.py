import warnings
warnings.filterwarnings(action = "ignore")
import boto3
import os
from botocore.config import Config
from pydantic import BaseModel, Field
from langchain_aws import ChatBedrock
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from typing import List

retry_config = Config(
                    region_name=os.environ.get("region_name"),
                    retries={
                        'max_attempts': 10,
                        'mode': 'standard'
                        }
                    )
    
session = boto3.Session()
boto3_bedrock_client = session.client(service_name='bedrock-runtime', 
                                       aws_access_key_id=os.environ.get("aws_access_key_id"),
                                       aws_secret_access_key=os.environ.get("aws_secret_access_key"),
                                       config=retry_config)
                                       
                                          
class ThemesOutput(BaseModel):
        themes_category: List[str] = Field(description="This will be list of values from the defined themes category")
        sentiment: str = Field(description="It can either be positive, negative or neutral")
 
def get_model():
    model = ChatBedrock(
        model_id = os.getenv("BEDROCK_MODEL"),
        client = boto3_bedrock_client,
        model_kwargs={
            "temperature": 0.5,
            "top_p": 0.9,
        },
    )
    return model 


def get_llm_response(model, comment, output_parser):
    prompt = PromptTemplate(
    template="""
                Given a list of classes, classify the comment into one or more of these classes. Skip any preamble text and just give the class name.

                <classes>[Issue Resolution and Follow-up, Helpfulness and Support, Issue Resolution and Repair Delays,
                Email Communication and Responsiveness, Pricing and Value for Money, Customer Service and Responsiveness Concerns,
                Design and Customization Limitations, Material and Workmanship Quality, Builder Trust and Quality,
                Home Features and Amenities, Customer Service Excellence, Overall Home Buying Experience,
                Home Features and Structural Design, Environmental and Structural Issues, Landscaping and Exterior Issues,
                Paint and Aesthetic Finishes, Construction Management and Knowledge, Communication and Follow-Up Issues,
                Sales Team Professionalism, Home Inspection and Quality Control, Sales and Marketing Experience,
                Project Management and Efficiency, Staff Communication and Friendliness, Construction Management and Scheduling,
                Purchasing Process and Closing, Home Expectations Met or Exceeded, Positive Experience with Ashton Woods,
                Appliance and System Malfunctions, Customer Feedback and Experience, Real Estate Professional Interactions,
                Material Quality and Workmanship, Warranty and Claims Handling, Warranty Issues,
                Design Studio and Options, Home Ownership and Living Experience, Smooth and Easy Process,
                Warranty Attentiveness, Sales Experience, Home Features and Design, Project Management and Oversight,
                Financial and Lending Process, Inadequate Design Consultation Process, Material and Design Quality,
                Meeting Expectations, Sales Representatives Issues, Construction Management and Support,
                Client and Realtor Negotiations, Property and Community Features, Design Studio Experience,
                Closing Process and Timeliness, Document and Information Management, Team Support and Satisfaction,
                Warranty and Service Requests, Lender Collaboration and Responsiveness, Lender Responsiveness Issues,
                Financial Services and Programs, Poor workmanship, Home Customization Options,
                Sales Representative Interaction, Lender Partner Benefits, Quality of Construction and Communication,
                Lighting and Fixture Concerns, Specific Design Aspects, Materials and Workmanship Quality,
                Low-quality materials, Studio Experience Issues, Home Upgrades and Improvement Suggestions,
                Construction Feedback, Condition of home concerns, Home Design Quality, Satisfaction with Construction,
                Unprofessional construction manager, Sales Reps Honesty and Knowledge, Home Selection and Model Options,
                Community and HOA Management, Workmanship Pride and Quality, Warranty Coverage Experience,
                Construction quality issues, Lender Referral Process, Contract and Choice Limitations,
                Sales and Design Process, Fair Pricing and Value, Responsive Construction Management,
                Construction Phase Responsiveness, Material change suggestions, Professionalism of construction workers,
                Misalignment Between Sales and Design, Sales and Design Team Coordination, Construction and Building Process,
                Subpar materials, Pre-closing issues, Construction Process Insights, Knowledge of Financing Options,
                Design Quality Feedback, Sales Office Staff Mentioned, Materials Quality Comments,
                Positive Construction Experience, Negative new home construction experience, Warranty Update Calls,
                Overall Design Satisfaction, Unaddressed Warranty Claims, Home Layout Design, Design Decisions and Limitations,
                Initial Sales Experience, High-Quality Materials and Workmanship, Pre-Closing Expectations,
                Salesperson Knowledge and Integrity, Miscommunication About Mortgage Process, Sales Representative Attentiveness,
                Market Conditions and Pricing, Communication Across Teams, Design and Sales Coordination Problems,
                Landscaping Warranty Problems, Builder and Contractor Management, Design Appeal, Clarity in Construction Timeline,
                Lying Sales Rep, Warranty Division Praise, Design and Service Quality, Comprehensive Warranty Communication,
                Cust Serv Rep Experience, Professionalism and Organizational Issues, Warranty and Post-Construction Support,
                3-D Model for Home Designs, Sales Price and Build Quality, Built Ins Addition Miscommunication,
                Flooring Warranty Claims, Sales and Management Communication, Sales and Closing Process,
                Design Selection Color Options, Quality Materials and Fan in Master Bedroom, Outside Designer Experience,
                Ashton Woods Design Information, Unprofessional Sales Department, Subcontractor Professionalism,
                Wishes to Change Design Features, Communication with Sales and Project Manager, Sales Team Performance,
                Construction and Design Selections, Sales Rep Model Home Miscommunication, Poor Sales and Customer Service,
                Warranty Process Ease, Builders and Warranty Team Performance, Construction and Flooring Options,
                Patio and Backyard Design, Sales and Project Management Ratings, Exceeding Expectations with Material,
                Structural and Design Flaws, Responsive Sales and Construction Agents, Home Design and Construction,
                Sales and Warranty Staff Professionalism, Warranty Team Members, Kitchen Materials Options,
                Sales Office Communication, Outstanding Warranty Process, Attentive Customer Service Representative,
                Design Quality Concerns, Poor Design Descriptions and Execution, Home Design Options,
                Sales and Warranty, Construction Manager Praise, Post-Sale Warranty Process, Warranty Work Responsiveness,
                Customer Service and Construction Quality, Paint and Workmanship Warranty Issues, Response Time for Warranty Issues,
                Overall Warranty and Customer Service, Design Meets Expectations, Design, Salesperson and Warranty Department,
                Sales Representative Excellence, Sales Team and Home Building, Sales and Warranty Coordination,
                Knowledgeable and Honest Construction, False Promises at Time of Sale, Material Quality in Warranty,
                Post-Closing Warranty Experience, Exceptional Warranty Service by Audilio, Warranty Service for Sliding Door,
                Air Conditioning Warranty Concerns, Project and Sales Managers, Poor Overall Sales Service,
                Customer Service and Design, Warranty Rep's Expertise, Floorplan and Design Thoughtfulness,
                Final Quality After Warranty Work, Courteous Cust Serv Rep, Effective Warranty Resolution,
                Salesperson Professionalism, Builder Warranty Service, Salesperson Recognition, Warranty Repair Timing and Management,
                Proactive Cust Serv Rep, Warranty Rating, Home Issues and Warranty Questions, Warranty Expectations Met,
                Best Cust Serv Rep, Sales Involvement in Upgrades, Construction to Warranty Process,
                Customer Service and Communication, Exterior Design Change, New Construction Home Purchase Experience,
                Initial Sales and Quality, Warranty Representative Excellence, Warranty Satisfaction,
                Warranty Team Recognition, House Floor Plan and Design, Omitted HOA Information,
                Inconsistent Trim Upgrades Information, Construction Quality by Erik, Warranty Issue Responsiveness,
                Non-involvement in Construction, Design Process Assistance, Sales Pricing Communication,
                Effective Lender Change Management, Helpful Construction Language Support, Mortgage Process and Lender Concerns,
                Building Post-Design, Closing Date Rigidity, Design Process Timing, Area and Design Satisfaction,
                Design and Quality Exceed Expectations, Post-Sale Experience, Design Studio Evaluation Needed,
                Customer Service and Design Choices, Studio Design Consultant, Construction Manager Experience,
                Construction Manager Communication, Sales and Design Package Awareness, Home Inspection Issue,
                Sales and Studio Interaction, Lender and Construction Management Suggestions, AW Collection Process,
                Design Options Suggestions, Misleading Easement Information, Design Homeowner Options,
                Salesperson Vacation Before Closing, AW Collection Satisfaction, Mortgage Lender Contract Understanding,
                Design Quality, Desire for Material Customization, Exceeding Warranty Expectations,
                Mid-Construction Purchase Experience, Professionalism in Sales and Design, Warranty Assessment,
                Pre-Sale Selections Crediting, Sales Transparency, Builder Responsiveness After Sale, Post-Sale Service,
                Misrepresentation of Home Features, Knowledgeable and Courteous Construction, Design Studio Issue Resolution,
                Salesperson's Professionalism, Design Upgrade Responsibility, Misrepresentation of Price,
                Sales Process Explanation, Efficient Warranty Concern Handling, Design Studio Assistance,
                Lender Recommendation Dissatisfaction, Lender Professionalism] </classes>
                <document>{comment}<document>
                
                Recheck the response is strictly in a json format with the following keys and nothings else:
                1. theme_category: List[str]: You are strictly prohibited to use values outside the list given in the classes. This will strictly be list of values from the defined classes, 
                2. sentiment: str You are strictly required to return either positive, negative or neutral and nothing else""",
    input_variables=["comment"],
    partial_variables={"format_instructions": output_parser.get_format_instructions()},
    )
    
    chain = prompt | model | output_parser

    resp = chain.invoke({"comment": comment})
    return resp



def lambda_handler(event, context):
    comment = event.get("comment")   
    model = get_model()
    output_parser = JsonOutputParser(pydantic_object=ThemesOutput)
    
    response = get_llm_response(model = model,
                                comment=comment,
                                output_parser = output_parser)  

    return {
        'statusCode': 200,
        'body': response
    }
