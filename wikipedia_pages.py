import wikipediaapi

def get_links_for_wikipedia_page(page_title):
	wiki_wiki = wikipediaapi.Wikipedia('en')
	page = wiki_wiki.page(page_title)

	if not page.exists():
		print(f'cannot find page for {page_title}')
		return

	page_names = set()

	for linkPage in page.links.values():
		try:
			page_url = linkPage.canonicalurl
		except KeyError as e:
			print(f'Encountered KeyError {e} for {linkPage}')
			continue
		print(f'Found URL: {page_url}')
		cleaned_url = page_url.replace('https://en.wikipedia.org/wiki/', '')
		print(f'Cleaned URL: {cleaned_url}')
		page_names.add(cleaned_url)

	print(f'Final URL set: {page_names}')
	return page_names

# generated using get_links_for_wikipedia_page('Cancer'')
WIKIPEDIA_PAGES_SET = {'Evolutionary_therapy', 'Financial_toxicity', 'ArXiv', 'Stomach_cancer', 'Parasitism', 'Pregnancy', 'Cachexia', 'Therapy', 'Tyrosine_kinase', 'Uterine_fibroid', 'Gene_duplication', 'Promoter_(genetics)', 'Nickel', 'Barbara_Ehrenreich', 'Immune_system', 'Benzene', 'Apoptosis', 'Vitamin_D', 'Cancer_pain', 'Hippocrates', 'CiteSeerX', 'Areca_nut', 'Gastrointestinal_disease', 'Category:Articles_needing_additional_references_from_July_2021', 'Coding_region', 'Malignancy', 'Protein', 'Whole_grain', 'Category:Articles_with_J9U_identifiers', 'Disability-adjusted_life_year', 'Vaccination', 'Perfluorooctanoic_acid', 'Sarcoma', 'Wikipedia:VideoWiki/Cancer', 'Tumor_suppressor_gene', 'Greek_language', 'Sedentary_lifestyle', 'Evolution', 'Category:Articles_with_BNF_identifiers', 'Larynx', 'Sirtuin', 'Probability', 'Hemoptysis', 'Estrogen', 'Urogenital_neoplasm', 'Cytotoxicity', 'Parenchyma', 'Ungulate', 'Defecation', 'Silicon_dioxide', 'Cell_biology', 'MSH2', 'Tumour_heterogeneity', 'Oncovirus', 'Neurological_disorder', 'Centenarian', 'BCR_(gene)', 'Bone_marrow', 'Ascites', 'Hereditary_nonpolyposis_colorectal_cancer', 'Sick_leave', 'Infectious_causes_of_cancer', 'Growth_hormone', 'Ann_Arbor_staging', 'Tobacco_smoking', 'Asbestos', 'Targeted_therapy', 'Douglas_Hanahan', 'Human_papillomavirus_infection', 'Prostate_cancer', 'CT_scan', 'Kidney_cancer', 'Cancer_syndrome', 'Malignant_transformation', 'Mastitis', 'PMS1', 'Bladder_cancer', 'Carcinoma', 'Alcohol_and_health', 'Angiogenesis', 'Radiation', 'Evidence-based_medicine', 'Breast_cancer_awareness', 'Oral_cancer', 'New_York_Daily_News', 'Mitosis', 'Tumor_microenvironment', 'Veterinary_oncology', 'List_of_distinct_cell_types_in_the_adult_human_body', 'Positron_emission_tomography', 'Cholangiocarcinoma', 'Esophagus', 'Cancer_biomarker', 'Wilhelm_Fabry', 'Cell_Metabolism', 'Cristobalite', 'Kidney', 'The_great_imitator', 'Mutation', 'Help:Referencing_for_beginners', 'Medical_diagnosis', 'Sun', 'Category:Articles_containing_potentially_dated_statements_from_2010', 'Wollastonite', 'Category:Articles_with_NARA_identifiers', 'Chimney_sweep', 'Adenoma', 'Chondropathy', 'Dysplasia', 'Healthy_diet', 'PubMed_Central', 'Hormone', 'Ali_Montazeri', 'Invasive_carcinoma_of_no_special_type', 'Altered_state_of_consciousness', 'Brain_tumor', 'Major_depressive_disorder', 'Body_mass_index', 'Germline', 'Pneumonia', 'Soft_tissue', 'Alcohol_and_cancer', 'Folate', 'Histone', 'Projectional_radiography', 'Exercise', 'Hyperplasia', 'Laser', 'Vertebral_column', 'Population_ageing', 'Low-carbohydrate_diet', 'Gleason_grading_system', 'Encyclop%C3%A6dia_Britannica_Eleventh_Edition', 'Cancer_Immunology,_Immunotherapy', 'Gene_expression', 'OCLC', 'Category:CS1_maint:_uses_authors_parameter', 'Five-year_survival_rate', 'Oxidative_phosphorylation', 'Gluten-free_diet', 'Edwin_Smith_Papyrus', 'Bronchus', 'List_of_ICD-9_codes', '2019', 'Liver_fluke', 'National_Center_for_Complementary_and_Integrative_Health', 'Cobalt', 'Cancer_signs_and_symptoms', 'Shared_decision-making_in_medicine', 'The_Spill_Canvas', 'American_Cancer_Society', 'Kaposi%27s_sarcoma-associated_herpesvirus', 'Serous_membrane', 'Victim_blaming', 'United_States_Preventive_Services_Task_Force', 'Warburg_effect_(oncology)', 'Mastectomy', 'Metastasis', 'Anticoagulant', 'Pathogen_transmission', 'Quality_of_life_(healthcare)', 'Bone_disease', 'Dysgerminoma', 'Percivall_Pott', 'Squamous-cell_carcinoma', 'Inflammation', 'International_Standard_Serial_Number', 'End-of-life_care', 'Franciscus_Sylvius', 'Radio_frequency', 'Caesarean_section', 'Leukemia', 'Clouded_leopard', 'Cancer_research', 'Paraneoplastic_syndrome', 'Cell_division', 'Invasion_(cancer)', 'Survival_of_the_fittest', 'Illness_as_Metaphor', 'Help:Authority_control', 'Antimetabolite', 'Aspirin', 'DNA_methylation', 'DNA_repair-deficiency_disorder', 'Individualism', 'Chemotherapy', 'Wikipedia:Protection_policy', 'Category:Articles_with_NDL_identifiers', 'Fecal_occult_blood', 'NBC_News', 'Wikipedia:Verifiability', 'Nonsteroidal_anti-inflammatory_drug', 'MedlinePlus', 'Template:Tumors', 'Mesothelioma', 'Coeliac_disease', 'Overnutrition', 'Endocrine_disease', 'Hormonal_therapy_(oncology)', 'Testicle', 'Anatomical_pathology', 'Endocrine_gland_neoplasm', 'Euphemism', 'Bibcode', 'Oncology_Letters', 'Ketogenic_diet', 'Canine_transmissible_venereal_tumor', 'Epstein%E2%80%93Barr_virus', 'Otology', 'Gene', 'Category:Articles_with_LNB_identifiers', 'Gina_Kolata', 'Weight_loss', 'Spontaneous_remission', 'Carnivora', 'Homologous_recombination', 'Hodgkin_lymphoma', 'Colonoscopy', 'Macmillan_Cancer_Support', 'Pickling', 'Carcinogenesis', 'Gardasil', 'Monoclonal_antibody', 'Ulcer_(dermatology)', 'BRCA2', 'Genetic_testing', 'The_Hallmarks_of_Cancer', 'Endometrium', 'Venous_thrombosis', 'Diagnosis', 'Patient_UK', 'Colectomy', 'Urologic_disease', 'War_on_cancer', 'Devil_facial_tumour_disease', 'Hospice', 'Category:Articles_needing_additional_references_from_September_2021', 'Liposarcoma', 'Heparin', 'Physical_examination', 'Scrotum', 'COX-2_inhibitor', 'Causes_of_cancer', 'Mental_disorder', 'Genetic_disorder', 'Hazard_symbol', 'Esophageal_cancer', 'Spleen', 'Bat-eared_fox', 'Helicobacter_pylori', 'History_of_cancer', 'Laser_ablation', 'Mayo_Clinic', 'Hormone_replacement_therapy', 'Richard_Nixon', 'Hematuria', 'Seminoma', 'Optimism', 'Relative_risk', 'Lymphoma', 'Carcinoma_in_situ', 'Hepatitis_C', 'First-degree_relative', 'Lymph', 'Germ_cell_tumor', 'PubMed', 'Cartilage', 'Category:Articles_containing_potentially_dated_statements_from_2018', 'Crayfish', 'International_Agency_for_Research_on_Cancer', 'Obstetric_labor_complication', 'Processed_meat', 'Primary_tumor', 'Crohn%27s_disease', 'Signs_and_symptoms', 'Engraving', 'American_Society_of_Clinical_Oncology', 'Animal_fat', 'Musculoskeletal_disorder', 'Raloxifene', 'Cancer_epigenetics', 'ICD-10', 'Help:Maintenance_template_removal', 'Medical_specialty', 'DNA_mismatch_repair', 'Finasteride', 'Public_domain', 'Clinical_trial', 'Testosterone', 'Cyst', 'Developed_country', 'Clonorchis_sinensis', 'Eye_disease', 'Google_Books', 'Progesterone', 'Category:Articles_with_LCCN_identifiers', 'Sentinel_lymph_node', 'Male_genital_disease', 'Genetics', 'Overweight', 'Radiation-induced_cancer', 'ABL_(gene)', 'Nasopharyngeal_carcinoma', 'Lower_gastrointestinal_bleeding', 'Ulcerative_colitis', 'Birth_defect', 'Mutagen', 'Nitrosamine', 'Dissection', 'Template:Disease_groups', 'Electric_power_transmission', 'DNA_virus', 'Hot_dog', 'MicroRNA', 'Ultraviolet', 'Health_effects_of_tobacco', 'Chain_reaction', 'Cochrane_(organisation)', 'Brachytherapy', 'Even-toed_ungulate', 'Liver', 'Breast_self-examination', 'Methylation', 'Bone_metastasis', 'Vitamin', 'Physical_inactivity', 'Schistosoma_haematobium', 'Neoplasm', 'Aulus_Cornelius_Celsus', 'Maternal%E2%80%93fetal_medicine', 'Cancer_prevention', 'Cancer_cell', 'Cancer_screening', 'Skin_condition', 'Tamoxifen', 'Lung_cancer', 'Live_Science', 'Cervical_cancer', 'Polyp_(medicine)', 'Urinalysis', 'Autopsy', 'Radiation_therapy', 'Fusion_protein', 'Alcohol_(drug)', 'Glass_wool', 'Cancer_immunotherapy', 'Ham', 'Colorectal_cancer', 'Thyroid_cancer', 'Medical_test', 'Endoscopy', 'Extravasation', 'Oncogene', 'Background_radiation', 'Hepatitis_B', 'Hepatitis_B_vaccine', 'Colon_cancer_staging', 'Hypoxia_(medical)', 'Lasers_in_cancer_treatment', 'Polytetrafluoroethylene', 'Resistant_starch', 'Bcr-Abl_tyrosine-kinase_inhibitor', 'Latin', 'Sigmoidoscopy', 'Kaposi%27s_sarcoma', 'Bone_tumor', 'Oral_and_maxillofacial_pathology', 'Ding-Shinn_Chen', 'Immunosenescence', 'Breast_cancer', 'Retrovirus', 'List_of_causes_of_death_by_rate', 'Chromosome', 'Cervix', 'Tumor_Biology', 'Digital_object_identifier', 'Cancer_survivor', 'Hepatoblastoma', 'Prognosis', 'Anemia', 'Blood_test', 'Diet_and_cancer', 'List_of_countries_by_cancer_rate', '5%CE%B1-Reductase_inhibitor', 'Metabolic_disorder', 'Organ_donation', 'Endocrine_system', 'Chimney_sweeps%27_carcinoma', 'Contrast_CT', 'Category:Articles_with_GND_identifiers', 'Gastrointestinal_tract', 'Template_talk:Tumors', 'Alkylating_antineoplastic_agent', 'Leiomyoma', 'Handle_System', 'Photofluorography', 'Robert_Weinberg_(biologist)', 'Pancreas', 'Estrogen_receptor', 'Postpartum_disorder', 'Glutamine', 'Magnetic_resonance_imaging', 'Combination_therapy', 'Hand_warmer', 'Connective_tissue_disease', 'Sausage', 'BRCA1', 'Familial_adenomatous_polyposis', 'Carcinogen', 'Complications_of_pregnancy', 'Immune_disorder', 'Skin_cancer', 'Cancer_survival_rates', 'DMOZ', 'Virotherapy', 'Nucleotide', 'Prostate_cancer_staging', 'Provirus', 'Acute_lymphoblastic_leukemia', 'Ovarian_cancer', 'Wayback_Machine', 'Lymphoproliferative_disorders', 'Nerve', 'Social_stigma', 'Hepatocellular_carcinoma', 'UF_Health_Shands_Cancer_Hospital', 'HMGA2', 'Angiogenesis_inhibitor', 'Carbohydrate', 'Developing_country', 'Palygorskite', 'Cancer_stem_cell', 'Nervous_system_neoplasm', 'Infection', 'DNA', 'Small-cell_carcinoma', 'MSH6', 'Susan_Domchek', 'Obesity', 'Pseudocyst', 'TNM_staging_system', 'Galen', 'Cell_potency', 'Philadelphia_chromosome', 'Medical_Subject_Headings', 'Bone', 'Benign_tumor', 'Lymphatic_disease', 'Wikisource', 'Insulin-like_growth_factor', 'Cancer_(disambiguation)', 'Immunohistochemistry', 'Hyponatremia', 'Androstanediol_glucuronide', 'HMGA1', 'COVID-19', 'Template:Cite_journal', 'Experimental_cancer_treatment', 'Index_of_oncology_articles', 'Sirtuin_6', 'Smoking', 'Mammography', 'Oncology', 'Hamartoma', 'Epithelium', 'Testicular_cancer', 'Mitotic_catastrophe', 'Aflatoxin_B1', 'Category:Use_dmy_dates_from_September_2022', 'Bruce_Alberts', 'Liver_cancer', 'Ionizing_radiation', 'Risk_factor', 'Opisthorchis_viverrini', 'Red_wolf', 'Precancerous_condition', 'Talk:Cancer', 'United_States_dollar', 'Genome_instability', 'Digestive_system_neoplasm', 'Immunotherapy', 'Lymph_node', 'Blastoma', 'ISBN', 'Lymphatic_system', 'Melanoma', 'Carcinogenic_bacteria', 'Tasmanian_devil', 'Fusion_gene', 'List_of_chemotherapeutic_agents', 'Connective_tissue', 'Organ_(biology)', 'Category:Articles_with_BNE_identifiers', 'Osteosarcoma', 'Theranostics_(journal)', 'Mineral_wool', 'Childhood_cancer', 'Chromosomal_translocation', 'World_Health_Organization', 'Biological_therapy_for_inflammatory_bowel_disease', 'Systematic_review', 'Treatment_of_cancer', 'PMS2', 'Harper%27s_Magazine', 'List_of_cancer_mortality_rates_in_the_United_States', 'Snuff_(tobacco)', 'Polycyclic_aromatic_hydrocarbon', 'Somatic_evolution_in_cancer', 'Semantic_Scholar', 'Grading_(tumors)', 'Vaginal_bleeding', 'Fever_of_unknown_origin', 'Cancer_staging', 'Breast_cancer_screening', 'Diseases_Database', 'HIV', 'Silver_bullet', 'List_of_cancer_types', 'Bacon', 'Rib', 'Campbell_De_Morgan', 'Medical_imaging', 'Heredity', 'Siddhartha_Mukherjee', 'Biopsy', 'Chemotherapy_regimen', 'Human_T-lymphotropic_virus_1', 'Hypercalcaemia', 'Passive_smoking', 'Palliative_care', 'Susan_Sontag', 'Kanger', 'MLH1', 'Tumor_metabolome', 'National_Cancer_Institute', 'Screening_(medicine)', 'Cervarix', 'Hematologic_disease', 'Oncogenomics', 'Molecular_Metabolism', 'Professional_association', 'HPV_vaccine', 'Alcohol_abuse', 'Patients%27_rights', 'Pleural_effusion', 'Ren%C3%A9_Descartes', 'Cardiovascular_disease', 'Glycolysis', 'Diet_(nutrition)', 'Cancer_and_nausea', 'Female_genital_disease', 'DNA_sequencing', 'Molecular_biology', 'Alternative_cancer_treatments', 'Mesenchyme', 'International_Classification_of_Diseases', 'Biological_immortality', 'Pancreatic_cancer', 'Red_meat', 'Breast_disease', 'Global_Burden_of_Disease_Study', 'Equal_Employment_Opportunity_Commission', 'Parasitic_disease', 'Human_sexual_activity', 'Spindle_cell_carcinoma', 'Sirtuin_3', 'Epidemiology_of_cancer', 'The_Atlantic', 'Performance_status', 'Biophysical_environment', 'Medical_guideline', 'Papilloma', 'Respiratory_disease', 'Adoptive_cell_transfer', 'Ductal_carcinoma', 'Intravasation', 'Vaccine', 'Sexually_transmitted_infection', 'Overdiagnosis', 'Non-Hodgkin_lymphoma', 'Frontiers_Media', 'Clonally_transmissible_cancer', 'Malnutrition', 'Lung', 'Laxative', 'Radon', 'Aorta', 'Psychotherapy', 'Just-world_hypothesis', 'Nicolaes_Tulp', 'Cell_growth', 'Endometrial_cancer', 'Cytogenetics', 'Head_and_neck_cancer', 'Chronic_myelogenous_leukemia', 'Epigenetics', 'Laser_coagulation', 'Tridymite', 'Syndrome'}
WIKIPEDIA_PAGES_LIST = list(WIKIPEDIA_PAGES_SET)