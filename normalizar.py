class Pre_processamento:

    def __init__(self, dados):
        # Mapeie cada valor único da coluna 'moeda' para um ID numérico
        self.salary_currency_id = {salary_currency: id for id, salary_currency in enumerate(dados['salary_currency'].unique())}
        self.employee_residence_id = {employee_residence: id for id, employee_residence in enumerate(dados['employee_residence'].unique())}
        self.experience_level_id = {experience_level: id for id, experience_level in enumerate(dados['experience_level'].unique())}
        self.employment_type_id = {employment_type: id for id, employment_type in enumerate(dados['employment_type'].unique())}
        self.job_title_id = {job_title: id for id, job_title in enumerate(dados['job_title'].unique())}
        self.company_location_id = {company_location: id for id, company_location in enumerate(dados['company_location'].unique())}
        self.company_size_id = {company_size: id for id, company_size in enumerate(dados['company_size'].unique())}
        
    def substitute_str(self, dados):
        dados['salary_currency'] = dados['salary_currency'].map(self.salary_currency_id)
        dados['employee_residence'] = dados['employee_residence'].map(self.employee_residence_id)
        dados['experience_level'] = dados['experience_level'].map(self.experience_level_id)
        dados['employment_type'] = dados['employment_type'].map(self.employment_type_id)
        dados['job_title'] = dados['job_title'].map(self.job_title_id)
        dados['company_location'] = dados['company_location'].map(self.company_location_id)
        dados['company_size'] = dados['company_size'].map(self.company_size_id)
        return dados
    
    def drop_too_little_data(self, dados, col, min_val):
        job_title_counts = dados[col].value_counts()
        job_titles_to_drop = job_title_counts[job_title_counts < min_val].index
        return dados[~dados[col].isin(job_titles_to_drop)]