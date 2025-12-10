"""
AWS Login helper module - Enhanced version.
Handles AWS Portal authentication and credentials management.
"""

import json
import logging
import boto3
from typing import Dict, Optional
import urllib3
from botocore.exceptions import ClientError


logger = logging.getLogger(__name__)


class AWSPortalLoginError(Exception):
    """Raised when AWS Portal authentication fails"""
    pass


class AWSAccountIdError(Exception):
    """Raised when AWS Account ID is invalid"""
    pass


class AWSPortalClient:
    """Client for authenticating with AWS Portal and obtaining credentials"""
    
    def __init__(self, username: str, password: str):
        self.username = username
        self.password = password
        self.http = urllib3.PoolManager()
        self.token = None
        self.logger = logging.getLogger(__name__)
    
    def get_token(self) -> str:
        """
        Authenticate with AWS Portal and retrieve temporary token.
        
        Returns:
            Authentication token for credential retrieval
        """
        try:
            token_url = "https://awsportal.barcapint.com/v1/w/token"
            token_body = json.dumps({'username': self.username, 'password': self.password})
            
            token_response = self.http.request(
                "POST",
                token_url,
                body=token_body,
                headers={"Content-Type": "application/json"}
            )
            
            if token_response.status != 200:
                raise AWSPortalLoginError("Incorrect username or password")
            
            self.logger.info("Successfully obtained AWS Portal token")
            self.token = json.loads(token_response.data.decode('utf-8')).get('token')
            return self.token
            
        except Exception as e:
            self.logger.error(f"Token retrieval failed: {e}")
            raise AWSPortalLoginError(f"Failed to get token: {str(e)}")
    
    def list_accounts(self, token: str) -> list:
        """
        List all AWS accounts the user has access to.
        
        Args:
            token: Authentication token from get_token()
            
        Returns:
            List of account information
        """
        try:
            roles_url = "https://awsportal.barcapint.com/v1/creds-provider/roles?size=200"
            roles_headers = {"authorization": f"Bearer {token}"}
            
            roles_response = self.http.request(
                "GET",
                roles_url,
                headers=roles_headers
            )
            
            if roles_response.status != 200:
                raise Exception("Failed to list accounts")
            
            roles_list = json.loads(roles_response.data.decode('utf-8')).get('items', [])
            
            self.logger.info(f"Retrieved {len(roles_list)} AWS accounts")
            return roles_list
            
        except Exception as e:
            self.logger.error(f"Failed to list accounts: {e}")
            raise
    
    def fetch_sts_creds(self, token: str, account_id: str) -> Dict:
        """
        Fetch temporary STS credentials for specified AWS account.
        
        Args:
            token: Authentication token
            account_id: AWS Account ID
            
        Returns:
            Dictionary with temporary AWS credentials
        """
        try:
            # Get roles list
            roles_url = "https://awsportal.barcapint.com/v1/creds-provider/roles?size=200"
            roles_headers = {"authorization": f"Bearer {token}"}
            
            roles_response = self.http.request(
                "GET",
                roles_url,
                headers=roles_headers
            )
            
            if roles_response.status != 200:
                raise Exception("Failed to retrieve roles")
            
            roles_list = json.loads(roles_response.data.decode('utf-8')).get('items', [])
            
            # Find the account
            role_arn = None
            for role in roles_list:
                if role.get('account_id') == account_id:
                    role_arn = role.get('role_arn')
                    break
            
            if not role_arn:
                raise AWSAccountIdError(f"Account {account_id} not found or accessible")
            
            # Get credentials
            credentials_url = (
                "https://awsportal.barcapint.com/v1/creds-provider/provide-credential?"
                f"account_id={account_id}&role_arn={role_arn}"
            )
            credentials_headers = {"authorization": f"Bearer {token}"}
            
            credentials_response = self.http.request(
                "GET",
                credentials_url,
                headers=credentials_headers
            )
            
            if credentials_response.status != 200:
                raise Exception("Failed to retrieve credentials")
            
            credentials = json.loads(
                credentials_response.data.decode('utf-8')
            ).get('credentials', {})
            
            self.logger.info(f"Successfully retrieved credentials for account {account_id}")
            return credentials
            
        except Exception as e:
            self.logger.error(f"Failed to fetch STS credentials: {e}")
            raise
    
    def create_client(
        self,
        credentials: Dict,
        service: str,
        region: str
    ) -> Optional[object]:
        """
        Create boto3 client using temporary credentials.
        
        Args:
            credentials: Dictionary with AWS credentials
            service: AWS service name (e.g., 'bedrock-runtime')
            region: AWS region
            
        Returns:
            Configured boto3 client or None if error
        """
        try:
            session = boto3.Session(
                region_name=region,
                aws_access_key_id=credentials.get('AccessKeyId'),
                aws_secret_access_key=credentials.get('SecretAccessKey'),
                aws_session_token=credentials.get('SessionToken')
            )
            
            client = session.client(service)
            
            self.logger.info(f"Successfully created {service} client for region {region}")
            return client
            
        except ClientError as e:
            self.logger.error(f"Failed to create boto3 client: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error creating client: {e}")
            raise
    
    def get_bedrock_client(
        self,
        account_id: str,
        region: str = "us-west-2"
    ) -> Optional[object]:
        """
        Convenience method to get authenticated Bedrock client in one call.
        
        Args:
            account_id: AWS Account ID
            region: AWS region
            
        Returns:
            Configured boto3 Bedrock client
        """
        try:
            token = self.get_token()
            credentials = self.fetch_sts_creds(token, account_id)
            client = self.create_client(credentials, "bedrock-runtime", region)
            return client
        except Exception as e:
            self.logger.error(f"Failed to get Bedrock client: {e}")
            raise
