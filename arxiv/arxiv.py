import asyncio
import threading
import traceback
import aiohttp
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Optional, Tuple
import datetime as dt
import json
from urllib.parse import urlparse, parse_qs, urlencode


class DateRange:
    """Simple date range class"""
    def __init__(self, start: dt.datetime, end: dt.datetime):
        self.start = start
        self.end = end


class DataLabel:
    """Simple label class"""
    def __init__(self, value: str):
        self.value = value


class DataEntity:
    """Represents a data entity with content from a source"""
    def __init__(
        self,
        uri: str,
        datetime: dt.datetime,
        content: str,
        content_size_bytes: int,
        source: str = "arxiv",
        label: Optional[DataLabel] = None
    ):
        self.uri = uri
        self.datetime = datetime
        self.content = content
        self.content_size_bytes = content_size_bytes
        self.source = source
        self.label = label or DataLabel(value="arxiv")


class ValidationResult:
    """Result of a validation operation"""
    def __init__(
        self,
        is_valid: bool,
        reason: str,
        content_size_bytes_validated: int
    ):
        self.is_valid = is_valid
        self.reason = reason
        self.content_size_bytes_validated = content_size_bytes_validated


class HFValidationResult:
    """Result of a HuggingFace validation operation"""
    def __init__(
        self,
        is_valid: bool,
        validation_percentage: float,
        reason: str
    ):
        self.is_valid = is_valid
        self.validation_percentage = validation_percentage
        self.reason = reason


class ScrapeConfig:
    """Configuration for a scrape operation"""
    def __init__(
        self,
        date_range: DateRange,
        entity_limit: Optional[int] = None,
        labels: Optional[List[DataLabel]] = None
    ):
        self.date_range = date_range
        self.entity_limit = entity_limit
        self.labels = labels or []


class ArxivContent:
    """
    Represents content from arXiv.org
    """
    def __init__(
        self,
        arxiv_id: str,
        title: str,
        url: str,
        timestamp: dt.datetime,
        abstract: str,
        authors: List[str],
        categories: List[str],
        pdf_url: Optional[str] = None,
        doi: Optional[str] = None,
        journal_ref: Optional[str] = None,
        comment: Optional[str] = None,
    ):
        self.arxiv_id = arxiv_id
        self.title = title
        self.url = url
        self.timestamp = timestamp
        self.abstract = abstract
        self.authors = authors
        self.categories = categories
        self.pdf_url = pdf_url
        self.doi = doi
        self.journal_ref = journal_ref
        self.comment = comment

    def to_data_entity(self) -> DataEntity:
        """
        Converts the ArxivContent to DataEntity
        """
        content_dict = {
            "arxiv_id": self.arxiv_id,
            "title": self.title,
            "url": self.url,
            "timestamp": self.timestamp.isoformat(),
            "abstract": self.abstract,
            "authors": self.authors,
            "categories": self.categories,
            "pdf_url": self.pdf_url,
            "doi": self.doi,
            "journal_ref": self.journal_ref,
            "comment": self.comment
        }
        
        content_json = json.dumps(content_dict)
        
        # Extract a meaningful label from primary category
        label_value = self.categories[0] if self.categories else "arxiv"
            
        return DataEntity(
            uri=self.url,
            datetime=self.timestamp,
            source="arxiv",
            label=DataLabel(value=label_value),
            content=content_json,
            content_size_bytes=len(content_json)
        )

    @staticmethod
    def from_json(json_data: dict) -> 'ArxivContent':
        """
        Creates an ArxivContent object from JSON dictionary
        """
        # Parse timestamp from string to datetime
        timestamp_str = json_data.get("timestamp")
        if isinstance(timestamp_str, str):
            timestamp = dt.datetime.fromisoformat(timestamp_str)
        else:
            timestamp = dt.datetime.now(dt.timezone.utc)
        
        return ArxivContent(
            arxiv_id=json_data.get("arxiv_id"),
            title=json_data.get("title"),
            url=json_data.get("url"),
            timestamp=timestamp,
            abstract=json_data.get("abstract"),
            authors=json_data.get("authors", []),
            categories=json_data.get("categories", []),
            pdf_url=json_data.get("pdf_url"),
            doi=json_data.get("doi"),
            journal_ref=json_data.get("journal_ref"),
            comment=json_data.get("comment")
        )


class ArxivUtils:
    """
    Utility functions for processing arXiv.org data
    """
    # arXiv OAI namespaces
    ATOM_NS = {'atom': 'http://www.w3.org/2005/Atom',
               'arxiv': 'http://arxiv.org/schemas/atom',
               'opensearch': 'http://a9.com/-/spec/opensearch/1.1/'}
    
    @staticmethod
    def is_valid_arxiv_url(url: str) -> bool:
        """
        Validates if a URL is a valid arXiv.org URL
        """
        try:
            parsed_url = urlparse(url)
            return (
                parsed_url.netloc == "arxiv.org" or 
                parsed_url.netloc.endswith(".arxiv.org")
            )
        except:
            return False
    
    @staticmethod
    def normalize_url(url: str) -> str:
        """
        Normalizes an arXiv.org URL for comparison
        """
        parsed_url = urlparse(url)
        path = parsed_url.path.rstrip('/')
        
        # Reconstruct normalized URL
        normalized_url = f"{parsed_url.scheme}://{parsed_url.netloc}{path}"
        return normalized_url
    
    @staticmethod
    def extract_arxiv_id(url: str) -> str:
        """
        Extracts the arXiv ID from a URL
        """
        parsed_url = urlparse(url)
        path_parts = parsed_url.path.strip('/').split('/')
        
        # Handle different URL formats
        # Format 1: arxiv.org/abs/1234.5678
        # Format 2: arxiv.org/pdf/1234.5678.pdf
        
        if len(path_parts) >= 2:
            if path_parts[0] in ["abs", "pdf"]:
                arxiv_id = path_parts[1]
                # Remove .pdf extension if present
                if arxiv_id.endswith(".pdf"):
                    arxiv_id = arxiv_id[:-4]
                return arxiv_id
        return ""
    
    @staticmethod
    def validate_arxiv_content(actual_content: dict, entity: DataEntity) -> ValidationResult:
        """
        Validates an arXiv item against a DataEntity
        """
        try:
            # Parse entity content
            entity_content = json.loads(entity.content)
            
            # Basic validation: check if the arxiv_id matches
            if entity_content.get("arxiv_id") != actual_content.get("arxiv_id"):
                return ValidationResult(
                    is_valid=False,
                    reason="arXiv ID mismatch",
                    content_size_bytes_validated=entity.content_size_bytes
                )
                
            # Check if title matches (after normalization - whitespace, etc.)
            entity_title = entity_content.get("title", "").strip()
            actual_title = actual_content.get("title", "").strip()
            if entity_title != actual_title:
                return ValidationResult(
                    is_valid=False,
                    reason="Title mismatch",
                    content_size_bytes_validated=entity.content_size_bytes
                )
                
            # If all checks pass
            return ValidationResult(
                is_valid=True,
                reason="Content validated successfully",
                content_size_bytes_validated=entity.content_size_bytes
            )
            
        except json.JSONDecodeError:
            return ValidationResult(
                is_valid=False,
                reason="Invalid JSON content in entity",
                content_size_bytes_validated=entity.content_size_bytes
            )
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                reason=f"Validation error: {str(e)}",
                content_size_bytes_validated=entity.content_size_bytes
            )

    @staticmethod
    def parse_atom_entry(entry_element: ET.Element) -> Dict[str, Any]:
        """
        Parse an arXiv API Atom entry into a dictionary
        """
        try:
            # Extract ID (arxiv_id)
            id_element = entry_element.find('atom:id', ArxivUtils.ATOM_NS)
            id_text = id_element.text if id_element is not None else ""
            arxiv_id = id_text.split('/')[-1] if id_text else ""
            
            # Extract title
            title_element = entry_element.find('atom:title', ArxivUtils.ATOM_NS)
            title = title_element.text if title_element is not None else ""
            
            # Extract URL (abstract page)
            url = f"https://arxiv.org/abs/{arxiv_id}"
            
            # Extract PDF URL
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
            
            # Extract published date
            published_element = entry_element.find('atom:published', ArxivUtils.ATOM_NS)
            published_text = published_element.text if published_element is not None else ""
            timestamp = dt.datetime.fromisoformat(published_text) if published_text else dt.datetime.now(dt.timezone.utc)
            
            # Extract abstract
            summary_element = entry_element.find('atom:summary', ArxivUtils.ATOM_NS)
            abstract = summary_element.text if summary_element is not None else ""
            
            # Extract authors
            author_elements = entry_element.findall('atom:author', ArxivUtils.ATOM_NS)
            authors = []
            for author_element in author_elements:
                name_element = author_element.find('atom:name', ArxivUtils.ATOM_NS)
                if name_element is not None and name_element.text:
                    authors.append(name_element.text)
            
            # Extract categories (primary plus other categories)
            primary_category = entry_element.find('arxiv:primary_category', ArxivUtils.ATOM_NS)
            categories = []
            if primary_category is not None:
                primary_term = primary_category.get('term')
                if primary_term:
                    categories.append(primary_term)
            
            category_elements = entry_element.findall('atom:category', ArxivUtils.ATOM_NS)
            for category_element in category_elements:
                category_term = category_element.get('term')
                if category_term and category_term not in categories:
                    categories.append(category_term)
            
            # Extract DOI if available
            doi_element = entry_element.find(".//arxiv:doi", ArxivUtils.ATOM_NS)
            doi = doi_element.text if doi_element is not None else None
            
            # Extract journal reference if available
            journal_ref_element = entry_element.find(".//arxiv:journal_ref", ArxivUtils.ATOM_NS)
            journal_ref = journal_ref_element.text if journal_ref_element is not None else None
            
            # Extract comment if available
            comment_element = entry_element.find(".//arxiv:comment", ArxivUtils.ATOM_NS)
            comment = comment_element.text if comment_element is not None else None
            
            return {
                "arxiv_id": arxiv_id,
                "title": title,
                "url": url,
                "pdf_url": pdf_url,
                "timestamp": timestamp,
                "abstract": abstract,
                "authors": authors,
                "categories": categories,
                "doi": doi,
                "journal_ref": journal_ref,
                "comment": comment
            }
        except Exception as e:
            print(f"Error parsing arXiv entry: {str(e)}")
            return {}


class ArxivScraper:
    """
    Scrapes content from arXiv.org using the arXiv API
    """
    
    # arXiv API endpoint
    API_URL = "http://export.arxiv.org/api/query"
    
    SCRAPE_TIMEOUT_SECS = 60
    MAX_RESULTS_PER_QUERY = 100  # arXiv API limit
    
    # Semaphore to control concurrent validations
    concurrent_validates_semaphore = threading.BoundedSemaphore(10)
    
    def __init__(self, session: Optional[aiohttp.ClientSession] = None):
        """
        Initialize the scraper with an optional aiohttp session
        """
        self.session = session
        self._own_session = False
    
    async def _ensure_session(self):
        """
        Ensures an aiohttp session exists
        """
        if self.session is None:
            self.session = aiohttp.ClientSession()
            self._own_session = True
    
    async def _close_session(self):
        """
        Closes the session if it was created by this instance
        """
        if self._own_session and self.session is not None:
            await self.session.close()
            self.session = None
            self._own_session = False
    
    async def __aenter__(self):
        await self._ensure_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._close_session()
    
    async def _fetch_by_id(self, arxiv_id: str) -> Dict[str, Any]:
        """
        Fetches an arXiv paper by its ID
        """
        await self._ensure_session()
        
        params = {
            'id_list': arxiv_id,
            'max_results': 1
        }
        
        try:
            async with self.session.get(self.API_URL, params=params, timeout=30) as response:
                if response.status == 200:
                    content = await response.text()
                    root = ET.fromstring(content)
                    
                    # Find the entry element
                    entry = root.find('.//atom:entry', ArxivUtils.ATOM_NS)
                    if entry is not None:
                        return ArxivUtils.parse_atom_entry(entry)
                    else:
                        print(f"No entry found for arXiv ID: {arxiv_id}")
                        return {}
                else:
                    print(f"Failed to fetch arXiv ID {arxiv_id}: HTTP {response.status}")
                    return {}
        except Exception as e:
            print(f"Error fetching arXiv ID {arxiv_id}: {str(e)}")
            return {}
    
    async def _search_arxiv(self, 
                           search_query: str, 
                           start: int = 0, 
                           max_results: int = MAX_RESULTS_PER_QUERY,
                           sort_by: str = 'submittedDate',
                           sort_order: str = 'descending') -> List[Dict[str, Any]]:
        """
        Searches arXiv for papers matching the query
        """
        await self._ensure_session()
        
        params = {
            'search_query': search_query,
            'start': start,
            'max_results': min(max_results, self.MAX_RESULTS_PER_QUERY),
            'sortBy': sort_by,
            'sortOrder': sort_order
        }
        
        try:
            async with self.session.get(self.API_URL, params=params, timeout=30) as response:
                if response.status == 200:
                    content = await response.text()
                    root = ET.fromstring(content)
                    
                    # Find all entry elements
                    entries = root.findall('.//atom:entry', ArxivUtils.ATOM_NS)
                    results = []
                    
                    for entry in entries:
                        # Skip the first entry if it's just the OpenSearch description
                        if entry.find('atom:id', ArxivUtils.ATOM_NS) is None:
                            continue
                        
                        paper_data = ArxivUtils.parse_atom_entry(entry)
                        if paper_data:
                            results.append(paper_data)
                    
                    return results
                else:
                    print(f"Failed to search arXiv with query {search_query}: HTTP {response.status}")
                    return []
        except Exception as e:
            print(f"Error searching arXiv with query {search_query}: {str(e)}")
            return []
    
    async def validate(self, entities: List[DataEntity]) -> List[ValidationResult]:
        """
        Validate the correctness of a DataEntity by URI
        """
        
        async def validate_entity(entity) -> ValidationResult:
            if not ArxivUtils.is_valid_arxiv_url(entity.uri):
                return ValidationResult(
                    is_valid=False,
                    reason="Invalid URI. Not an arXiv.org URL.",
                    content_size_bytes_validated=entity.content_size_bytes
                )
                
            # Extract arXiv ID from the URL
            arxiv_id = ArxivUtils.extract_arxiv_id(entity.uri)
            if not arxiv_id:
                return ValidationResult(
                    is_valid=False,
                    reason="Could not extract arXiv ID from URL",
                    content_size_bytes_validated=entity.content_size_bytes
                )
            
            # Fetch the paper from arXiv API
            attempt = 0
            max_attempts = 2
            
            while attempt < max_attempts:
                # Increment attempt
                attempt += 1
                
                try:
                    # Fetch the paper details
                    paper_data = await self._fetch_by_id(arxiv_id)
                    
                    if not paper_data:
                        if attempt == max_attempts:
                            return ValidationResult(
                                is_valid=False,
                                reason=f"Could not fetch paper with arXiv ID: {arxiv_id}",
                                content_size_bytes_validated=entity.content_size_bytes
                            )
                        # Try again
                        await asyncio.sleep(1)
                        continue
                    
                    # Validate the content
                    return ArxivUtils.validate_arxiv_content(paper_data, entity)
                    
                except Exception as e:
                    if attempt == max_attempts:
                        print(f"Error validating entity {entity.uri}: {traceback.format_exc()}")
                        return ValidationResult(
                            is_valid=False,
                            reason=f"Validation error: {str(e)}",
                            content_size_bytes_validated=entity.content_size_bytes
                        )
                    # Try again
                    await asyncio.sleep(1)
            
            # Should never reach here, but just in case
            return ValidationResult(
                is_valid=False,
                reason="Unexpected validation failure",
                content_size_bytes_validated=entity.content_size_bytes
            )
        
        if not entities:
            return []
            
        # Use semaphore to limit concurrent validations
        print("Acquiring semaphore for concurrent arXiv validations.")
        
        with ArxivScraper.concurrent_validates_semaphore:
            print("Acquired semaphore for concurrent arXiv validations.")
            results = await asyncio.gather(
                *[validate_entity(entity) for entity in entities]
            )
            
        return results
    
    async def validate_hf(self, entities) -> HFValidationResult:
        """
        Validate the correctness of HFEntities by URL
        """
        
        async def validate_hf_entity(entity) -> ValidationResult:
            if not ArxivUtils.is_valid_arxiv_url(entity.get('url')):
                return ValidationResult(
                    is_valid=False,
                    reason="Invalid URI. Not an arXiv.org URL.",
                    content_size_bytes_validated=0
                )
                
            # Extract arXiv ID from the URL
            arxiv_id = ArxivUtils.extract_arxiv_id(entity.get('url'))
            if not arxiv_id:
                return ValidationResult(
                    is_valid=False,
                    reason="Could not extract arXiv ID from URL",
                    content_size_bytes_validated=0
                )
            
            # Fetch the paper from arXiv API
            attempt = 0
            max_attempts = 2
            
            while attempt < max_attempts:
                # Increment attempt
                attempt += 1
                
                try:
                    # Fetch the paper details
                    paper_data = await self._fetch_by_id(arxiv_id)
                    
                    if not paper_data:
                        if attempt == max_attempts:
                            return ValidationResult(
                                is_valid=False,
                                reason=f"Could not fetch paper with arXiv ID: {arxiv_id}",
                                content_size_bytes_validated=0
                            )
                        # Try again
                        await asyncio.sleep(1)
                        continue
                    
                    # Basic validation for HF entities
                    if entity.get('arxiv_id') != paper_data.get('arxiv_id'):
                        return ValidationResult(
                            is_valid=False,
                            reason="arXiv ID mismatch",
                            content_size_bytes_validated=0
                        )
                    
                    # If all checks pass
                    return ValidationResult(
                        is_valid=True,
                        reason="Content validated successfully",
                        content_size_bytes_validated=0
                    )
                    
                except Exception as e:
                    if attempt == max_attempts:
                        print(f"Error validating HF entity {entity.get('url')}: {traceback.format_exc()}")
                        return ValidationResult(
                            is_valid=False,
                            reason=f"Validation error: {str(e)}",
                            content_size_bytes_validated=0
                        )
                    # Try again
                    await asyncio.sleep(1)
            
            # Should never reach here, but just in case
            return ValidationResult(
                is_valid=False,
                reason="Unexpected validation failure",
                content_size_bytes_validated=0
            )
        
        # Use semaphore to limit concurrent validations
        print("Acquiring semaphore for concurrent arXiv validations.")
        
        with ArxivScraper.concurrent_validates_semaphore:
            print("Acquired semaphore for concurrent arXiv validations.")
            results = await asyncio.gather(
                *[validate_hf_entity(entity) for entity in entities]
            )
        
        # Calculate validation percentage
        valid_count = sum(1 for result in results if result.is_valid)
        valid_percent = (valid_count / len(results)) * 100 if results else 0
        
        return HFValidationResult(
            is_valid=(valid_percent >= 80),  # Adjust threshold as needed
            validation_percentage=valid_percent,
            reason=f"Validation Percentage = {valid_percent}"
        )
    
    async def scrape(self, scrape_config: ScrapeConfig) -> List[DataEntity]:
        """
        Scrapes a batch of arXiv papers according to the scrape config
        """
        await self._ensure_session()
        
        # Parse date range for arXiv query format (YYYYMMDDHHMMSS)
        start_date = scrape_config.date_range.start.strftime("%Y%m%d%H%M%S")
        end_date = scrape_config.date_range.end.strftime("%Y%m%d%H%M%S")
        
        # Build search query
        search_terms = []
        
        # Add date range constraint
        search_terms.append(f"submittedDate:[{start_date} TO {end_date}]")
        
        # Handle category/keyword filters
        if scrape_config.labels:
            label_terms = []
            for label in scrape_config.labels:
                # Check if it's a category prefix like "cat:cs.AI"
                if label.value.startswith("cat:"):
                    category = label.value[4:]  # Remove 'cat:' prefix
                    label_terms.append(f"cat:{category}")
                # Check if it's an author search like "au:lastname"
                elif label.value.startswith("au:"):
                    author = label.value[3:]  # Remove 'au:' prefix
                    label_terms.append(f"au:{author}")
                else:
                    # Treat as a general search term (title, abstract, etc.)
                    label_terms.append(f"all:{label.value}")
            
            if label_terms:
                # Join with OR if multiple terms
                if len(label_terms) > 1:
                    search_terms.append(f"({' OR '.join(label_terms)})")
                else:
                    search_terms.append(label_terms[0])
        
        # Create the final search query
        search_query = " AND ".join(search_terms)
        
        # Determine the number of papers to fetch
        max_results = scrape_config.entity_limit or 100
        
        print(f"Performing arXiv scrape for query: {search_query}")
        
        # Fetch the papers
        try:
            results = []
            remaining = max_results
            start = 0
            
            while remaining > 0:
                batch_size = min(remaining, self.MAX_RESULTS_PER_QUERY)
                batch_results = await self._search_arxiv(
                    search_query=search_query,
                    start=start,
                    max_results=batch_size,
                    sort_by='submittedDate',
                    sort_order='descending'
                )
                
                if not batch_results:
                    break
                
                results.extend(batch_results)
                
                # Update counters
                fetched = len(batch_results)
                remaining -= fetched
                start += fetched
                
                # If we got fewer results than requested, there are no more results
                if fetched < batch_size:
                    break
                
                # Be nice to the API with a small delay
                await asyncio.sleep(0.5)
            
            print(f"Completed arXiv scrape. Fetched {len(results)} papers.")
            
            # Convert to ArxivContent objects, then to DataEntities
            data_entities = []
            for paper_data in results:
                try:
                    # Create ArxivContent object
                    arxiv_content = ArxivContent(
                        arxiv_id=paper_data.get("arxiv_id"),
                        title=paper_data.get("title"),
                        url=paper_data.get("url"),
                        timestamp=paper_data.get("timestamp"),
                        abstract=paper_data.get("abstract"),
                        authors=paper_data.get("authors", []),
                        categories=paper_data.get("categories", []),
                        pdf_url=paper_data.get("pdf_url"),
                        doi=paper_data.get("doi"),
                        journal_ref=paper_data.get("journal_ref"),
                        comment=paper_data.get("comment")
                    )
                    
                    # Convert to DataEntity
                    data_entities.append(arxiv_content.to_data_entity())
                except Exception as e:
                    print(f"Failed to convert paper to DataEntity: {traceback.format_exc()}")
            
            return data_entities
            
        except Exception as e:
            print(f"Error scraping arXiv: {traceback.format_exc()}")
            return []
        finally:
            # Ensure we clean up if we're not in a context manager
            if not self._own_session:
                await self._close_session()


async def test_scrape():
    """Test function for the scraper"""
    async with ArxivScraper() as scraper:
        # Create a date range for last month
        end_date = dt.datetime.now(dt.timezone.utc)
        start_date = end_date - dt.timedelta(days=30)
        
        # Create scrape config
        config = ScrapeConfig(
            entity_limit=5,
            date_range=DateRange(start=start_date, end=end_date),
            labels=[DataLabel(value="cat:cs.AI")]  # Computer Science - Artificial Intelligence
        )
        
        # Execute the scrape
        entities = await scraper.scrape(config)
        
        # Print summary of scraped entities
        print(f"Scraped {len(entities)} entities")
        for i, entity in enumerate(entities):
            content = json.loads(entity.content)
            print(f"{i+1}. {content['title']} (arXiv:{content['arxiv_id']})")
            print(f"   Categories: {', '.join(content['categories'])}")
            print(f"   Authors: {', '.join(content['authors'][:3])}{'...' if len(content['authors']) > 3 else ''}")
            print(f"   URL: {content['url']}")
            print()
        
        return entities


async def test_validate():
    """Test function for validation"""
    async with ArxivScraper() as scraper:
        # Create sample entity to validate
        sample_entity = DataEntity(
            uri="https://arxiv.org/abs/2303.08774",
            datetime=dt.datetime(2023, 3, 15, tzinfo=dt.timezone.utc),
            source="arxiv",
            label=DataLabel(value="cs.CL"),
            content='{"arxiv_id":"2303.08774","title":"GPT-4 Technical Report","url":"https://arxiv.org/abs/2303.08774","timestamp":"2023-03-15T00:00:00+00:00","abstract":"We report the development of GPT-4, a large-scale, multimodal model which can accept image and text inputs and produce text outputs. While less capable than humans in many real-world scenarios, GPT-4 exhibits human-level performance on various professional and academic benchmarks...","authors":["OpenAI"],"categories":["cs.CL","cs.AI","cs.LG"],"pdf_url":"https://arxiv.org/pdf/2303.08774.pdf"}',
            content_size_bytes=600
        )
        
        results = await scraper.validate(entities=[sample_entity])
        
        # Print validation results
        for i, result in enumerate(results):
            print(f"Entity {i+1}: Valid = {result.is_valid}, Reason: {result.reason}")
        
        return results


# Example of how to use the scraper in a standalone script
if __name__ == "__main__":
    # Run test scrape function
    print("Testing arXiv scraper...")
    loop = asyncio.get_event_loop()
    loop.run_until_complete(test_scrape())
    
    # Uncomment to run validation test
    # print("\nTesting arXiv validator...")
    # loop.run_until_complete(test_validate())
